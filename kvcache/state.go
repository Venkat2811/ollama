package kvcache

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"github.com/ollama/ollama/ml"
)

// Errors surfaced by the (de)serialization helpers.
var (
	ErrSerializeUnsupported = errors.New("kvcache: cache type does not support serialization")
	ErrDeserializeMismatch  = errors.New("kvcache: serialized state shape doesn't match cache")
	ErrDeserializeMagic     = errors.New("kvcache: serialized state magic mismatch")
)

const (
	stateMagic   = "TPCV"
	stateVersion = uint32(1)
)

// SerializableCache is the optional interface a kvcache.Cache implements
// when it can dump and restore its full per-process state as bytes.
//
// Used by external integrations (e.g. tensorpuffer) to checkpoint a
// causal cache on a successful prefill and restore it on a fresh
// process — the same role llama_state_seq_get_data plays for the
// llamarunner side.
//
// Bytes are opaque to the caller and only valid against an identically-
// shaped cache (same model, same dtype, same numSequences/capacity/
// maxBatch as Init received).
//
// The cache reuses its own internal backend to construct restored
// tensors, so the caller doesn't need an ml.Context — important
// because LoadCacheSlot fires before the per-batch context exists.
type SerializableCache interface {
	// SerializeState dumps the entire cache's state (every layer's
	// keys/values + cells + cellRanges) into a self-describing byte
	// slice. Multi-sequence caches dump all sequences; the caller is
	// expected to restore into a fresh cache.
	SerializeState() ([]byte, error)

	// DeserializeState restores state previously produced by
	// SerializeState. The cache must have been Init'd with shape
	// matching the source cache; otherwise ErrDeserializeMismatch.
	DeserializeState(data []byte) error
}

// dtypeCode maps ml.DType to a stable u8 used in the wire format.
// Adding a new dtype is a wire-version bump.
func dtypeCode(d ml.DType) (uint8, error) {
	switch d {
	case ml.DTypeF32:
		return 1, nil
	case ml.DTypeF16:
		return 2, nil
	case ml.DTypeQ80:
		return 4, nil
	case ml.DTypeQ40:
		return 5, nil
	case ml.DTypeI32:
		return 6, nil
	default:
		return 0, fmt.Errorf("kvcache: unsupported dtype %v in state codec", d)
	}
}

func dtypeFromCode(code uint8) (ml.DType, error) {
	switch code {
	case 1:
		return ml.DTypeF32, nil
	case 2:
		return ml.DTypeF16, nil
	case 4:
		return ml.DTypeQ80, nil
	case 5:
		return ml.DTypeQ40, nil
	case 6:
		return ml.DTypeI32, nil
	default:
		return 0, fmt.Errorf("kvcache: unknown dtype code %d", code)
	}
}

// writeTensor appends [dtype:u8, ndim:u8, shape:u32×ndim, nbytes:u64, data:nbytes].
//
// Stores the tensor's native bytes (whatever ml.Tensor.Bytes returns
// for the cache's working dtype). Restore on the same backend that
// produced them. Cross-backend round-trip is NOT guaranteed by this
// codec — different backends may use different in-memory layouts.
//
// Pairs with the deserialize path's
// `ctx.Zeros(dtype, shape...)` + `tensor.FromBytes(...)` strategy:
// pre-allocate with a layout the runner expects (matches what
// Causal.Put used) then bytes-fill in place.
func writeTensor(buf []byte, t ml.Tensor) ([]byte, error) {
	dc, err := dtypeCode(t.DType())
	if err != nil {
		return nil, err
	}
	bytes := t.Bytes()
	shape := t.Shape()
	header := make([]byte, 2+4*len(shape)+8)
	header[0] = dc
	header[1] = uint8(len(shape))
	for i, d := range shape {
		binary.LittleEndian.PutUint32(header[2+4*i:], uint32(d))
	}
	binary.LittleEndian.PutUint64(header[2+4*len(shape):], uint64(len(bytes)))
	buf = append(buf, header...)
	buf = append(buf, bytes...)
	return buf, nil
}

// bytesToF32 reinterprets a byte slice (4 bytes per float, little-endian)
// as a []float32. Used by the deserialize path before calling FromFloats.
func bytesToF32(b []byte) []float32 {
	n := len(b) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return out
}

// f32BytesToCacheBytes converts the wire-format f32 bytes (one of two
// per-tensor blobs in the state) back into the cache's working-dtype
// byte representation, ready to be passed to Tensor.FromBytes on a
// pre-allocated Zeros'd tensor with matching shape.
//
// Splitting this from the deserialize hot loop keeps the dispatch
// table in one place — adding a new dtype (e.g. Q5_K, Q4_K_M) only
// changes this function.
func f32BytesToCacheBytes(kBytes, vBytes []byte, dtype ml.DType) (
	[]byte, []byte, error,
) {
	switch dtype {
	case ml.DTypeF32:
		return kBytes, vBytes, nil
	case ml.DTypeF16:
		return f32BytesToF16Bytes(kBytes), f32BytesToF16Bytes(vBytes), nil
	}
	// Quantized formats (Q8_0, Q4_0) round-trip the f32 -> quant
	// encoding too; defer until we actually have a model that
	// needs it.
	return nil, nil, fmt.Errorf(
		"kvcache: state codec restore for dtype %v not implemented", dtype,
	)
}

// f32BytesToF16Bytes packs a stream of f32-bytes into IEEE 754 f16
// using a fast bit-level conversion. The output buffer is half the
// length of the input.
func f32BytesToF16Bytes(in []byte) []byte {
	n := len(in) / 4
	out := make([]byte, n*2)
	for i := 0; i < n; i++ {
		bits := binary.LittleEndian.Uint32(in[i*4:])
		f16 := f32BitsToF16Bits(bits)
		binary.LittleEndian.PutUint16(out[i*2:], f16)
	}
	return out
}

// f32BitsToF16Bits implements the round-to-nearest-even conversion
// from IEEE 754 binary32 to binary16. Handles +/- inf, NaN,
// subnormals, and overflow saturation. Equivalent to the table-based
// conversions in encoding/binary's f16 helper, written inline so we
// don't pick up another dep.
func f32BitsToF16Bits(f uint32) uint16 {
	sign := uint16((f >> 31) & 0x1)
	expF := int32((f >> 23) & 0xFF)
	mantF := f & 0x007FFFFF

	switch expF {
	case 255:
		// inf or NaN
		if mantF != 0 {
			// NaN: keep at least one mantissa bit set
			return uint16((sign << 15) | 0x7C00 | uint16((mantF>>13)&0x3FF) | 0x0001)
		}
		return uint16((sign << 15) | 0x7C00)
	case 0:
		// zero or subnormal — round to f16 zero
		return uint16(sign << 15)
	}

	expH := expF - 127 + 15
	if expH >= 31 {
		// overflow → inf
		return uint16((sign << 15) | 0x7C00)
	}
	if expH <= 0 {
		// underflow / subnormal in f16 — flush to zero (same posture
		// as ggml's quick conversion)
		return uint16(sign << 15)
	}

	// round-to-nearest-even on the dropped 13 bits
	mantH := uint16(mantF >> 13)
	round := mantF & 0x00001FFF
	if round > 0x1000 || (round == 0x1000 && (mantH&1) != 0) {
		mantH++
		if mantH == 0x400 {
			mantH = 0
			expH++
			if expH >= 31 {
				return uint16((sign << 15) | 0x7C00)
			}
		}
	}
	return uint16((sign << 15) | (uint16(expH) << 10) | mantH)
}

// readTensor reads a [dtype, ndim, shape, nbytes, data] header back out.
// Returns dtype, shape, raw bytes, and the new offset.
func readTensor(data []byte, off int) (ml.DType, []int, []byte, int, error) {
	if off+2 > len(data) {
		return 0, nil, nil, 0, errors.New("kvcache: short read on tensor header")
	}
	dc := data[off]
	ndim := int(data[off+1])
	off += 2
	if off+4*ndim+8 > len(data) {
		return 0, nil, nil, 0, errors.New("kvcache: short read on tensor shape")
	}
	shape := make([]int, ndim)
	for i := 0; i < ndim; i++ {
		shape[i] = int(binary.LittleEndian.Uint32(data[off+4*i:]))
	}
	off += 4 * ndim
	nbytes := int(binary.LittleEndian.Uint64(data[off:]))
	off += 8
	if off+nbytes > len(data) {
		return 0, nil, nil, 0, errors.New("kvcache: short read on tensor body")
	}
	dtype, err := dtypeFromCode(dc)
	if err != nil {
		return 0, nil, nil, 0, err
	}
	body := data[off : off+nbytes]
	off += nbytes
	return dtype, shape, body, off, nil
}
