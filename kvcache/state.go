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
// Always written as float32 to make the wire format backend-independent.
// Backend-native byte layouts (Metal-padded F16, Q8/Q4 packed, etc.) differ
// from what `Context.FromBytes(dtype, bytes, shape)` re-allocates, so a
// raw `tensor.Bytes()` round-trip restores tensors that decode to garbage
// on Metal. Floats() forces a backend-managed conversion to dense f32 and
// FromFloats reconstructs cleanly. We pay a 2× size hit on F16 caches in
// exchange for correctness; future work could use a backend-versioned
// codec for the layout-equivalent fast path.
func writeTensor(buf []byte, t ml.Tensor) ([]byte, error) {
	// We deliberately ignore t.DType() in the wire format: serialize as
	// f32 always so the stash is portable. The receiving side hands the
	// f32 array back to FromFloats which casts to the cache's working
	// dtype.
	floats := t.Floats()
	shape := t.Shape()
	header := make([]byte, 2+4*len(shape)+8)
	header[0] = 1 // dtype f32
	header[1] = uint8(len(shape))
	for i, d := range shape {
		binary.LittleEndian.PutUint32(header[2+4*i:], uint32(d))
	}
	nbytes := len(floats) * 4
	binary.LittleEndian.PutUint64(header[2+4*len(shape):], uint64(nbytes))
	buf = append(buf, header...)
	// Append the f32 bytes via math.Float32bits. (We could
	// reinterpret-cast the slice via unsafe, but the explicit copy is
	// portable and the codec is one-shot per request.)
	for _, f := range floats {
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(f))
		buf = append(buf, b[:]...)
	}
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
