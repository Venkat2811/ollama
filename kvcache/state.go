package kvcache

import (
	"encoding/binary"
	"errors"
	"fmt"

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
func writeTensor(buf []byte, t ml.Tensor) ([]byte, error) {
	dc, err := dtypeCode(t.DType())
	if err != nil {
		return nil, err
	}
	shape := t.Shape()
	bytesData := t.Bytes()
	header := make([]byte, 2+4*len(shape)+8)
	header[0] = dc
	header[1] = uint8(len(shape))
	for i, d := range shape {
		binary.LittleEndian.PutUint32(header[2+4*i:], uint32(d))
	}
	binary.LittleEndian.PutUint64(header[2+4*len(shape):], uint64(len(bytesData)))
	buf = append(buf, header...)
	buf = append(buf, bytesData...)
	return buf, nil
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
