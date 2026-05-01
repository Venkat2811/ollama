package kvcache

import (
	"encoding/binary"
	"errors"
	"fmt"
	"sort"

	"github.com/ollama/ollama/ml"
)

// SerializeState — see SerializableCache.
//
// Wire layout (little-endian):
//
//	magic        "TPCV"  (4 B)
//	version      u32     (= 1)
//	dtype        u8      (the cache's working dtype)
//	num_cells    u32     (== len(c.cells))
//	num_layers   u32     (== len(c.keys))
//	swa_window   i32
//	swa_memory   i32
//	chunk_size   i32
//
//	for each cell: pos:i32, num_seqs:u32, seqs:i32×num_seqs
//	for each cellRange entry: seq:i32, min:i32, max:i32 (sentinel-encoded)
//	for each layer (sorted ascending by index):
//	    layer_idx:u32
//	    keys tensor:  [dtype:u8, ndim:u8, shape:u32×ndim, nbytes:u64, data:nbytes]
//	    values tensor:[dtype:u8, ndim:u8, shape:u32×ndim, nbytes:u64, data:nbytes]
func (c *Causal) SerializeState() ([]byte, error) {
	if len(c.keys) == 0 {
		return nil, errors.New("kvcache: empty Causal state — nothing to serialize")
	}

	dc, err := dtypeCode(c.DType)
	if err != nil {
		return nil, err
	}

	// Header
	buf := make([]byte, 0, 1024)
	buf = append(buf, []byte(stateMagic)...)
	buf = binary.LittleEndian.AppendUint32(buf, stateVersion)
	buf = append(buf, dc)
	buf = binary.LittleEndian.AppendUint32(buf, uint32(len(c.cells)))
	buf = binary.LittleEndian.AppendUint32(buf, uint32(len(c.keys)))
	buf = binary.LittleEndian.AppendUint32(buf, uint32(c.swaWindowSize))
	buf = binary.LittleEndian.AppendUint32(buf, uint32(c.swaMemorySize))
	buf = binary.LittleEndian.AppendUint32(buf, uint32(c.chunkSize))

	// cells
	for i := range c.cells {
		buf = binary.LittleEndian.AppendUint32(buf, uint32(c.cells[i].pos))
		buf = binary.LittleEndian.AppendUint32(buf, uint32(len(c.cells[i].sequences)))
		for _, s := range c.cells[i].sequences {
			buf = binary.LittleEndian.AppendUint32(buf, uint32(s))
		}
	}

	// cellRanges (deterministic order: sort by seq)
	seqIDs := make([]int, 0, len(c.cellRanges))
	for s := range c.cellRanges {
		seqIDs = append(seqIDs, s)
	}
	sort.Ints(seqIDs)
	buf = binary.LittleEndian.AppendUint32(buf, uint32(len(seqIDs)))
	for _, s := range seqIDs {
		r := c.cellRanges[s]
		buf = binary.LittleEndian.AppendUint32(buf, uint32(s))
		buf = binary.LittleEndian.AppendUint32(buf, uint32(r.min))
		buf = binary.LittleEndian.AppendUint32(buf, uint32(r.max))
	}

	// Per-layer keys + values. Sort by layer index for stable output.
	layers := make([]int, 0, len(c.keys))
	for k := range c.keys {
		layers = append(layers, k)
	}
	sort.Ints(layers)

	for _, layer := range layers {
		buf = binary.LittleEndian.AppendUint32(buf, uint32(layer))
		k := c.keys[layer]
		v, ok := c.values[layer]
		if k == nil || !ok || v == nil {
			return nil, fmt.Errorf("kvcache: layer %d missing tensor in serialize", layer)
		}
		if buf, err = writeTensor(buf, k); err != nil {
			return nil, fmt.Errorf("kvcache: write keys layer %d: %w", layer, err)
		}
		if buf, err = writeTensor(buf, v); err != nil {
			return nil, fmt.Errorf("kvcache: write values layer %d: %w", layer, err)
		}
	}

	return buf, nil
}

// DeserializeState — see SerializableCache.
//
// The receiving Causal must be Init'd with the same dtype and capacity
// as the source. Replaces all in-memory state (cells, cellRanges, and
// every layer's keys/values tensors) with the bytes encoded by
// SerializeState.
//
// The replacement tensors are constructed via ctx.FromBytes so the
// data lives in the receiving cache's backend (Metal/CUDA/CPU)
// regardless of what the original cache used. Receiver code that holds
// references to the old tensors after this call will see stale data —
// but in the InputCache integration this only runs at the start of a
// LoadCacheSlot, before any Get/Put fires.
func (c *Causal) DeserializeState(data []byte) error {
	if len(data) < 4+4+1+4+4+4+4+4 {
		return ErrDeserializeMagic
	}
	if string(data[:4]) != stateMagic {
		return ErrDeserializeMagic
	}
	off := 4
	version := binary.LittleEndian.Uint32(data[off:])
	off += 4
	if version != stateVersion {
		return fmt.Errorf("kvcache: state version %d, expected %d", version, stateVersion)
	}
	dc := data[off]
	off++
	dtype, err := dtypeFromCode(dc)
	if err != nil {
		return err
	}
	if dtype != c.DType {
		return fmt.Errorf("%w: dtype %v vs cache dtype %v", ErrDeserializeMismatch, dtype, c.DType)
	}
	numCells := int(binary.LittleEndian.Uint32(data[off:]))
	off += 4
	numLayers := int(binary.LittleEndian.Uint32(data[off:]))
	off += 4
	swaWindow := int32(binary.LittleEndian.Uint32(data[off:]))
	off += 4
	swaMemory := int32(binary.LittleEndian.Uint32(data[off:]))
	off += 4
	chunkSize := int32(binary.LittleEndian.Uint32(data[off:]))
	off += 4

	if numCells != len(c.cells) {
		return fmt.Errorf("%w: numCells %d vs %d", ErrDeserializeMismatch, numCells, len(c.cells))
	}

	c.swaWindowSize = swaWindow
	c.swaMemorySize = swaMemory
	c.chunkSize = chunkSize

	// cells
	c.cells = make([]cacheCell, numCells)
	for i := 0; i < numCells; i++ {
		if off+8 > len(data) {
			return errors.New("kvcache: short read on cells")
		}
		c.cells[i].pos = int32(binary.LittleEndian.Uint32(data[off:]))
		off += 4
		nseq := int(binary.LittleEndian.Uint32(data[off:]))
		off += 4
		c.cells[i].sequences = make([]int, nseq)
		for j := 0; j < nseq; j++ {
			if off+4 > len(data) {
				return errors.New("kvcache: short read on cell.sequences")
			}
			c.cells[i].sequences[j] = int(binary.LittleEndian.Uint32(data[off:]))
			off += 4
		}
	}

	// cellRanges
	if off+4 > len(data) {
		return errors.New("kvcache: short read on cellRanges count")
	}
	nseqs := int(binary.LittleEndian.Uint32(data[off:]))
	off += 4
	c.cellRanges = make(map[int]cellRange, nseqs)
	for i := 0; i < nseqs; i++ {
		if off+12 > len(data) {
			return errors.New("kvcache: short read on cellRanges entry")
		}
		s := int(binary.LittleEndian.Uint32(data[off:]))
		off += 4
		r := cellRange{
			min: int(binary.LittleEndian.Uint32(data[off:])),
			max: int(binary.LittleEndian.Uint32(data[off+4:])),
		}
		off += 8
		c.cellRanges[s] = r
	}

	// Per-layer tensors
	c.keys = make(map[int]ml.Tensor, numLayers)
	c.values = make(map[int]ml.Tensor, numLayers)
	for i := 0; i < numLayers; i++ {
		if off+4 > len(data) {
			return errors.New("kvcache: short read on layer index")
		}
		layer := int(binary.LittleEndian.Uint32(data[off:]))
		off += 4

		var (
			kShape, vShape []int
			kBytes, vBytes []byte
			rerr           error
		)
		// readTensor returns f32-encoded bytes regardless of the
		// original cache dtype — see writeTensor's comment for why.
		_, kShape, kBytes, off, rerr = readTensor(data, off)
		if rerr != nil {
			return fmt.Errorf("kvcache: layer %d keys: %w", layer, rerr)
		}
		_, vShape, vBytes, off, rerr = readTensor(data, off)
		if rerr != nil {
			return fmt.Errorf("kvcache: layer %d values: %w", layer, rerr)
		}

		// Always build a fresh per-layer context. Reusing c.ctxs[layer]
		// from a prior cold-path allocation collides with the new
		// tensors we're about to materialize.
		layerCtx := c.backend.NewContext().Layer(layer)
		c.ctxs[layer] = layerCtx

		kFloats := bytesToF32(kBytes)
		vFloats := bytesToF32(vBytes)
		c.keys[layer] = layerCtx.FromFloats(kFloats, kShape...)
		c.values[layer] = layerCtx.FromFloats(vFloats, vShape...)
	}

	return nil
}
