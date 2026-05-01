package kvcache

import (
	"encoding/binary"
	"fmt"
)

// SerializableCache implementation for WrapperCache.
//
// The wire format wraps each inner cache's blob with a u32 length prefix:
//
//	magic        "TPWP"   (4 B)
//	version      u32      (= 1)
//	num_caches   u32
//	for each inner cache:
//	    blob_len  u64
//	    blob      blob_len bytes (the inner cache's SerializeState output)
//
// All inner caches must implement SerializableCache; otherwise we
// return ErrSerializeUnsupported. That posture matches Direction B's
// "fall through on unsupported types" contract — the caller treats a
// failure here as "skip stash" and runs cold prefill normally.

const wrapperMagic = "TPWP"

func (c *WrapperCache) SerializeState() ([]byte, error) {
	for i, inner := range c.caches {
		if _, ok := inner.(SerializableCache); !ok {
			return nil, fmt.Errorf("%w: inner cache %d (%T) is not serializable",
				ErrSerializeUnsupported, i, inner)
		}
	}

	out := make([]byte, 0, 16)
	out = append(out, wrapperMagic...)
	out = binary.LittleEndian.AppendUint32(out, stateVersion)
	out = binary.LittleEndian.AppendUint32(out, uint32(len(c.caches)))

	for i, inner := range c.caches {
		sc := inner.(SerializableCache) // guarded above
		blob, err := sc.SerializeState()
		if err != nil {
			return nil, fmt.Errorf("kvcache.Wrapper: inner cache %d (%T) SerializeState: %w",
				i, inner, err)
		}
		out = binary.LittleEndian.AppendUint64(out, uint64(len(blob)))
		out = append(out, blob...)
	}
	return out, nil
}

func (c *WrapperCache) DeserializeState(data []byte) error {
	if len(data) < 12 || string(data[:4]) != wrapperMagic {
		return ErrDeserializeMagic
	}
	version := binary.LittleEndian.Uint32(data[4:])
	if version != stateVersion {
		return fmt.Errorf("kvcache.Wrapper: unsupported state version %d", version)
	}
	numCaches := int(binary.LittleEndian.Uint32(data[8:]))
	if numCaches != len(c.caches) {
		return fmt.Errorf("%w: serialized %d inner caches, current wrapper has %d",
			ErrDeserializeMismatch, numCaches, len(c.caches))
	}
	off := 12
	for i, inner := range c.caches {
		sc, ok := inner.(SerializableCache)
		if !ok {
			return fmt.Errorf("%w: inner cache %d (%T) is not deserializable",
				ErrSerializeUnsupported, i, inner)
		}
		if off+8 > len(data) {
			return fmt.Errorf("kvcache.Wrapper: short read on inner-blob length at idx %d", i)
		}
		blobLen := int(binary.LittleEndian.Uint64(data[off:]))
		off += 8
		if off+blobLen > len(data) {
			return fmt.Errorf("kvcache.Wrapper: short read on inner blob %d (need %d, have %d)",
				i, blobLen, len(data)-off)
		}
		if err := sc.DeserializeState(data[off : off+blobLen]); err != nil {
			return fmt.Errorf("kvcache.Wrapper: inner cache %d (%T) DeserializeState: %w",
				i, inner, err)
		}
		off += blobLen
	}
	return nil
}
