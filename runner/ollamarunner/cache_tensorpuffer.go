package ollamarunner

// Direction B in-tree integration with tensorpuffer for the native
// Go runner (ollamarunner). Mirrors the llamarunner integration but
// uses kvcache.SerializableCache for the per-process state codec
// instead of llama_state_seq_get/set_data.
//
// Scope (M0): only fires when the underlying kvcache.Cache implements
// kvcache.SerializableCache (today: Causal, including SWA / chunked
// variants). Recurrent / Encoder / Wrapper caches don't yet have a
// codec — they fall through to a normal cold prefill, no harm done.
//
// Hooks:
//
//   InputCache.tpufHandle / tpufModelID — lazy-init at NewInputCache
//   InputCache.tryLoadFromPuffer        — pre-scan probe + restore
//   InputCache.StashToPuffer            — post-prefill stash
//
// The wiring in cache.go's LoadCacheSlot adds a single call at the
// top; the wiring in runner.go calls StashToPuffer after the first
// successful Decode that completes a slot's prefill.

import (
	"log/slog"
	"math"
	"os"
	"reflect"

	"github.com/ollama/ollama/integrations/tensorpuffer"
	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/model/input"
)

// IsTpufEnabled reports whether the in-tree puffer hooks should fire
// (env flag set + dylib loadable).
func IsTpufEnabled() bool {
	return tensorpuffer.IsEnabled()
}

func (c *InputCache) initTpuf() {
	if c.tpufHandle != nil {
		return
	}
	if !IsTpufEnabled() {
		return
	}
	if c.cache == nil {
		return // dummy / no-cache mode
	}
	if _, ok := c.cache.(kvcache.SerializableCache); !ok {
		slog.Info("tensorpuffer: cache type does not implement SerializableCache, skipping",
			"cache_type", reflect.TypeOf(c.cache).String())
		return
	}
	hd, err := tensorpuffer.Init()
	if err != nil {
		slog.Warn("tensorpuffer init failed (falling back)", "err", err)
		return
	}
	id := os.Getenv("TPUF_KVBM_MODEL_ID")
	if id == "" {
		id = "ollama-ollamarunner"
	}
	c.tpufHandle = hd
	c.tpufModelID = id
	slog.Info("tensorpuffer enabled for ollamarunner.InputCache", "model_id", id)
}

// tokenIDs flattens a prompt's text tokens into the uint32 list used
// as the puffer's content-hash key. Multimodal entries (multimodalHash
// set) opt out of the puffer for now — the codec doesn't yet stash
// tensor inputs.
func (c *InputCache) tokenIDs(prompt []*input.Input) ([]uint32, bool) {
	out := make([]uint32, 0, len(prompt))
	for _, p := range prompt {
		if p == nil {
			return nil, false
		}
		if p.Multimodal != nil || p.MultimodalHash != 0 {
			return nil, false
		}
		out = append(out, uint32(p.Token))
	}
	return out, true
}

// tryLoadFromPuffer probes the puffer for an exact-match prompt. On
// hit it picks a free InputCacheSlot, replaces the underlying cache's
// state with the deserialized blob, and returns (slot, remaining,
// true). The remaining slice carries the last token so the runner can
// re-decode it and produce sampler logits.
//
// Misses, unsupported shapes, and decode failures all return
// (nil, nil, false) so LoadCacheSlot falls through to its existing
// in-memory path.
func (c *InputCache) tryLoadFromPuffer(
	prompt []*input.Input, cachePrompt bool,
) (*InputCacheSlot, []*input.Input, bool) {
	if c.tpufHandle == nil || !cachePrompt || len(prompt) < 1 {
		return nil, nil, false
	}
	tokens, ok := c.tokenIDs(prompt)
	if !ok {
		return nil, nil, false
	}
	blob, hit, err := c.tpufHandle.TryLoadPrefix(c.tpufModelID, tokens)
	if err != nil {
		slog.Warn("tensorpuffer load failed (falling through)", "err", err)
		return nil, nil, false
	}
	if !hit || len(blob) == 0 {
		return nil, nil, false
	}

	sc, ok := c.cache.(kvcache.SerializableCache)
	if !ok {
		return nil, nil, false
	}

	// Pick the oldest free slot.
	var target *InputCacheSlot
	for i := range c.slots {
		s := &c.slots[i]
		if s.InUse {
			continue
		}
		if target == nil || s.lastUsed.Before(target.lastUsed) {
			target = s
		}
	}
	if target == nil {
		slog.Debug("tensorpuffer hit but no free slot — falling through")
		return nil, nil, false
	}

	// Apply the restored state. DeserializeState replaces the
	// in-memory cells/cellRanges/keys/values; the next forward pass
	// will see them. After restore, trim back to keep = len-1 so the
	// runner can re-decode the last token and produce logits — same
	// posture LoadCacheSlot's normal path takes when numPast ==
	// len(prompt).
	if err := sc.DeserializeState(blob); err != nil {
		slog.Warn("tensorpuffer DeserializeState failed (falling through)",
			"err", err, "blob_bytes", len(blob))
		return nil, nil, false
	}
	keep := int32(len(prompt) - 1)
	if keep < 0 {
		keep = 0
	}
	if err := c.cache.Remove(target.Id, keep, math.MaxInt32); err != nil {
		// Recover by clearing entirely; LoadCacheSlot's normal path
		// will reprefill. Don't claim a hit.
		_ = c.cache.Remove(target.Id, 0, math.MaxInt32)
		slog.Warn("tensorpuffer trim post-restore failed — wiped slot, falling through",
			"slot", target.Id, "err", err)
		return nil, nil, false
	}

	slog.Info("tensorpuffer hit (ollamarunner)",
		"slot", target.Id, "prompt_len", len(prompt), "keep", keep,
		"blob_bytes", len(blob))

	target.Inputs = make([]*input.Input, keep)
	copy(target.Inputs, prompt[:keep])
	target.InUse = true
	return target, prompt[keep:], true
}

// StashToPuffer captures the underlying cache's current state and PUTs
// it under content-hash of (modelID, slot.Inputs).
//
// Caller invokes this once per request, on the iteration where the
// slot's prefill has just finished (numPredicted transitions 0 → 1 in
// runner.go).
func (c *InputCache) StashToPuffer(slot *InputCacheSlot) {
	if c.tpufHandle == nil || slot == nil || c.cache == nil {
		return
	}
	sc, ok := c.cache.(kvcache.SerializableCache)
	if !ok {
		return
	}
	tokens, ok := c.tokenIDs(slot.Inputs)
	if !ok || len(tokens) == 0 {
		return
	}
	blob, err := sc.SerializeState()
	if err != nil {
		slog.Warn("tensorpuffer SerializeState failed (continuing)",
			"err", err, "slot", slot.Id)
		return
	}
	if len(blob) == 0 {
		return
	}
	n, err := c.tpufHandle.StashPrefix(c.tpufModelID, tokens, blob)
	if err != nil {
		slog.Warn("tensorpuffer stash failed (continuing)", "err", err,
			"slot", slot.Id, "tokens", len(tokens))
		return
	}
	slog.Info("tensorpuffer stashed (ollamarunner)",
		"slot", slot.Id, "tokens", len(tokens), "bytes", n)
}
