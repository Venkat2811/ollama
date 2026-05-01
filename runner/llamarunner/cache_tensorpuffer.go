package llamarunner

// Direction B in-tree integration with tensorpuffer.
//
// Hooks added in this file (no changes to existing cache.go behaviour
// when TPUF_KVBM_ENABLE is unset):
//
//   InputCache.tpufHandle        — lazy-init in NewInputCache
//   InputCache.tpufModelID       — namespace, env-overridable
//   InputCache.tryLoadFromPuffer — pre-scan probe; on hit, picks a
//                                  free slot, restores via
//                                  StateSeqSetData, returns it as if
//                                  the in-memory matcher had found a
//                                  perfect prefix.
//   InputCache.StashToPuffer     — post-prefill stash of slot state via
//                                  StateSeqGetData. Caller (runner.go)
//                                  must invoke after a successful
//                                  Decode that completes the prefill.
//
// To wire the load-side hook into LoadCacheSlot, add a single check
// at the top of that function (after the !c.multiUserCache branch):
//
//     if slot, prompt2, ok := c.tryLoadFromPuffer(prompt, cachePrompt); ok {
//         return slot, prompt2, nil
//     }
//
// To wire the stash-side hook, call c.StashToPuffer(slot) from the
// runner once a prefill batch completes (where seq.numDecoded just
// transitioned 0 → 1). Both wirings are gated by IsTpufEnabled() so
// the default behaviour is unchanged when TPUF_KVBM_ENABLE is unset.

import (
	"log/slog"
	"os"
	"reflect"

	"github.com/ollama/ollama/integrations/tensorpuffer"
)

// IsTpufEnabled reports whether the in-tree puffer hooks should fire.
// The check covers both the env flag and the dylib being available.
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
	hd, err := tensorpuffer.Init()
	if err != nil {
		slog.Warn("tensorpuffer init failed (falling back)", "err", err)
		return
	}
	id := os.Getenv("TPUF_KVBM_MODEL_ID")
	if id == "" {
		id = "ollama-llamarunner"
	}
	c.tpufHandle = hd
	c.tpufModelID = id
	slog.Info("tensorpuffer enabled for InputCache", "model_id", id)
}

// tokenIDs converts a slice of `input` records into the uint32 token
// list used as the puffer's content-hash key. Multimodal inputs (where
// `input.embed != nil`) are NOT yet supported by the codec — the
// in-memory countCommonPrefix already DeepEqual's full input records,
// so multi-modal prompts will simply miss the puffer and fall through.
func (c *InputCache) tokenIDs(prompt []input) ([]uint32, bool) {
	out := make([]uint32, 0, len(prompt))
	for i := range prompt {
		// We only stash text-only prompts. If any record carries an
		// embedding, return false to signal "skip puffer".
		if !reflect.DeepEqual(prompt[i].embed, []float32(nil)) {
			return nil, false
		}
		out = append(out, uint32(prompt[i].token))
	}
	return out, true
}

// tryLoadFromPuffer probes the puffer for an exact-match prompt. On
// hit it picks a free InputCacheSlot, restores its KV state via
// StateSeqSetData, sets the slot's Inputs to the prefix, and returns
// (slot, remaining prompt, true) — same shape LoadCacheSlot returns.
// On miss / unsupported-shape / restore failure returns (nil, nil, false).
func (c *InputCache) tryLoadFromPuffer(prompt []input, cachePrompt bool) (*InputCacheSlot, []input, bool) {
	if c.tpufHandle == nil || !cachePrompt {
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

	// Pick the oldest free slot — same posture findBestCacheSlot uses
	// when promoting a new prefix.
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

	// Clear whatever's in this slot's KV before restoring; same
	// guarantee LoadCacheSlot makes when it picks a slot.
	if c.lc != nil {
		c.lc.KvCacheSeqRm(target.Id, 0, -1)
		n := c.lc.StateSeqSetData(target.Id, blob)
		if n <= 0 {
			slog.Warn("tensorpuffer restore returned 0 tokens — falling through",
				"slot", target.Id, "blob_bytes", len(blob))
			return nil, nil, false
		}
		slog.Info("tensorpuffer hit", "slot", target.Id, "tokens", n,
			"prompt_len", len(prompt), "blob_bytes", len(blob))
	}

	// Match LoadCacheSlot's "leave one input to sample" convention.
	keep := len(prompt) - 1
	if keep < 0 {
		keep = 0
	}
	target.Inputs = make([]input, keep)
	copy(target.Inputs, prompt[:keep])
	target.InUse = true
	return target, prompt[keep:], true
}

// StashToPuffer captures the current KV state of `slot` and PUTs it
// into the puffer, keyed by content-hash of (modelID, slot.Inputs).
// Best-effort: errors are logged and swallowed so prefill correctness
// is never affected by puffer outages.
//
// Caller should invoke this immediately after a successful Decode
// that completes prefill for `slot` (i.e. the moment seq.numDecoded
// transitions 0 → 1 in runner.go's processBatch loop).
func (c *InputCache) StashToPuffer(slot *InputCacheSlot) {
	if c.tpufHandle == nil || slot == nil {
		return
	}
	if c.lc == nil {
		return // tests with a nil llama context
	}
	tokens, ok := c.tokenIDs(slot.Inputs)
	if !ok || len(tokens) == 0 {
		return
	}
	bytes := c.lc.StateSeqGetData(slot.Id)
	if len(bytes) == 0 {
		return
	}
	n, err := c.tpufHandle.StashPrefix(c.tpufModelID, tokens, bytes)
	if err != nil {
		slog.Warn("tensorpuffer stash failed (continuing)", "err", err,
			"slot", slot.Id, "tokens", len(tokens))
		return
	}
	slog.Info("tensorpuffer stashed", "slot", slot.Id,
		"tokens", len(tokens), "bytes", n)
}
