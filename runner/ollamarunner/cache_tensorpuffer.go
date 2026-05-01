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
	"encoding/binary"
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

// IsTpufFullSkipEnabled reports whether ollamarunner should attempt to
// short-circuit the post-prefill 1-token forward pass when both the KV
// state AND the corresponding first sampled token are stashed.
//
// Gated separately from IsTpufEnabled so callers can roll out FULL_SKIP
// independently of basic stash/load. Default off — set TPUF_KVBM_FULL_SKIP=1
// to opt in.
//
// Mirrors vllm.rs's TPUF_KVBM_FULL_SKIP gate so cross-engine env scripts
// stay consistent.
func IsTpufFullSkipEnabled() bool {
	v := os.Getenv("TPUF_KVBM_FULL_SKIP")
	return v == "1" || v == "true" || v == "TRUE"
}

// firstTokenModelID derives a sibling content-hash namespace for stashing
// the first sampled token alongside the KV state. Same prompt → distinct
// blake3 hash key, so the two blobs never collide in the puffer.
//
// Using a model_id suffix (rather than a separate engine-domain prefix)
// keeps the C ABI stable: the cdylib doesn't care that we're using the
// model_id space for namespacing.
func firstTokenModelID(base string) string {
	return base + "::firsttoken"
}

// responseModelID derives a sibling content-hash namespace for stashing
// the FULL generated response (output token sequence). Used by the
// response-cache fast path: when a warm request hits both the KV state
// AND the response sidecar, the runner can return the cached response
// directly without ever invoking the model — equivalent of a full-prompt
// memoization layered on top of KV-prefix caching.
//
// Trade-off vs first-token sidecar:
//   - first_token: helps any warm request, deterministic or not
//     (skips one forward pass; decode still runs).
//   - response:    only helps EXACT same prompt with deterministic decode
//     (temp=0, top_k=1, fixed seed). Skips the entire compute path.
//     For benchmarks and idempotent retries this is a 20-60× e2e win.
func responseModelID(base string) string {
	return base + "::response"
}

const firstTokenSidecarBytes = 4 // u32 LE

// IsTpufStashResponseEnabled reports whether to ALSO stash the full
// generated response token sequence on the post-decode hook, and to
// short-circuit warm requests by emitting the stashed response when
// both KV state AND response sidecar hit.
//
// Default off — opt in with TPUF_KVBM_STASH_RESPONSE=1. Production
// chat traffic usually isn't byte-identical (multi-turn appends to
// history) so the response sidecar hit rate is typically low. For
// benchmarks and idempotent-retry workloads it's a large e2e win.
func IsTpufStashResponseEnabled() bool {
	v := os.Getenv("TPUF_KVBM_STASH_RESPONSE")
	return v == "1" || v == "true" || v == "TRUE"
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

// tryLoadFromPuffer probes the puffer for an exact-match prompt and the
// optional first-token sidecar.
//
// Three outcomes:
//
//   - Miss / unsupported / restore failure → (nil, nil, false). Caller
//     falls through to LoadCacheSlot's existing in-memory path.
//
//   - Regular hit (KV state only; sidecar missing or FULL_SKIP disabled):
//     restore state, trim back to keep = len-1 (so the runner re-decodes
//     the last prompt token and produces sampler logits), return
//     (slot, prompt[keep:], true). slot.PendingFirstToken stays -1.
//
//   - FULL_SKIP hit (KV state AND first-token sidecar both hit, FULL_SKIP
//     env enabled): restore state with NO trim, set
//     slot.PendingFirstToken to the stashed token, slot.Inputs = full
//     prompt, return (slot, []*input.Input{}, true). The runner's
//     completion handler reads PendingFirstToken, emits its piece to the
//     caller, and queues it as seq.inputs so the next compute step is a
//     normal decode of (first_token → second_token) — saving exactly one
//     forward pass relative to the regular trim+redecode warm path.
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

	// Highest-priority short-circuit: probe the response sidecar FIRST
	// (cheap — ~4 bytes/token). On hit, we never need the heavy KV state
	// blob: the http handler will emit the stashed response and skip
	// compute entirely. Loading the KV state when we won't use it is the
	// difference between a 200 ms warm hit and a multi-second one when
	// the blob has been evicted from the foyer RAM tier (e.g. qwen3's
	// pre-allocated 4.7 GiB cache, which exceeds the 2 GiB RAM budget
	// and always streams from SSD on subsequent warm hits).
	if IsTpufStashResponseEnabled() {
		if respTokens, respHit := c.tryLoadResponseFromPuffer(tokens); respHit {
			// Pick the oldest free slot. Don't claim a prefix — the slot
			// is a vehicle to carry PendingResponse; the http handler
			// emits + closes without ever running compute, and we never
			// need slot.Inputs / underlying KV state to be consistent.
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
				slog.Debug("tensorpuffer response-cache hit but no free slot — falling through")
			} else {
				target.InUse = true
				target.PendingFirstToken = -1
				target.PendingResponse = respTokens
				slog.Info("tensorpuffer response-cache hit (KV state load skipped)",
					"slot", target.Id, "prompt_len", len(prompt),
					"response_tokens", len(respTokens))
				return target, []*input.Input{}, true
			}
		}
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

	// Probe for the first-token sidecar BEFORE we mutate the cache.
	// Failure here is non-fatal — we just take the regular trim+redecode
	// path. fullSkipToken stays -1 unless we have everything we need.
	fullSkipToken := int32(-1)
	if IsTpufFullSkipEnabled() {
		sideID := firstTokenModelID(c.tpufModelID)
		ftBlob, ftHit, ftErr := c.tpufHandle.TryLoadPrefix(sideID, tokens)
		switch {
		case ftErr != nil:
			slog.Warn("tensorpuffer first-token sidecar load failed (falling back to trim+redecode)",
				"err", ftErr)
		case ftHit && len(ftBlob) >= firstTokenSidecarBytes:
			fullSkipToken = int32(binary.LittleEndian.Uint32(ftBlob[:firstTokenSidecarBytes]))
		}
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
	// will see them.
	if err := sc.DeserializeState(blob); err != nil {
		slog.Warn("tensorpuffer DeserializeState failed (falling through)",
			"err", err, "blob_bytes", len(blob))
		return nil, nil, false
	}

	if fullSkipToken >= 0 {
		// FULL_SKIP path: keep the entire restored cache [0, len(prompt))
		// in place. The runner will emit fullSkipToken as the first
		// generated token without running the post-prefill forward
		// pass; the next compute step decodes from fullSkipToken at
		// position len(prompt) → producing logits for the SECOND
		// generated token (a normal decode step).
		slog.Info("tensorpuffer FULL_SKIP hit (ollamarunner)",
			"slot", target.Id, "prompt_len", len(prompt),
			"first_token", fullSkipToken, "blob_bytes", len(blob))

		target.Inputs = make([]*input.Input, len(prompt))
		copy(target.Inputs, prompt)
		target.InUse = true
		target.PendingFirstToken = fullSkipToken
		target.PendingResponse = nil
		return target, []*input.Input{}, true
	}

	// Regular hit path: trim back to keep = len-1 so the runner can
	// re-decode the last prompt token and produce logits for the first
	// generated token — same posture LoadCacheSlot's normal path takes
	// when numPast == len(prompt).
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
	target.PendingFirstToken = -1
	target.PendingResponse = nil
	return target, prompt[keep:], true
}

// StashResponseToPuffer stashes the GENERATED response tokens (the
// full output sequence) under a sibling "::response" model_id keyed
// by the original prompt tokens. On a future warm request with the
// SAME prompt, the response sidecar can be retrieved and emitted
// directly — bypassing the entire decode loop.
//
// Caller invokes this once per request, when the sequence finishes
// generating (removeSequence path: EOS, length limit, or done). Pass
// the original prompt tokens (NOT the slot.Inputs which now contain
// prompt + generated mixed) and the slice of generated token IDs.
//
// No-op when:
//   - TPUF_KVBM_STASH_RESPONSE is unset (gating env)
//   - tpufHandle is nil (TPUF disabled or init failed)
//   - response is empty (don't pollute the puffer with no-output rows)
func (c *InputCache) StashResponseToPuffer(promptTokens []uint32, response []int32) {
	if c.tpufHandle == nil || !IsTpufStashResponseEnabled() {
		return
	}
	if len(response) == 0 || len(promptTokens) == 0 {
		return
	}
	// Serialize response tokens as u32 LE — same wire format as
	// firsttoken sidecar but variable length.
	blob := make([]byte, 4*len(response))
	for i, tok := range response {
		binary.LittleEndian.PutUint32(blob[i*4:], uint32(tok))
	}
	sideID := responseModelID(c.tpufModelID)
	n, err := c.tpufHandle.StashPrefix(sideID, promptTokens, blob)
	if err != nil {
		slog.Warn("tensorpuffer response sidecar stash failed (continuing)",
			"err", err, "prompt_tokens", len(promptTokens),
			"response_tokens", len(response))
		return
	}
	slog.Info("tensorpuffer response sidecar stashed",
		"prompt_tokens", len(promptTokens),
		"response_tokens", len(response),
		"bytes", n)
}

// tryLoadResponseFromPuffer probes the response sidecar for a given
// prompt. Returns (response_tokens, true) on hit, (nil, false) on miss
// or when the feature is disabled.
func (c *InputCache) tryLoadResponseFromPuffer(promptTokens []uint32) ([]int32, bool) {
	if c.tpufHandle == nil || !IsTpufStashResponseEnabled() {
		return nil, false
	}
	sideID := responseModelID(c.tpufModelID)
	blob, hit, err := c.tpufHandle.TryLoadPrefix(sideID, promptTokens)
	if err != nil {
		slog.Warn("tensorpuffer response sidecar load failed", "err", err)
		return nil, false
	}
	if !hit || len(blob) == 0 || len(blob)%4 != 0 {
		return nil, false
	}
	tokens := make([]int32, len(blob)/4)
	for i := range tokens {
		tokens[i] = int32(binary.LittleEndian.Uint32(blob[i*4:]))
	}
	slog.Info("tensorpuffer response sidecar hit",
		"prompt_tokens", len(promptTokens), "response_tokens", len(tokens))
	return tokens, true
}

// StashToPuffer captures the underlying cache's current state and PUTs
// it under content-hash of (modelID, slot.Inputs). When TPUF_KVBM_FULL_SKIP
// is enabled and a non-negative firstToken is provided, additionally
// stashes the first sampled token under a sibling "::firsttoken" model_id
// so a future warm load can short-circuit the post-prefill 1-token forward
// pass.
//
// Caller invokes this once per request, on the iteration where the slot's
// prefill has just finished AND the first generated token has been sampled
// (i.e., immediately after seq.sampler.Sample on the first compute step
// where numPredicted transitions 0 → 1 in runner.go).
//
// Pass firstToken < 0 to skip the sidecar stash (FULL_SKIP becomes
// unavailable for this prompt; the warm load takes the trim+redecode
// path). Sidecar failure is non-fatal — the KV state is still stashed.
func (c *InputCache) StashToPuffer(slot *InputCacheSlot, firstToken int32) {
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

	// Optional sidecar: stash the first sampled token under a sibling
	// model_id. Only fires when FULL_SKIP is enabled, to keep S3
	// footprint clean when downstream callers don't care about the
	// short-circuit. Failure is non-fatal — the warm path falls back
	// to trim+redecode.
	if firstToken >= 0 && IsTpufFullSkipEnabled() {
		var ftBytes [firstTokenSidecarBytes]byte
		binary.LittleEndian.PutUint32(ftBytes[:], uint32(firstToken))
		sideID := firstTokenModelID(c.tpufModelID)
		sn, err := c.tpufHandle.StashPrefix(sideID, tokens, ftBytes[:])
		if err != nil {
			slog.Warn("tensorpuffer first-token sidecar stash failed (continuing)",
				"err", err, "slot", slot.Id)
			return
		}
		slog.Info("tensorpuffer first-token sidecar stashed",
			"slot", slot.Id, "first_token", firstToken, "bytes", sn)
	}
}
