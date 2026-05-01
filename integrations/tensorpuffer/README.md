# Ollama × tensorpuffer KVBM integration

Cross-process / cross-restart KV-cache reuse for Ollama's
`llamarunner`, backed by tensorpuffer (foyer RAM+SSD + S3
write-through).

## Components

```
integrations/tensorpuffer/
  tensorpuffer.go               cgo binding around libtensorpuffer.dylib.
                                Mirrors the Python ctypes binding used by
                                exo's integration. Same dylib, same
                                content-hash domain ("tpuf-cabi-v1").
  cinclude/tensorpuffer.h       vendored header from tp-cabi.
  smoke/main.go                 stash → load round-trip + miss probe
                                via MinIO. Run with `go run`.

llama/
  llama.go                      adds StateSeqGetSize / StateSeqGetData /
                                StateSeqSetData on *Context — thin
                                wrappers around llama_state_seq_*. These
                                are the bytes the integration round-trips.

runner/llamarunner/
  cache.go                      InputCache gets two new fields
                                (tpufHandle, tpufModelID) and a tpuf
                                fast-path at the top of LoadCacheSlot.
  cache_tensorpuffer.go         all the integration glue: initTpuf(),
                                tokenIDs(), tryLoadFromPuffer(),
                                StashToPuffer().
  runner.go                     calls cache.StashToPuffer(seq.cache)
                                exactly once per request, on the
                                iteration where seq.numDecoded just
                                transitioned 0 → 1 (prefill done +
                                first sample landed).
```

## How Direction B fires

```
                                ┌──────────────────────────────────────────────┐
                                │   InputCache.LoadCacheSlot(prompt, cache=true)│
                                └────────────────────┬─────────────────────────┘
                                                     │
                          tryLoadFromPuffer(prompt, cachePrompt)
                                                     │
                       ┌─────────────────────────────┴─────────────────────────┐
                       │                                                       │
                       ▼ hit                                                   ▼ miss
       tpuf_try_load_prefix → bytes                              fall through to existing
       lc.StateSeqSetData(slot.Id, bytes)                        findLongestCacheSlot /
       slot.Inputs = prompt[:N-1]                                findBestCacheSlot path
       return (slot, prompt[N-1:], nil)



                                ┌──────────────────────────────────────────────┐
                                │   runner.go processBatch — successful Decode  │
                                └────────────────────┬─────────────────────────┘
                                                     │
                         seq.numDecoded transitions 0 → 1
                                                     │
                          cache.StashToPuffer(seq.cache)
                                                     │
                                 lc.StateSeqGetData(slot.Id) → bytes
                                                     │
                       tpuf_stash_prefix(model_id, tokens, bytes)
                                                     │
                                                     ▼
                              foyer RAM/SSD  →  S3 PUT
```

Both hooks are no-ops when `TPUF_KVBM_ENABLE` is unset — the existing
in-memory matcher is unaffected.

## Required environment

| var | required | purpose |
| :--- | :--- | :--- |
| `TPUF_KVBM_ENABLE=1` | yes | turns the in-tree hooks on |
| `TPUF_KVBM_MODEL_ID` | optional | namespace; default `ollama-llamarunner` |
| `TPUF_S3_ENDPOINT` | yes | e.g. `http://localhost:9100` for MinIO |
| `TPUF_S3_BUCKET` | yes | e.g. `tensorpuffer` |
| `TPUF_S3_ACCESS_KEY` / `_SECRET_KEY` | yes | |
| `TPUF_S3_REGION` / `_FORCE_PATH_STYLE` | optional | |
| `TPUF_KVBM_NAMESPACE` / `_S3_PREFIX` | optional | S3 key scope |
| `TPUF_FOYER_RAM_BYTES` / `_SSD_BYTES` / `_SSD_DIR` / `_BLOCK_SIZE_BYTES` | optional | foyer sizing |
| `TPUF_DYLIB_PATH` | optional | explicit dylib path (build-time linkage uses CGO_LDFLAGS) |

At build time:

```sh
# Build the dylib once
cd ${HOME}/Documents/p/venkat-github/tensorpuffer
cargo build -p tp-cabi --release

# Tell cgo where to find -ltensorpuffer
export CGO_LDFLAGS="-L${HOME}/Documents/p/venkat-github/tensorpuffer/target/release"

# At runtime macOS will need DYLD_LIBRARY_PATH to point at the same dir
export DYLD_LIBRARY_PATH="${HOME}/Documents/p/venkat-github/tensorpuffer/target/release"
```

## macOS operator caveat

`launchctl limit maxfiles` defaults to a 256-soft / unlimited-hard pair.
Foyer's SSD allocator opens many small files at init — **expect
`EMFILE (os error 24)` with the default cap**. For benches keep
`TPUF_FOYER_SSD_BYTES` ≤ 256 MB and `TPUF_FOYER_BLOCK_SIZE_BYTES`
≥ 4 MiB. For production:

```sh
sudo launchctl limit maxfiles 65536 524288
sudo reboot
```

## Direction A — external harness

Ollama's `llamarunner.InputCache` keeps `slots []InputCacheSlot`
unexported, which makes a clean external composition wrapper hard
without API exposure. Until upstream exports either the slot list or
an `InputCacheSlot` interface, the recommended Direction A demo is
the cgo smoke (`integrations/tensorpuffer/smoke/main.go`) — it proves
the C ABI ↔ Go ↔ tensorpuffer round-trip end-to-end without requiring
any Ollama state.

For full external integration (drop-in replacement for `*InputCache`
in a vendored fork), see the exo Direction A pattern at
`exo/src/exo/integrations/tensorpuffer/direction_a_harness.py` for the
shape — same composition idea, just blocked by Go's package-private
fields here.

## Smoke tests

```sh
docker run -d --name tensorpuffer-minio -p 9100:9000 minio/minio:latest server /data
docker exec tensorpuffer-minio mc alias set local http://localhost:9000 minioadmin minioadmin
docker exec tensorpuffer-minio mc mb local/tensorpuffer

# C ABI round-trip
CGO_LDFLAGS="-L${HOME}/Documents/p/venkat-github/tensorpuffer/target/release" \
DYLD_LIBRARY_PATH="${HOME}/Documents/p/venkat-github/tensorpuffer/target/release" \
TPUF_KVBM_ENABLE=1 \
TPUF_S3_ENDPOINT=http://localhost:9100 ... \
go run ./integrations/tensorpuffer/smoke
```

## Status

- [x] cgo binding around `libtensorpuffer.dylib` (M1, commit `7bc5b444`)
- [x] `llama.Context.StateSeqGet/SetData` bindings (M2, commit `6324e41e`)
- [x] Direction B in-tree hooks: load-side in `LoadCacheSlot`, stash-side
      after `Decode` (M3, commit `83ad5121`)
- [x] Build-clean with `go build ./runner/...`
- [ ] End-to-end smoke: load a real GGUF, prefill, kill, restart,
      verify warm hit. (deferred — requires a real model file in the
      test environment)
- [ ] Cross-engine compatibility note: `tpuf-cabi-v1` domain matches
      llama.cpp upstream's `tools/tensorpuffer-bench`, so an Ollama
      stash *should* be loadable by an upstream llama.cpp
      tensorpuffer-bench process and vice versa. To verify.
- [ ] Direction A as a true composition wrapper — blocked on Ollama
      API exposure of `slots` / `InputCacheSlot`.
- [ ] `ollamarunner` (the pure-Go ML path) — separate codec needed since
      KV state lives in `kvcache.Cache` not in a `*llama.Context`.

## Related

- exo side: `Venkat2811/exo:feat/tensorpuffer-kvbm` (commits `12b2eb6d`
  through `82a74a6d`).
- vllm.rs side: `vllm.rs:feat/tensorpuffer-kvbm` (commits `0f37d48`,
  `8643f2d`).
- llama.cpp side: `tp-cabi` cdylib + upstream `tools/tensorpuffer-bench`.
- Tensorpuffer architecture diagrams:
  `0_venkat-worklog/kanban/artifacts/RFC-0008/architecture-diagrams.md`.
