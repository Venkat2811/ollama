/* SPDX-License-Identifier: MIT
 *
 * tensorpuffer.h — C ABI for the Tensorpuffer KV store.
 *
 * Producing dylib: libtensorpuffer.dylib (macOS) / libtensorpuffer.so (Linux)
 * Built from `crates/tp-cabi` with:
 *     cargo build -p tp-cabi --release
 *
 * Engines (e.g. llama.cpp) link against this library and invoke the
 * symbols below at the prefill/decode boundary to stash and restore
 * per-sequence KV state.
 *
 * ABI version: 1.0  (see tpuf_abi_version()).
 *
 * THREADING: tpuf_handle_t is internally synchronized; multiple
 * threads may share one handle. tpuf_last_error() is thread-local.
 *
 * MEMORY: all input pointers are borrowed for the duration of the
 * call and may be freed by the caller immediately afterward. The
 * library makes its own copies. Output pointers (out_buf in
 * tpuf_load) are written into by the call.
 */

#ifndef TENSORPUFFER_H
#define TENSORPUFFER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle. Construct with tpuf_init_from_env, free with tpuf_free. */
typedef struct tpuf_handle tpuf_handle_t;

/* ABI version. Pack as (major << 16) | minor. */
uint32_t tpuf_abi_version(void);

/* Initialize a handle from environment variables.
 *
 * Required: TPUF_S3_ENDPOINT, TPUF_S3_BUCKET, TPUF_S3_ACCESS_KEY,
 *           TPUF_S3_SECRET_KEY.
 * Optional: TPUF_S3_REGION, TPUF_S3_FORCE_PATH_STYLE,
 *           TPUF_FOYER_RAM_BYTES, TPUF_FOYER_SSD_BYTES,
 *           TPUF_FOYER_SSD_DIR, TPUF_FOYER_BLOCK_SIZE_BYTES,
 *           TPUF_KVBM_S3_PREFIX, TPUF_KVBM_NAMESPACE.
 *
 * Returns NULL on error. Inspect tpuf_last_error() for diagnostics. */
tpuf_handle_t* tpuf_init_from_env(void);

/* Free a handle. Safe to pass NULL. */
void tpuf_free(tpuf_handle_t* handle);

/* PUT a per-sequence state blob keyed by content hash of
 * (model_id, token_ids).
 *
 * Returns:
 *   >= 0   bytes successfully stashed (== state_size_bytes on success)
 *   -1     error; see tpuf_last_error()
 */
int64_t tpuf_stash_prefix(
    tpuf_handle_t* handle,
    const char*    model_id,
    const uint32_t* token_ids,
    size_t          n_tokens,
    const uint8_t*  state,
    size_t          state_size_bytes
);

/* GET a per-sequence state blob by content hash of (model_id, token_ids).
 *
 * If the hash is found, copies up to out_cap bytes into out_buf and
 * returns the actual size. If the actual size exceeds out_cap, returns
 * -(actual_size) so the caller can resize and retry; out_buf contents
 * are unspecified in that case.
 *
 * Returns:
 *    > 0           bytes copied into out_buf (== actual size)
 *    0             miss (no stashed state for this hash)
 *    -2            buffer too small; abs(value) is the required size
 *    -1            error; see tpuf_last_error()
 */
int64_t tpuf_try_load_prefix(
    tpuf_handle_t* handle,
    const char*    model_id,
    const uint32_t* token_ids,
    size_t          n_tokens,
    uint8_t*        out_buf,
    size_t          out_cap
);

/* Most recent error message on the calling thread. The pointer is
 * valid until the next tpuf_* call on this thread; copy if needed.
 * Returns NULL when there is no recorded error. */
const char* tpuf_last_error(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* TENSORPUFFER_H */
