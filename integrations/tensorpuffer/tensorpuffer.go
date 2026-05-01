// Package tensorpuffer is a cgo binding around libtensorpuffer.dylib —
// the C ABI from the tensorpuffer repo's tp-cabi crate. It mirrors the
// Python ctypes wrapper used by exo's integration; same dylib, same
// content-hash domain ("tpuf-cabi-v1"), same wire format.
//
// Two integration modes for Ollama (matching the llama.cpp / vllm.rs / exo split):
//
//   - Direction A: external Go harness imports this package and wraps
//     a vanilla *llamarunner.InputCache via composition. No Ollama
//     source changes; ship as a sidecar build / standalone tool.
//   - Direction B: in-tree hooks inside runner/llamarunner/cache.go
//     and the prefill completion path, gated on TPUF_KVBM_ENABLE=1.
//
// Both share this binding. A future Linux build will use libtensorpuffer.so;
// the cgo LDFLAGS below already cover both via -ltensorpuffer.
package tensorpuffer

/*
#cgo darwin LDFLAGS: -ltensorpuffer
#cgo linux  LDFLAGS: -ltensorpuffer
#cgo CFLAGS: -I${SRCDIR}/cinclude

#include <stdlib.h>
#include <stdint.h>
#include "tensorpuffer.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"sync"
	"unsafe"
)

// ExpectedABIMajor is the major version this binding compiles against.
const ExpectedABIMajor = 1

// Errors surfaced from the dylib.
var (
	ErrNotLoaded = errors.New("tensorpuffer: dylib not loaded — set TPUF_DYLIB_PATH or build tp-cabi")
	ErrDisabled  = errors.New("tensorpuffer: TPUF_KVBM_ENABLE not set")
)

// Handle wraps a single tpuf_handle_t. Cheap to share across goroutines —
// the C ABI is internally synchronized.
type Handle struct {
	h    *C.tpuf_handle_t
	mu   sync.Mutex // serialize Free vs Stash/Load on shutdown
	once sync.Once
}

// IsEnabled returns true iff TPUF_KVBM_ENABLE=1 in the environment.
// The Init() call also requires the dylib to be linkable at build time.
func IsEnabled() bool {
	v := os.Getenv("TPUF_KVBM_ENABLE")
	return v == "1" || v == "true" || v == "TRUE" || v == "True"
}

// Init opens a handle from the standard env-var configuration
// (TPUF_S3_*, TPUF_FOYER_*, TPUF_KVBM_*). Returns nil with a wrapped
// error on init failure (auth, S3 down, foyer fd exhaustion …).
func Init() (*Handle, error) {
	abi := uint32(C.tpuf_abi_version())
	if (abi >> 16) != ExpectedABIMajor {
		return nil, fmt.Errorf(
			"tensorpuffer ABI mismatch: have %d.%d, expected %d.x",
			abi>>16, abi&0xFFFF, ExpectedABIMajor,
		)
	}
	h := C.tpuf_init_from_env()
	if h == nil {
		return nil, errors.New(lastError())
	}
	hd := &Handle{h: h}
	runtime.SetFinalizer(hd, func(h *Handle) { h.Close() })
	return hd, nil
}

// MustInit panics on failure; useful at startup when integration is required.
func MustInit() *Handle {
	hd, err := Init()
	if err != nil {
		panic(fmt.Errorf("tensorpuffer.MustInit: %w", err))
	}
	return hd
}

// Close drops the handle. Safe to call multiple times.
func (hd *Handle) Close() {
	hd.once.Do(func() {
		hd.mu.Lock()
		defer hd.mu.Unlock()
		if hd.h != nil {
			C.tpuf_free(hd.h)
			hd.h = nil
		}
	})
}

// StashPrefix stores `state` keyed by content hash of (modelID, tokens).
// Returns the number of bytes stashed (== len(state) on success).
//
// Callers may free `tokens` and `state` immediately after the call returns —
// the C ABI copies what it needs.
func (hd *Handle) StashPrefix(modelID string, tokens []uint32, state []byte) (int64, error) {
	hd.mu.Lock()
	defer hd.mu.Unlock()
	if hd.h == nil {
		return 0, errors.New("tensorpuffer: handle is closed")
	}
	cModel := C.CString(modelID)
	defer C.free(unsafe.Pointer(cModel))

	var cTokens *C.uint32_t
	if len(tokens) > 0 {
		cTokens = (*C.uint32_t)(unsafe.Pointer(&tokens[0]))
	}
	var cState *C.uint8_t
	if len(state) > 0 {
		cState = (*C.uint8_t)(unsafe.Pointer(&state[0]))
	}

	rc := C.tpuf_stash_prefix(
		hd.h,
		cModel,
		cTokens,
		C.size_t(len(tokens)),
		cState,
		C.size_t(len(state)),
	)
	if rc < 0 {
		return 0, errors.New(lastError())
	}
	return int64(rc), nil
}

// TryLoadPrefix probes the puffer for a stashed blob keyed by
// (modelID, tokens). Returns (bytes, true) on hit, (nil, false) on miss.
//
// Implements the standard "probe size, allocate, retry" dance documented
// by the C ABI's -2 return-code contract.
func (hd *Handle) TryLoadPrefix(modelID string, tokens []uint32) ([]byte, bool, error) {
	hd.mu.Lock()
	defer hd.mu.Unlock()
	if hd.h == nil {
		return nil, false, errors.New("tensorpuffer: handle is closed")
	}
	cModel := C.CString(modelID)
	defer C.free(unsafe.Pointer(cModel))

	var cTokens *C.uint32_t
	if len(tokens) > 0 {
		cTokens = (*C.uint32_t)(unsafe.Pointer(&tokens[0]))
	}

	// Probe: zero-size buffer to learn the actual blob size.
	rc := C.tpuf_try_load_prefix(
		hd.h, cModel, cTokens, C.size_t(len(tokens)),
		nil, 0,
	)
	switch {
	case rc == 0:
		return nil, false, nil // miss
	case rc == -1:
		return nil, false, errors.New(lastError())
	}
	size := int(rc)
	if size < 0 {
		size = -size // -2: required size encoded as -size
	}
	if size <= 0 {
		return nil, false, nil
	}
	out := make([]byte, size)
	rc2 := C.tpuf_try_load_prefix(
		hd.h, cModel, cTokens, C.size_t(len(tokens)),
		(*C.uint8_t)(unsafe.Pointer(&out[0])), C.size_t(size),
	)
	if rc2 <= 0 {
		// Race or unexpected; treat as miss.
		return nil, false, nil
	}
	return out[:int(rc2)], true, nil
}

// ABIVersion returns the major.minor version the loaded dylib advertises.
func ABIVersion() (uint16, uint16) {
	v := uint32(C.tpuf_abi_version())
	return uint16(v >> 16), uint16(v & 0xFFFF)
}

func lastError() string {
	p := C.tpuf_last_error()
	if p == nil {
		return "tensorpuffer: unknown error"
	}
	return C.GoString(p)
}
