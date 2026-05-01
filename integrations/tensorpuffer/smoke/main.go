// Smoke test for the cgo binding. Run with:
//
//	docker exec tensorpuffer-minio mc rm --recursive --force --quiet local/tensorpuffer
//	rm -rf /tmp/ollama_tpuf_smoke_foyer
//	export CGO_LDFLAGS="-L${HOME}/Documents/p/venkat-github/tensorpuffer/target/release"
//	export DYLD_LIBRARY_PATH="${HOME}/Documents/p/venkat-github/tensorpuffer/target/release"
//	export TPUF_KVBM_ENABLE=1
//	export TPUF_S3_ENDPOINT=http://localhost:9100
//	export TPUF_S3_REGION=us-east-1
//	export TPUF_S3_BUCKET=tensorpuffer
//	export TPUF_S3_ACCESS_KEY=minioadmin
//	export TPUF_S3_SECRET_KEY=minioadmin
//	export TPUF_S3_FORCE_PATH_STYLE=1
//	export TPUF_KVBM_NAMESPACE=ollama-smoke
//	export TPUF_FOYER_RAM_BYTES=$((128*1024*1024))
//	export TPUF_FOYER_SSD_BYTES=$((128*1024*1024))
//	export TPUF_FOYER_SSD_DIR=/tmp/ollama_tpuf_smoke_foyer
//	export TPUF_FOYER_BLOCK_SIZE_BYTES=$((4*1024*1024))
//	go run ./integrations/tensorpuffer/smoke
//
// Asserts a stash → load round-trip plus a deliberate miss.
package main

import (
	"bytes"
	"fmt"
	"log"
	"os"

	"github.com/ollama/ollama/integrations/tensorpuffer"
)

func main() {
	if !tensorpuffer.IsEnabled() {
		fmt.Println("DISABLED: TPUF_KVBM_ENABLE=1 not set")
		os.Exit(1)
	}
	hd, err := tensorpuffer.Init()
	if err != nil {
		log.Fatalf("Init: %v", err)
	}
	defer hd.Close()

	major, minor := tensorpuffer.ABIVersion()
	fmt.Printf("ABI %d.%d\n", major, minor)

	modelID := "ollama-smoke-llama-3.2-3b"
	tokens := []uint32{1, 2, 3, 4, 5, 100, 200, 300, 1234}
	blob := bytes.Repeat([]byte("hello tensorpuffer from ollama "), 64)

	fmt.Printf("stashing %d bytes under %s / %d tokens\n", len(blob), modelID, len(tokens))
	n, err := hd.StashPrefix(modelID, tokens, blob)
	if err != nil {
		log.Fatalf("StashPrefix: %v", err)
	}
	if n != int64(len(blob)) {
		log.Fatalf("stash returned %d, expected %d", n, len(blob))
	}

	fmt.Println("loading by content hash …")
	got, hit, err := hd.TryLoadPrefix(modelID, tokens)
	if err != nil {
		log.Fatalf("TryLoadPrefix: %v", err)
	}
	if !hit {
		log.Fatal("expected hit, got miss")
	}
	if !bytes.Equal(got, blob) {
		log.Fatalf("load mismatch: got %d bytes, want %d", len(got), len(blob))
	}

	fmt.Println("miss probe (different token list) …")
	miss, hit2, err := hd.TryLoadPrefix(modelID, append(tokens, 99999))
	if err != nil {
		log.Fatalf("miss probe: %v", err)
	}
	if hit2 || miss != nil {
		log.Fatalf("expected miss, got %d bytes hit=%v", len(miss), hit2)
	}

	fmt.Println("OK")
}
