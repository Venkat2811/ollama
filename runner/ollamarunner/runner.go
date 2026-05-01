package ollamarunner

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"hash/maphash"
	"image"
	"log"
	"log/slog"
	"net"
	"net/http"
	"os"
	"reflect"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"golang.org/x/image/bmp"
	"golang.org/x/sync/semaphore"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/ml/nn/pooling"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/runner/common"
	"github.com/ollama/ollama/sample"
	"github.com/ollama/ollama/tokenizer"

	_ "github.com/ollama/ollama/model/models"
)

// response contains a piece of generated text along with optional logprobs
type response struct {
	content  string
	logprobs []llm.Logprob
}

type Sequence struct {
	// ctxs are used for allocating tensors that last the lifetime of the sequence, such as
	// multimodal embeddings
	ctxs []ml.Context

	// mmStore holds multimodal embeddings to mange memory and enable splitting across batches
	mmStore multimodalStore

	// batch index
	iBatch int

	// prompt inputs left to evaluate
	inputs []*input.Input

	// inputs that have been added to a batch but not yet submitted to Forward
	pendingInputs []*input.Input

	// tokens that have been generated but not returned yet (e.g. for stop sequences)
	pendingResponses []string

	// logprobs for tokens that haven't been returned yet
	pendingLogprobs []llm.Logprob

	// input cache being used by this sequence
	cache *InputCacheSlot

	// channel to send responses over
	responses chan response

	// channel to stop decoding (such as if the remote connection is closed)
	quit chan bool

	// number of tokens to predict
	numPredict int

	// sampler with transforms to run on generated logits
	sampler sample.Sampler

	// channel to send back the embedding if embedding only
	embedding chan []float32

	// stop sequences
	stop []string

	// number of inputs to keep at the beginning when shifting context window
	numKeep int32

	// true if an embedding are to be returned instead of text generation
	embeddingOnly bool

	// shift if context window is exceeded
	shift bool

	doneReason llm.DoneReason

	// logprobs configuration
	logprobs    bool
	topLogprobs int

	// Metrics
	startedAt, lastUpdatedAt time.Time
	processingDuration       time.Duration
	samplingDuration         time.Duration
	numPredicted             int
	numPromptInputs          int

	// Tensorpuffer response-cache state.
	//
	// promptTokensForTpuf: snapshot of the original prompt tokens at
	// LoadCacheSlot time, used as the lookup key for the response
	// sidecar at request end. Captured separately from slot.Inputs
	// because slot.Inputs grows over time (prompt + generated tokens
	// mixed) and we need the BYTE-IDENTICAL prompt to key the stash.
	//
	// generatedTokens: appended to after each successful sample.
	// Stashed verbatim in removeSequence under the prompt key when
	// TPUF_KVBM_STASH_RESPONSE=1 is set.
	//
	// Both are non-nil only when TPUF is enabled. Cleared on Sequence
	// drop when the response is fully streamed by removeSequence.
	promptTokensForTpuf []uint32
	generatedTokens     []int32
}

type NewSequenceParams struct {
	numPredict  int
	stop        []string
	numKeep     int32
	sampler     sample.Sampler
	embedding   bool
	shift       bool
	truncate    bool
	logprobs    bool
	topLogprobs int
}

var errorInputTooLong = errors.New("the input length exceeds the context length")

func (s *Server) NewSequence(prompt string, images []llm.ImageData, params NewSequenceParams) (*Sequence, error) {
	s.ready.Wait()

	inputs, ctxs, mmStore, err := s.inputs(prompt, images)
	if err != nil {
		return nil, fmt.Errorf("failed to process inputs: %w", err)
	} else if len(inputs) == 0 {
		return nil, errors.New("no input provided")
	}

	if params.numKeep < 0 {
		params.numKeep = int32(len(inputs))
	}

	// Ensure that at least 1 input can be discarded during shift
	params.numKeep = min(params.numKeep, s.cache.numCtx-1)

	if int32(len(inputs)) > s.cache.numCtx {
		if !params.truncate {
			return nil, errorInputTooLong
		}

		discard := int32(len(inputs)) - s.cache.numCtx

		promptStart := params.numKeep + discard

		// If we need to truncate in the middle of a unbreakable batch, remove the entire batch
		sameBatch := 0
		for i, inp := range inputs {
			if sameBatch > 0 {
				sameBatch--

				if promptStart == int32(i) {
					promptStart++
				}
			} else if promptStart == int32(i) {
				break
			}

			if inp.SameBatch != 0 {
				if int32(i) < params.numKeep {
					return nil, fmt.Errorf("SameBatch may not be specified within numKeep (index: %v numKeep: %v SameBatch: %v)", i, params.numKeep, inp.SameBatch)
				}

				sameBatch = inp.SameBatch
			}
		}

		if promptStart >= int32(len(inputs)) {
			return nil, errors.New("entire prompt removed by truncation")
		}

		newInputs := inputs[:params.numKeep]
		newInputs = append(newInputs, inputs[promptStart:]...)

		slog.Warn("truncating input prompt", "limit", s.cache.numCtx, "prompt", len(inputs), "keep", params.numKeep, "new", len(newInputs))
		inputs = newInputs
	}

	// TODO(jessegross): Ingest cached history for grammar

	return &Sequence{
		ctxs:             ctxs,
		mmStore:          mmStore,
		inputs:           inputs,
		numPromptInputs:  len(inputs),
		numPredict:       params.numPredict,
		pendingResponses: make([]string, 0),
		responses:        make(chan response, 100),
		quit:             make(chan bool, 1),
		embedding:        make(chan []float32, 1),
		sampler:          params.sampler,
		embeddingOnly:    params.embedding,
		stop:             params.stop,
		numKeep:          params.numKeep,
		shift:            params.shift,
		logprobs:         params.logprobs,
		topLogprobs:      params.topLogprobs,
	}, nil
}

// calculateLogprobs converts raw logits to log probabilities and finds top K tokens
func calculateLogprobs(logits []float32, selectedToken int32, topK int, tok tokenizer.Tokenizer) []llm.Logprob {
	decoder := func(tokenID int) string {
		text, _ := tok.Decode([]int32{int32(tokenID)})
		return text
	}
	return common.CalculateLogprobs(logits, int(selectedToken), topK, decoder)
}

// inputs processes the prompt and images into a list of inputs
// by splitting the prompt on [img-<n>] tags, tokenizing text and
// decoding images
func (s *Server) inputs(prompt string, images []llm.ImageData) ([]*input.Input, []ml.Context, multimodalStore, error) {
	var inputs []*input.Input
	var ctxs []ml.Context
	var mmStore multimodalStore

	var parts []string
	var matches [][]string

	multimodalProcessor, visionModel := s.model.(model.MultimodalProcessor)

	if visionModel {
		re := regexp.MustCompile(`\[img-(\d+)\]`)
		parts = re.Split(prompt, -1)
		matches = re.FindAllStringSubmatch(prompt, -1)
		mmStore = newMultimodalStore()
	} else {
		parts = []string{prompt}
	}

	for i, part := range parts {
		// text - tokenize
		tokens, err := s.model.(tokenizer.Tokenizer).Encode(part, i == 0)
		if err != nil {
			return nil, nil, nil, err
		}

		for _, t := range tokens {
			inputs = append(inputs, &input.Input{Token: t})
		}

		// image - decode and store
		if i < len(matches) {
			n, _ := strconv.Atoi(matches[i][1])

			imageIndex := -1
			for j := range images {
				if images[j].ID == n {
					imageIndex = j
					break
				}
			}

			if imageIndex < 0 {
				return nil, nil, nil, fmt.Errorf("invalid image index: %d", n)
			}

			ctx := s.model.Backend().NewContext()
			runtime.SetFinalizer(ctx, func(c ml.Context) { c.Close() })
			ctxs = append(ctxs, ctx)
			imageEmbeddings, err := multimodalProcessor.EncodeMultimodal(ctx, images[imageIndex].Data)
			if err != nil {
				return nil, nil, nil, err
			}

			s.multimodalHash.Reset()
			_, _ = s.multimodalHash.Write(images[imageIndex].Data)
			imageHash := s.multimodalHash.Sum64()

			mmStore.addMultimodal(imageEmbeddings)

			inputs = append(inputs, &input.Input{Multimodal: imageEmbeddings, MultimodalHash: imageHash})
		}
	}

	if visionModel {
		var err error
		inputs, err = multimodalProcessor.PostTokenize(inputs)
		if err != nil {
			return nil, nil, nil, err
		}
	}

	return inputs, ctxs, mmStore, nil
}

type batchState struct {
	// id provides a counter for trace logging batches
	id int

	// ctx holds the backend context used for this batch
	ctx ml.Context

	// modelOutput holds the outputs from this batch
	modelOutput ml.Tensor

	// batchInputs holds the input token pointers which may start as
	// placeholders later filled in before calling ctx.Compute
	batchInputs []*input.Input

	// batch contains the inputs for a model forward pass
	batch input.Batch

	// full set of seqs at the time this batch was initiated
	seqs []*Sequence

	// Signaled when this batches inputs are ready and compute can proceed
	inputsReadyCh chan struct{}

	// Signaling when Compute is about to begin on this batch, and
	// seqs have been updated to prepare for the next batch
	computeStartedCh chan struct{}

	// Signaled when this batches outputs are complete and the next batch can proceed
	outputsReadyCh chan struct{}
}

type Server struct {
	// modelPath is the location of the model to be loaded
	modelPath string

	// loadMu prevents more than one load attempt from occurring at a time
	loadMu sync.Mutex

	// infoInitialized caches the result of the dummy /info backend
	// initialization. loadMu already serializes callers, so a simple cached
	// result avoids repeated dummy loads without needing sync.Once.
	infoInitialized bool
	infoModel       model.Model
	infoErr         error

	// lastLoad is the load request from the previous load attempt. Used to
	// detect if we can reuse an existing memory allocation.
	lastLoad llm.LoadRequest

	// is the server ready to process requests?
	// protects access to model and image
	ready sync.WaitGroup

	// loaded model
	model model.Model

	// status for external health reporting - loading, ready to serve, etc.
	status llm.ServerStatus

	// current progress on loading the model
	progress float32

	// number of simultaneous requests to handle
	parallel int

	// maximum number of elements in a batch (per sequence)
	// TODO (jmorganca): make this n_batch
	batchSize int

	// Simple counter used only for trace logging batches
	batchID int

	// protects access to everything below this line
	// this is context state needed for decoding
	mu sync.Mutex

	// indicates that data is ready for processing
	cond *sync.Cond

	// the list of simultaneous sequences being evaluated
	seqs []*Sequence

	// seqs can have a maximum of parallel entries, which
	// is enfoced by seqSem
	seqsSem *semaphore.Weighted

	// KV cache
	cache *InputCache

	// next sequence for prompt processing to avoid starvation
	nextSeq int

	// multimodalHash generates hashes for comparing equality
	// of non-text data
	multimodalHash maphash.Hash
}

func (s *Server) allNil() bool {
	for _, item := range s.seqs {
		if item != nil {
			return false
		}
	}
	return true
}

// applyTpufResponseCache handles the highest-priority warm-load path:
// the response sidecar hit, so we have the entire output token sequence
// from a previous cold run with the byte-identical prompt. Decode it
// once via the tokenizer, send the joined text as a single response
// chunk, mark the seq complete, and close the channel.
//
// Caller is responsible for releasing seqsSem and clearing
// slot.InUse / slot.PendingResponse — this helper just emits + closes.
//
// Returns an error only if tokenizer decode fails for the entire
// sequence; the caller treats that as "fall through to FULL_SKIP /
// normal compute" rather than failing the whole request.
//
// Net effect vs FULL_SKIP path:
//   FULL_SKIP saves one forward pass (~5-30 ms on small models, more
//   on bigger ones).
//   Response cache saves the ENTIRE decode loop — N forward passes
//   for an N-token response, which is the dominant share of e2e for
//   short prompts + long answers.
func (s *Server) applyTpufResponseCache(seq *Sequence, response []int32) error {
	if len(response) == 0 {
		return errors.New("response cache empty")
	}
	piece, err := s.model.(tokenizer.Tokenizer).Decode(response)
	if err != nil {
		return fmt.Errorf("tokenizer decode response: %w", err)
	}

	now := time.Now()
	seq.startedAt = now
	seq.lastUpdatedAt = now
	// Attribute zero compute time to prefill + decode — this is a pure
	// memoization hit. The puffer load + decode wall time shows up in
	// the http handler's outer timing.
	seq.processingDuration = 0
	seq.samplingDuration = 0
	seq.numPredicted = len(response)
	seq.doneReason = llm.DoneReasonStop

	// Single chunk via the existing pendingResponses pipeline so the
	// stop-string detection logic still applies (idempotent on a
	// completed response: if the stashed text already ended at a stop,
	// it stays stopped).
	seq.pendingResponses = append(seq.pendingResponses, piece)
	flushPending(seq)

	close(seq.responses)
	close(seq.embedding)
	return nil
}

// applyTpufFullSkip handles the warm-load fast-path where tensorpuffer
// dual-hit (KV state + first-token sidecar). The cache slot is already
// fully populated for the entire prompt — no trim was applied. This
// helper:
//
//   - Decodes the stashed first token to a piece, pushes it to
//     pendingResponses, and flushes so the http handler can stream it
//     immediately (the handler hasn't entered its response loop yet, but
//     the response channel is buffered to 100 so the send doesn't block).
//   - Marks numPredicted=1 so the next compute iteration treats this seq
//     as already past prefill (no-op-ing the post-prefill stash hook).
//   - Queues first_token as seq.inputs[0] so the next forward pass
//     operates at position len(prompt) — a normal decode of
//     (first_token → second_token).
//
// Caller must have verified token is NOT EOS (EOS first_token is handled
// inline at the LoadCacheSlot call site so no compute is scheduled).
//
// Net effect vs the regular trim+redecode warm path: saves exactly one
// forward pass (the trim's 1-token re-decode that produces first_token's
// logits). Mirrors vllm.rs's FULL_SKIP semantics in spirit, though the
// magnitude differs: vllm.rs's TRY_LOAD-only mode runs a full prefill
// kernel even with cached blocks (so its FULL_SKIP saves a much heavier
// pass), whereas ollamarunner's regular warm path already runs only a
// 1-token forward.
func (s *Server) applyTpufFullSkip(seq *Sequence, token int32) {
	seq.numPredicted = 1
	now := time.Now()
	seq.startedAt = now
	seq.lastUpdatedAt = now
	// FULL_SKIP attributes zero compute time to prefill. The puffer
	// load + state restore wall time is captured by the http handler's
	// outer timing.
	seq.processingDuration = 0

	piece, err := s.model.(tokenizer.Tokenizer).Decode([]int32{token})
	if err != nil {
		slog.Warn("tensorpuffer FULL_SKIP first_token decode failed (continuing without piece)",
			"err", err, "token", token)
	} else {
		seq.pendingResponses = append(seq.pendingResponses, piece)
		flushPending(seq)
	}

	// Queue first_token as the next decode input. The next compute
	// step runs Forward on this token at position len(prompt),
	// producing logits for the second generated token.
	seq.inputs = []*input.Input{{Token: token}}

	slog.Info("tensorpuffer FULL_SKIP applied", "slot", seq.cache.Id,
		"first_token", token)
}

func flushPending(seq *Sequence) bool {
	joined := strings.Join(seq.pendingResponses, "")
	logprobs := seq.pendingLogprobs
	seq.pendingResponses = []string{}
	seq.pendingLogprobs = []llm.Logprob{}

	// Check if there are any partial UTF-8 characters remaining.
	// We already check and queue as we are generating but some may
	// still make it here:
	// - Sequence is ending, e.g. generation limit has been hit
	// - Invalid characters in the middle of a string
	// This is a stricter check to ensure we never output invalid Unicode.
	for !utf8.ValidString(joined) {
		joined = joined[:len(joined)-1]
	}

	if len(joined) == 0 {
		return true
	}

	select {
	case seq.responses <- response{content: joined, logprobs: logprobs}:
		return true
	case <-seq.quit:
		return false
	}
}

func (s *Server) removeSequence(seqIndex int, reason llm.DoneReason) {
	seq := s.seqs[seqIndex]

	flushPending(seq)
	seq.doneReason = reason

	// Tensorpuffer response-cache stash: persist the full generated
	// output token sequence under the original prompt's content hash.
	// Only fires when TPUF_KVBM_STASH_RESPONSE=1 was set (the
	// promptTokensForTpuf snapshot is non-nil only in that case).
	// Best-effort — failure is logged inside StashResponseToPuffer.
	if len(seq.promptTokensForTpuf) > 0 && len(seq.generatedTokens) > 0 {
		s.cache.StashResponseToPuffer(seq.promptTokensForTpuf, seq.generatedTokens)
	}

	close(seq.responses)
	close(seq.embedding)
	seq.cache.InUse = false
	s.seqs[seqIndex] = nil
	s.seqsSem.Release(1)
}

// track batch state between forwardBatch, computeBatch and predictForwardBatch

func (s *Server) run(ctx context.Context) {
	s.ready.Wait()

	supportsAsync := pooling.Type(s.model.Backend().Config().Uint("pooling_type")) == pooling.TypeNone

	var previousBatch batchState
	for {
		select {
		case <-ctx.Done():
			return
		default:
			var err error
			nextBatch, err := s.forwardBatch(previousBatch)
			if err != nil {
				panic(err)
			}

			if supportsAsync {
				go s.computeBatch(nextBatch)
			} else {
				s.computeBatch(nextBatch)
			}

			previousBatch = nextBatch
		}
	}
}

// forwardBatch will calculate a batch.
func (s *Server) forwardBatch(pendingBatch batchState) (nextBatch batchState, err error) {
	// If we have a pending batch still processing, wait until Compute has started
	// before setting up the next batch so the seqs inputs are ready to receive their
	// token values and we get the correct input pointers for the batchInputs
	if pendingBatch.ctx != nil {
		logutil.Trace("forwardBatch waiting for compute to start", "pendingBatch.id", pendingBatch.id)
		<-pendingBatch.computeStartedCh
		logutil.Trace("forwardBatch compute started, setting up next batch", "pendingBatch.id", pendingBatch.id, "id", s.batchID)
		nextBatch.inputsReadyCh = pendingBatch.outputsReadyCh // Chain the ouputs from the pending batch to the next inputs batch
	} else {
		logutil.Trace("forwardBatch no pending batch detected", "batchID", s.batchID)
		// No pendingBatch, so the inputs will be ready in the seqs immediately
		nextBatch.inputsReadyCh = make(chan struct{}, 1)
		nextBatch.inputsReadyCh <- struct{}{}
	}

	s.mu.Lock()
	for s.allNil() {
		s.cond.Wait() // Wait until an item is added
	}
	defer s.mu.Unlock()

	nextBatch.ctx = s.model.Backend().NewContext()
	defer func() {
		if err != nil {
			nextBatch.ctx.Close()
			nextBatch.ctx = nil
		}
	}()
	nextBatch.id = s.batchID
	nextBatch.seqs = append([]*Sequence{}, s.seqs...)
	nextBatch.computeStartedCh = make(chan struct{}, 1)
	nextBatch.outputsReadyCh = make(chan struct{}, 1)

	// Prepare the seqs and batch, but defer the input token values as we may not be ready yet
	var batchInputs []*input.Input
	var batchOutputs []int32
	var batch input.Batch

	resumeSeq := -1
	seqIdx := s.nextSeq - 1
	for range s.seqs {
		seqIdx = (seqIdx + 1) % len(s.seqs)
		seq := s.seqs[seqIdx]
		if seq == nil {
			continue
		}

		if !s.cache.enabled {
			seq.inputs = append(seq.cache.Inputs, seq.inputs...)
			seq.cache.Inputs = []*input.Input{}
		}

		batchSize := s.batchSize

		for i, inp := range seq.inputs {
			// If we are required to put following inputs into a single batch then extend the
			// batch size. Since we are only extending the size the minimum amount possible, this
			// will cause a break if we have existing inputs.
			minBatch := 1 + inp.SameBatch
			if minBatch > batchSize {
				batchSize = minBatch
			}

			// Stop if the required batch would put us over the total batch size (including tokens
			// added by other sequences). If we haven't been able to add anything yet then pick up
			// here again for the next batch to avoid starvation, though we can opportunistically
			// check if other sequences can still squeeze something in.
			if len(batchInputs)+minBatch > batchSize {
				if len(seq.pendingInputs) == 0 && resumeSeq == -1 {
					resumeSeq = seqIdx
				}
				break
			}

			// If the sum of our working set (already processed tokens, tokens we added to this
			// batch, required following tokens) exceeds the context size, then trigger a shift
			// now so we don't have to do one later when we can't break the batch.
			if int32(len(seq.cache.Inputs)+len(seq.pendingInputs)+minBatch) > s.cache.numCtx {
				if len(seq.pendingInputs) != 0 {
					break
				}

				if !seq.shift {
					s.removeSequence(seqIdx, llm.DoneReasonLength)
					nextBatch.seqs[seqIdx] = nil
					break
				}

				err = s.cache.ShiftCacheSlot(seq.cache, seq.numKeep)
				if err != nil {
					var reprocess *ErrReprocessInputs
					if errors.As(err, &reprocess) {
						// Prepend these inputs to the sequence's inputs queue for reprocessing
						seq.inputs = append(reprocess.Inputs, seq.inputs...)
						// Skip this sequence but continue processing the rest
						nextBatch.seqs[seqIdx] = nil // clear this sequence for this batch
						err = nil
						continue
					} else {
						return
					}
				}
			}

			batchInputs = append(batchInputs, seq.inputs[i])
			if inp.Multimodal != nil {
				var mm []input.Multimodal
				mm, err = seq.mmStore.getMultimodal(s.model.Backend(), nextBatch.ctx, inp.Multimodal, false)
				if err != nil {
					return
				}
				batch.Multimodal = append(batch.Multimodal, input.MultimodalIndex{Index: len(batchInputs) - 1, Multimodal: mm})
			}

			batch.Positions = append(batch.Positions, int32(len(seq.cache.Inputs)+len(seq.pendingInputs)))
			batch.Sequences = append(batch.Sequences, seq.cache.Id)

			seq.iBatch = len(batchOutputs)
			if i+1 == len(seq.inputs) || seq.embeddingOnly {
				batchOutputs = append(batchOutputs, int32(len(batchInputs)-1))
			}
			logutil.Trace("forwardBatch iBatch", "batchID", s.batchID, "seqIdx", seqIdx, "seq.iBatch", seq.iBatch, "i+1", i+1, "len(seq.inputs)", len(seq.inputs))
			seq.pendingInputs = append(seq.pendingInputs, inp)
		}

		seq.inputs = seq.inputs[len(seq.pendingInputs):]
	}

	startedAt := time.Now()
	for i := range nextBatch.seqs {
		if nextBatch.seqs[i] != nil && nextBatch.seqs[i].startedAt.IsZero() {
			nextBatch.seqs[i].startedAt = startedAt
		}
	}

	if resumeSeq != -1 {
		s.nextSeq = resumeSeq
	} else {
		s.nextSeq = seqIdx + 1
	}

	if len(batchInputs) == 0 {
		logutil.Trace("forwardBatch no batchInputs, going idle", "batchID", s.batchID)
		nextBatch.ctx.Close()
		nextBatch.ctx = nil
		return
	}
	s.batchID++

	// Actual batchInputs values will be injected into the batch.Inputs tensor before calling Compute
	batch.Inputs = nextBatch.ctx.Input().Empty(ml.DTypeI32, len(batchInputs))
	batch.Outputs = nextBatch.ctx.Input().FromInts(batchOutputs, len(batchOutputs))
	nextBatch.ctx.SetBatchSize(len(batchInputs))
	nextBatch.modelOutput, err = model.Forward(nextBatch.ctx, s.model, batch)
	if err != nil {
		err = fmt.Errorf("failed to build graph: %w", err)
		return
	}
	nextBatch.batchInputs = batchInputs
	nextBatch.batch = batch

	return
}

// Async processing of the next batch
func (s *Server) computeBatch(activeBatch batchState) {
	if activeBatch.ctx == nil {
		// Nothing to compute
		return
	}
	defer activeBatch.ctx.Close()

	// Wait until inputs are ready
	logutil.Trace("computeBatch: waiting for inputs to be ready", "batchID", activeBatch.id)
	<-activeBatch.inputsReadyCh
	logutil.Trace("computeBatch: inputs are ready", "batchID", activeBatch.id)

	// Once we complete, signal the next batch of inputs are ready
	// This will unblock the next computeBatch, or forwardBatch if new seqs come in
	defer func() {
		logutil.Trace("computeBatch: outputs are ready", "batchID", activeBatch.id)
		activeBatch.outputsReadyCh <- struct{}{}
	}()

	s.mu.Lock()

	// Gather the actual input token values now that they're ready
	batchInputs := make([]int32, len(activeBatch.batchInputs))
	for i := range batchInputs {
		batchInputs[i] = activeBatch.batchInputs[i].Token
	}

	// Now we run part of the decoding algorithm to adjust the seq.inputs with placeholder tokens
	// so that forwardBatch can build a batchInputs set which will eventually contain the actual
	// decoded tokens.
	nextBatchTokens := make([]*input.Input, len(s.seqs))
	iBatches := make([]int, len(s.seqs)) // Record the iBatch values before releasing the lock
	for i, seq := range s.seqs {
		iBatches[i] = -1
		if seq == nil {
			continue
		}
		// Skip over any newly added or skipped sequences
		if activeBatch.seqs[i] == nil {
			continue
		}

		// Detect if the sequence we're processing has already been completed and replaced
		// with a new sequence
		if seq != activeBatch.seqs[i] {
			logutil.Trace("computeBatch: sequence replaced, discarding its results", "batchID", activeBatch.id, "seqIdx", i)
			continue
		}

		// Pending inputs will actually be in the cache after we call Compute.
		// However, we have already resolved any placeholder tokens.
		//
		// It's possible for incoming sequences to look at the values that we've
		// added to the cache here and start relying on them before we've done
		// the computation. This is OK as long as we ensure that this batch's
		// computation happens before any future batch's and we never fail
		// (unless we take down the whole runner).
		if len(seq.pendingInputs) > 0 {
			seq.cache.Inputs = append(seq.cache.Inputs, seq.pendingInputs...)
			seq.pendingInputs = []*input.Input{}
		}

		// don't sample prompt processing
		if len(seq.inputs) != 0 {
			if !s.cache.enabled {
				panic("caching disabled but unable to fit entire input in a batch")
			}
			continue
		}

		nextToken := &input.Input{Token: 0} // placeholder we'll fill in after Compute/Floats
		seq.inputs = []*input.Input{nextToken}
		nextBatchTokens[i] = nextToken
		iBatches[i] = seq.iBatch
	}

	// At this point the seqs are ready for forwardBatch to move forward so unblock
	s.mu.Unlock()

	activeBatch.batch.Inputs.FromInts(batchInputs)
	activeBatch.ctx.ComputeWithNotify(
		func() {
			logutil.Trace("computeBatch: signaling computeStartedCh", "batchID", activeBatch.id)
			activeBatch.computeStartedCh <- struct{}{}
		},
		activeBatch.modelOutput)

	outputs := activeBatch.modelOutput.Floats()
	t := time.Now()

	logutil.Trace("computeBatch: logits ready", "batchID", activeBatch.id)

	s.mu.Lock()
	defer s.mu.Unlock()

	logutil.Trace("computeBatch: decoding", "batchID", activeBatch.id)
	for i, seq := range s.seqs {
		if seq == nil || nextBatchTokens[i] == nil {
			continue
		}
		// If the sequence was replaced while this batch was computing, discard results.
		if activeBatch.seqs[i] != seq {
			logutil.Trace("computeBatch: sequence replaced, discarding its results", "batchID", activeBatch.id, "seqIdx", i)
			continue
		}

		seq.lastUpdatedAt = t
		seq.numPredicted++
		firstSample := seq.numPredicted == 1
		if firstSample {
			seq.processingDuration = seq.lastUpdatedAt.Sub(seq.startedAt)
			seq.startedAt = seq.lastUpdatedAt
		}

		// if done processing the prompt, generate an embedding and return
		if seq.embeddingOnly {
			seq.embedding <- outputs
			s.removeSequence(i, llm.DoneReasonStop)
			continue
		}

		// sample a token
		vocabSize := len(outputs) / activeBatch.batch.Outputs.Dim(0)
		logutil.Trace("computeBatch: vocab details", "batchID", activeBatch.id, "seqIdx", i, "len(logits)", len(outputs), "len(activeBatch.batch.Outputs)", activeBatch.batch.Outputs.Dim(0), "vocabSize", vocabSize, "iBatches", iBatches)
		logits := outputs[iBatches[i]*vocabSize : (iBatches[i]+1)*vocabSize]
		token, err := seq.sampler.Sample(logits)
		if err != nil {
			panic("failed to sample token")
		}

		nextBatchTokens[i].Token = token

		// Tensorpuffer response-cache: track every sampled token so we
		// can stash the full output sequence at request end. Cheap
		// O(1) append; only allocates when TPUF_KVBM_STASH_RESPONSE
		// is enabled (promptTokensForTpuf is the gate).
		if seq.promptTokensForTpuf != nil {
			seq.generatedTokens = append(seq.generatedTokens, token)
		}

		// Tensorpuffer post-prefill stash (Direction B). Fires exactly
		// once per request — right after the first generated token has
		// been sampled. Stashes both the KV state AND the first sampled
		// token so a future warm load can FULL_SKIP the post-prefill
		// 1-token forward pass. No-op when the underlying cache type
		// doesn't implement SerializableCache or TPUF_KVBM_ENABLE is
		// unset; sidecar stash is gated by TPUF_KVBM_FULL_SKIP separately.
		//
		// Stashes regardless of EOS — the cache state is valid even when
		// the first sample is EOS (cold run produced no content but the
		// state is reusable; warm load detects EOS first_token and
		// terminates without compute).
		if firstSample {
			s.cache.StashToPuffer(seq.cache, token)
		}

		// if it's an end of sequence token, break
		if s.model.(tokenizer.Tokenizer).Is(token, tokenizer.SpecialEOS) {
			// TODO (jmorganca): we should send this back
			// as it's important for the /api/generate context
			// seq.responses <- piece
			logutil.Trace("computeBatch: EOS", "batchID", activeBatch.id, "seqIdx", i)
			s.removeSequence(i, llm.DoneReasonStop)
			continue
		}

		piece, err := s.model.(tokenizer.Tokenizer).Decode([]int32{token})
		if err != nil {
			panic("failed to decode token")
		}

		// Calculate logprobs if requested (after EOS check to avoid logprobs for EOS tokens)
		if seq.logprobs {
			logprobs := calculateLogprobs(logits, token, seq.topLogprobs, s.model.(tokenizer.Tokenizer))
			seq.pendingLogprobs = append(seq.pendingLogprobs, logprobs...)
		}

		seq.pendingResponses = append(seq.pendingResponses, piece)

		// if past the num predict limit
		if seq.numPredict > 0 && seq.numPredicted >= seq.numPredict {
			s.removeSequence(i, llm.DoneReasonLength)
			continue
		}

		sequence := strings.Join(seq.pendingResponses, "")

		if ok, stop := common.FindStop(sequence, seq.stop); ok {
			slog.Debug("hit stop token", "pending", seq.pendingResponses, "stop", stop)

			var tokenTruncated bool
			origLen := len(seq.pendingResponses)
			seq.pendingResponses, tokenTruncated = common.TruncateStop(seq.pendingResponses, stop)
			newLen := len(seq.pendingResponses)

			// Truncate logprobs to match the truncated responses
			if seq.logprobs {
				origLogprobsLen := len(seq.pendingLogprobs)
				numTokensRemoved := origLen - newLen
				newLogprobsLen := origLogprobsLen - numTokensRemoved
				if newLogprobsLen < 0 {
					newLogprobsLen = 0
				}
				seq.pendingLogprobs = seq.pendingLogprobs[:newLogprobsLen]
			}

			// Update the cache based on the tokens that will be returned:
			// - We have 1 token more than is currently in the cache because
			// the last one generated wasn't submitted to Decode
			// - Remove any stop sequences that we stripped out
			// - If truncateStop removed a portion of a token, drop that
			// - As defense-in-depth, if truncatedToken didn't find a stop token
			// remove the extra one that we added to the cache len
			tokenLen := len(seq.cache.Inputs) + 1
			tokenLen -= origLen - newLen
			if tokenTruncated || origLen == newLen {
				tokenLen--
			}

			seq.cache.Inputs = seq.cache.Inputs[:tokenLen]

			s.removeSequence(i, llm.DoneReasonStop)
			continue
		}

		if common.ContainsStopSuffix(sequence, seq.stop) {
			continue
		}

		if common.IncompleteUnicode(sequence) {
			continue
		}

		if !flushPending(seq) {
			s.removeSequence(i, llm.DoneReasonConnectionClosed)
		}
	}

	samplingDuration := time.Since(t)
	for i, seq := range s.seqs {
		if seq != nil && nextBatchTokens[i] != nil {
			s.seqs[i].samplingDuration += samplingDuration
		}
	}
}

func (s *Server) completion(w http.ResponseWriter, r *http.Request) {
	var req llm.CompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	if req.Options == nil {
		opts := api.DefaultOptions()
		req.Options = &opts
	}

	// Set the headers to indicate streaming
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Transfer-Encoding", "chunked")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	var grammar *sample.GrammarSampler
	var err error
	if req.Grammar != "" {
		grammar, err = sample.NewGrammarSampler(s.model.(tokenizer.Tokenizer), req.Grammar)
		if err != nil {
			http.Error(w, "failed to load model vocabulary required for format", http.StatusInternalServerError)
			return
		}
		defer grammar.Free()
	}

	sampler := sample.NewSampler(
		req.Options.Temperature,
		req.Options.TopK,
		req.Options.TopP,
		req.Options.MinP,
		req.Options.Seed,
		grammar,
	)

	seq, err := s.NewSequence(req.Prompt, req.Images, NewSequenceParams{
		numPredict:  req.Options.NumPredict,
		stop:        req.Options.Stop,
		numKeep:     int32(req.Options.NumKeep),
		sampler:     sampler,
		embedding:   false,
		shift:       req.Shift,
		truncate:    req.Truncate,
		logprobs:    req.Logprobs,
		topLogprobs: req.TopLogprobs,
	})
	if err != nil {
		if errors.Is(err, errorInputTooLong) {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		http.Error(w, fmt.Sprintf("Failed to create new sequence: %v", err), http.StatusInternalServerError)
		return
	}

	// Ensure there is a place to put the sequence, released when removed from s.seqs
	if err := s.seqsSem.Acquire(r.Context(), 1); err != nil {
		if errors.Is(err, context.Canceled) {
			slog.Info("aborting completion request due to client closing the connection")
		} else {
			http.Error(w, fmt.Sprintf("Failed to acquire semaphore: %v", err), http.StatusInternalServerError)
		}
		return
	}

	s.mu.Lock()
	found := false
	for i, sq := range s.seqs {
		if sq == nil {
			// Tensorpuffer response-cache key: snapshot the prompt
			// tokens before LoadCacheSlot mutates seq.inputs (the
			// warm path consumes the prompt and leaves only the
			// trim+redecode tail). Used at removeSequence time to
			// stash the generated response under this exact key.
			if IsTpufStashResponseEnabled() {
				promptCopy := make([]uint32, 0, len(seq.inputs))
				for _, p := range seq.inputs {
					if p != nil {
						promptCopy = append(promptCopy, uint32(p.Token))
					}
				}
				seq.promptTokensForTpuf = promptCopy
			}

			seq.cache, seq.inputs, err = s.cache.LoadCacheSlot(seq.inputs, true)
			if err != nil {
				s.mu.Unlock()
				s.seqsSem.Release(1)
				http.Error(w, fmt.Sprintf("Failed to load cache: %v", err), http.StatusInternalServerError)
				return
			}

			// Tensorpuffer FULL_SKIP fast-path. tryLoadFromPuffer dual-
			// hit (KV state + first-token sidecar): emit the stashed
			// first token to the caller now, queue it as seq.inputs so
			// the next compute step is a normal decode of (first_token
			// → second_token). Saves exactly one forward pass relative
			// to the regular trim+redecode warm path.
			//
			// On EOS first_token, mirror the cold-run termination path:
			// emit Done immediately without ever scheduling compute for
			// this seq, and release the semaphore + skip the s.seqs slot
			// allocation. The handler's response loop receives the close
			// and writes the Done envelope.
			fullSkipDone := false

			// Tensorpuffer response-cache fast-path. Highest-priority
			// short-circuit: if the response sidecar hit, we have the
			// FULL output token sequence from the cold run. Decode and
			// emit it directly, then close — no compute scheduled.
			//
			// Only fires when prompt is byte-identical to a previous
			// stashed prompt AND decode was deterministic (caller is
			// responsible for ensuring temp/top_k/seed match).
			if seq.cache.PendingResponse != nil {
				resp := seq.cache.PendingResponse
				seq.cache.PendingResponse = nil
				seq.cache.PendingFirstToken = -1

				if err := s.applyTpufResponseCache(seq, resp); err != nil {
					slog.Warn("tensorpuffer response-cache emit failed (falling back to FULL_SKIP)",
						"err", err)
					// Don't bail — fall through to FULL_SKIP if the
					// first_token is also pending; otherwise normal compute.
				} else {
					seq.cache.InUse = false
					s.seqsSem.Release(1)
					fullSkipDone = true
					slog.Info("tensorpuffer response-cache emitted — skipping compute entirely",
						"slot", seq.cache.Id, "response_tokens", len(resp))
				}
			}

			if !fullSkipDone && seq.cache.PendingFirstToken >= 0 {
				token := seq.cache.PendingFirstToken
				seq.cache.PendingFirstToken = -1

				if s.model.(tokenizer.Tokenizer).Is(token, tokenizer.SpecialEOS) {
					seq.doneReason = llm.DoneReasonStop
					seq.numPredicted = 1
					seq.processingDuration = 0
					close(seq.responses)
					seq.cache.InUse = false
					s.seqsSem.Release(1)
					fullSkipDone = true
					slog.Info("tensorpuffer FULL_SKIP first_token=EOS — no compute issued",
						"slot", seq.cache.Id)
				} else {
					s.applyTpufFullSkip(seq, token)
				}
			}

			if !fullSkipDone {
				s.seqs[i] = seq
				s.cond.Signal()
			}
			found = true
			break
		}
	}
	s.mu.Unlock()

	if !found {
		s.seqsSem.Release(1)
		http.Error(w, "could not find an available sequence", http.StatusInternalServerError)
		return
	}

	for {
		select {
		case <-r.Context().Done():
			close(seq.quit)
			return
		case resp, ok := <-seq.responses:
			if ok {
				if err := json.NewEncoder(w).Encode(&llm.CompletionResponse{
					Content:  resp.content,
					Logprobs: resp.logprobs,
				}); err != nil {
					http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
					close(seq.quit)
					return
				}

				flusher.Flush()
			} else {
				if err := json.NewEncoder(w).Encode(&llm.CompletionResponse{
					Done:               true,
					DoneReason:         seq.doneReason,
					PromptEvalCount:    seq.numPromptInputs,
					PromptEvalDuration: seq.processingDuration,
					EvalCount:          seq.numPredicted,
					EvalDuration:       seq.lastUpdatedAt.Sub(seq.startedAt) - seq.samplingDuration,
				}); err != nil {
					http.Error(w, fmt.Sprintf("failed to encode final response: %v", err), http.StatusInternalServerError)
				}

				return
			}
		}
	}
}

func (s *Server) embeddings(w http.ResponseWriter, r *http.Request) {
	if pooling.Type(s.model.Backend().Config().Uint("pooling_type")) == pooling.TypeNone {
		http.Error(w, "this model does not support embeddings", http.StatusNotImplemented)
		return
	}

	var req llm.EmbeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("bad request: %s", err), http.StatusBadRequest)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	seq, err := s.NewSequence(req.Content, nil, NewSequenceParams{
		embedding: true,
		truncate:  false,
	})
	if err != nil {
		if errors.Is(err, errorInputTooLong) {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		http.Error(w, fmt.Sprintf("failed to create new sequence: %v", err), http.StatusInternalServerError)
		return
	}

	if err := s.seqsSem.Acquire(r.Context(), 1); err != nil {
		if errors.Is(err, context.Canceled) {
			slog.Info("aborting embedding request due to client closing the connection")
		} else {
			http.Error(w, fmt.Sprintf("failed to acquire semaphore: %v", err), http.StatusInternalServerError)
		}
		return
	}

	s.mu.Lock()
	found := false
	for i, sq := range s.seqs {
		if sq == nil {
			seq.cache, seq.inputs, err = s.cache.LoadCacheSlot(seq.inputs, false)
			if err != nil {
				s.mu.Unlock()
				s.seqsSem.Release(1)
				http.Error(w, fmt.Sprintf("failed to load cache: %v", err), http.StatusInternalServerError)
				return
			}

			s.seqs[i] = seq
			s.cond.Signal()
			found = true
			break
		}
	}
	s.mu.Unlock()

	if !found {
		s.seqsSem.Release(1)
		http.Error(w, "could not find an available sequence", http.StatusInternalServerError)
		return
	}

	if err := json.NewEncoder(w).Encode(&llm.EmbeddingResponse{
		Embedding:       <-seq.embedding,
		PromptEvalCount: seq.numPromptInputs,
	}); err != nil {
		http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
	}
}

func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(&llm.ServerStatusResponse{
		Status:   s.status,
		Progress: s.progress,
	}); err != nil {
		http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
	}
}

func (s *Server) reserveWorstCaseGraph(prompt bool) error {
	ctx := s.model.Backend().NewContext()
	defer ctx.Close()

	var err error
	batchSize := 1
	if prompt {
		batchSize = s.batchSize
	}

	inputs := make([]*input.Input, batchSize)
	for i := range inputs {
		inputs[i] = &input.Input{}
	}
	mmStore := newMultimodalStore()

	// Multimodal strategy:
	// - Encode a 2048x2048 image. This assumes that a single image of this
	//   size is sufficient to trigger the worst case. This is currently true
	//   because for existing models, only a single image fits in a batch.
	// - Add the embedding to a full batch of tokens - this is necessary because
	//   the model may be looking for non-image data, such as <image> tags.
	// - Run PostTokenize to execute any transformations between generated
	//   embeddings and what the forward pass expects.
	// - The result may now be larger than a batch (images may not fit in a
	//   single batch), so trim based on what will fit and must be grouped together.
	// - Fill out the rest of the space with text tokens.
	if multimodalProcessor, ok := s.model.(model.MultimodalProcessor); prompt && ok {
		mmCtx := s.model.Backend().NewContext()
		defer mmCtx.Close()

		img := image.NewGray(image.Rect(0, 0, 2048, 2048))
		var buf bytes.Buffer
		bmp.Encode(&buf, img)

		if inputs[0].Multimodal, err = multimodalProcessor.EncodeMultimodal(mmCtx, buf.Bytes()); err == nil {
			mmStore.addMultimodal(inputs[0].Multimodal)

			inputs, err = multimodalProcessor.PostTokenize(inputs)
			if err != nil {
				return err
			}

			for i, inp := range inputs {
				minBatch := 1 + inp.SameBatch
				if minBatch > s.batchSize {
					inputs = inputs[i:min(i+minBatch, len(inputs))]
					break
				} else if i+minBatch > s.batchSize {
					inputs = inputs[:i]
					break
				}
			}

			if len(inputs) < batchSize {
				newInputs := make([]*input.Input, batchSize)
				copy(newInputs, inputs)
				for i := len(inputs); i < batchSize; i++ {
					newInputs[i] = &input.Input{}
				}
				inputs = newInputs
			}
		}
	}

	var batch input.Batch

	batchInputs := make([]int32, len(inputs))
	batch.Positions = make([]int32, len(inputs))
	batch.Sequences = make([]int, len(inputs))
	for i, inp := range inputs {
		batchInputs[i] = inp.Token
		if inp.Multimodal != nil {
			mm, err := mmStore.getMultimodal(s.model.Backend(), ctx, inp.Multimodal, true)
			if err != nil {
				return err
			}
			batch.Multimodal = append(batch.Multimodal, input.MultimodalIndex{Index: i, Multimodal: mm})
		}

		batch.Positions[i] = int32(i)
	}

	batch.Inputs = ctx.Input().FromInts(batchInputs, len(batchInputs))
	batch.Outputs = ctx.Input().Empty(ml.DTypeI32, s.parallel)

	cache := s.model.Config().Cache
	if cache != nil {
		err := cache.StartForward(ctx, batch, true)
		if err != nil {
			return err
		}
	}

	t, err := s.model.Forward(ctx, batch)
	if err != nil {
		return err
	}

	ctx.SetBatchSize(batchSize)
	ctx.Forward(t).Reserve()

	return nil
}

// allocModel pre-allocates the maximum needed memory for a model
// based on the given parameters
func (s *Server) allocModel(
	mpath string,
	params ml.BackendParams,
	loraPath []string,
	parallel int,
	kvCacheType string,
	kvSize int,
	multiUserCache bool,
) (panicErr error) {
	// Convert memory allocation panics to errors
	defer func() {
		if r := recover(); r != nil {
			if err, ok := r.(error); ok {
				var noMem ml.ErrNoMem
				if errors.As(err, &noMem) {
					panicErr = noMem
				} else {
					panic(r)
				}
			} else {
				panic(r)
			}
		}
	}()

	var err error
	s.model, err = model.New(mpath, params)
	if err != nil {
		return err
	}

	// TODO(jessegross): LoRA loading
	if len(loraPath) > 0 {
		return errors.New("loras are not yet implemented")
	}

	if s.model.Config().Cache == nil {
		if parallel > 1 {
			parallel = 1
			slog.Warn("model does not support caching, disabling parallel processing")
		}
		if s.batchSize < kvSize {
			s.batchSize = kvSize
			slog.Warn("model does not support caching, setting batch size to context length", "batch_size", kvSize)
		}
	}

	s.cache, err = NewInputCache(s.model, kvCacheType, int32(kvSize), parallel, s.batchSize, multiUserCache)
	if err != nil {
		return err
	}

	s.parallel = parallel
	s.seqs = make([]*Sequence, s.parallel)
	s.seqsSem = semaphore.NewWeighted(int64(s.parallel))

	err = s.reserveWorstCaseGraph(true)
	if err != nil {
		return err
	}

	return s.reserveWorstCaseGraph(false)
}

// closeModel frees all memory associated with a model
func (s *Server) closeModel() {
	s.cache.Close()
	s.cache = nil
	if s.model != nil {
		s.model.Backend().Close()
		s.model = nil
	}
}

// loadModel loads the weights for a model. The memory must already
// have been allocated with allocModel
func (s *Server) loadModel() {
	err := s.model.Backend().Load(context.TODO(),
		func(progress float32) {
			s.progress = progress
		})
	if err != nil {
		panic(fmt.Errorf("failed to load model: %v", err))
	}

	if postLoader, ok := s.model.(model.PostLoader); ok {
		if err := postLoader.PostLoad(); err != nil {
			panic(fmt.Errorf("failed to finalize model initialization: %v", err))
		}
	}

	s.status = llm.ServerStatusReady
	s.ready.Done()
}

// load is the handler called by the Ollama server to process different
// load operations
func (s *Server) load(w http.ResponseWriter, r *http.Request) {
	s.loadMu.Lock()
	defer s.loadMu.Unlock()

	w.Header().Set("Content-Type", "application/json")

	if s.status != llm.ServerStatusLaunched {
		http.Error(w, "model already loaded", http.StatusInternalServerError)
		return
	}

	var req llm.LoadRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}

	slog.Info("load", "request", req)

	if req.Operation == llm.LoadOperationClose {
		s.closeModel()
		if err := json.NewEncoder(w).Encode(&llm.LoadResponse{}); err != nil {
			http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
		}
		return
	}

	s.lastLoad.Operation = req.Operation
	loadModel := s.model == nil || !reflect.DeepEqual(req, s.lastLoad)

	s.lastLoad = req

	if loadModel {
		s.closeModel()

		params := ml.BackendParams{
			AllocMemory:    req.Operation != llm.LoadOperationFit,
			NumThreads:     req.NumThreads,
			GPULayers:      req.GPULayers,
			FlashAttention: req.FlashAttention,
		}

		s.batchSize = req.BatchSize

		err := s.allocModel(s.modelPath, params, req.LoraPath, req.Parallel, req.KvCacheType, req.KvSize, req.MultiUserCache)
		if err != nil {
			s.closeModel()

			var noMem ml.ErrNoMem
			if errors.As(err, &noMem) {
				resp := llm.LoadResponse{Success: false, Memory: noMem.BackendMemory}
				if err := json.NewEncoder(w).Encode(&resp); err != nil {
					http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
				}

				return
			}

			http.Error(w, fmt.Sprintf("failed to initialize model: %v", err), http.StatusInternalServerError)
			return
		}
	}

	mem := s.model.Backend().BackendMemory()

	switch req.Operation {
	case llm.LoadOperationFit:
		// LoadOperationFit can't be used for anything else, so just close it
		s.closeModel()

	// LoadOperationAlloc should stay open for future operations

	case llm.LoadOperationCommit:
		s.status = llm.ServerStatusLoadingModel
		go s.loadModel()
	}

	resp := llm.LoadResponse{Success: true, Memory: mem}
	if err := json.NewEncoder(w).Encode(&resp); err != nil {
		http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
		return
	}
}

// info is the handler called by the Ollama server to report information
// about the GPU devices in use by this runner
func (s *Server) info(w http.ResponseWriter, r *http.Request) {
	s.loadMu.Lock()
	defer s.loadMu.Unlock()

	w.Header().Set("Content-Type", "application/json")

	m, err := s.infoModelLocked()
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to initialize backend: %v", err), http.StatusInternalServerError)
		return
	}
	startDevices := time.Now()
	infos := m.Backend().BackendDevices()
	slog.Debug("gathering device infos took", "duration", time.Since(startDevices))
	if err := json.NewEncoder(w).Encode(&infos); err != nil {
		http.Error(w, fmt.Sprintf("failed to encode response: %v", err), http.StatusInternalServerError)
		return
	}
}

func (s *Server) infoModelLocked() (model.Model, error) {
	if s.model != nil {
		return s.model, nil
	}

	if !s.infoInitialized {
		s.infoInitialized = true

		func() {
			startLoad := time.Now()
			defer func() {
				if rec := recover(); rec != nil {
					s.infoErr = fmt.Errorf("panic during dummy backend initialization: %v", rec)
				}
			}()

			// Dummy load to get the backend wired up.
			f, err := os.CreateTemp("", "*.bin")
			if err != nil {
				s.infoErr = err
				return
			}
			defer f.Close()
			defer os.Remove(f.Name())

			if err := ggml.WriteGGUF(f, ggml.KV{
				"general.architecture": "llama",
				"tokenizer.ggml.model": "gpt2",
			}, nil); err != nil {
				s.infoErr = err
				return
			}

			s.infoModel, s.infoErr = model.New(f.Name(), ml.BackendParams{NumThreads: runtime.NumCPU(), AllocMemory: false, GPULayers: ml.GPULayersList{{}}})
			if s.infoErr == nil {
				slog.Debug("dummy model load took", "duration", time.Since(startLoad))
			}
		}()
	}

	if s.infoErr != nil {
		return nil, s.infoErr
	}
	if s.infoModel == nil {
		return nil, fmt.Errorf("dummy backend initialization did not produce a model")
	}

	return s.infoModel, nil
}

func Execute(args []string) error {
	fs := flag.NewFlagSet("runner", flag.ExitOnError)
	mpath := fs.String("model", "", "Path to model binary file")
	port := fs.Int("port", 8080, "Port to expose the server on")
	_ = fs.Bool("verbose", false, "verbose output (default: disabled)")

	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), "Runner usage\n")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return err
	}
	slog.SetDefault(logutil.NewLogger(os.Stderr, envconfig.LogLevel()))
	slog.Info("starting ollama engine")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	server := &Server{
		modelPath: *mpath,
		status:    llm.ServerStatusLaunched,
	}

	server.cond = sync.NewCond(&server.mu)
	server.ready.Add(1)

	go server.run(ctx)

	addr := "127.0.0.1:" + strconv.Itoa(*port)
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		fmt.Println("Listen error:", err)
		return err
	}
	defer listener.Close()

	mux := http.NewServeMux()
	// TODO: support embeddings
	mux.HandleFunc("GET /info", server.info)
	mux.HandleFunc("POST /load", server.load)
	mux.HandleFunc("POST /embedding", server.embeddings)
	mux.HandleFunc("POST /completion", server.completion)
	mux.HandleFunc("GET /health", server.health)

	httpServer := http.Server{
		Handler: mux,
	}

	log.Println("Server listening on", addr)
	if err := httpServer.Serve(listener); err != nil {
		log.Fatal("server error:", err)
		return err
	}

	return nil
}
