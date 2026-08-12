/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  * @authors        :
  ******************************************************************************
  * @attention
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  *             
  *          _______                        ______   ______           
  *         /       \                      /      \ /      |          
  *         $$$$$$$  |  ______    ______  /$$$$$$  |$$$$$$/   ______  
  *         $$ |__$$ | /      \  /      \ $$ |__$$ |  $$ |   /      \ 
  *         $$    $$< /$$$$$$  |/$$$$$$  |$$    $$ |  $$ |  /$$$$$$  |
  *         $$$$$$$  |$$ |  $$ |$$ |  $$/ $$$$$$$$ |  $$ |  $$ |  $$ |
  *         $$ |__$$ |$$ \__$$ |$$ |      $$ |  $$ | _$$ |_ $$ \__$$ |
  *         $$    $$/ $$    $$/ $$ |      $$ |  $$ |/ $$   |$$    $$ |
  *         $$$$$$$/   $$$$$$/  $$/       $$/   $$/ $$$$$$/  $$$$$$$ |
  *                                                               $$ |
  *                                                               $$ |
  *                                                               $$/ 
  *
  *             Powered by Llama2 based upon llama2.c by @karpathy
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#ifdef KWS_LLAMA_COMBINED
/* In the combined KWS+Llama binary the KWS consumer TU also has a header literally named "main.h"
 * on the include path; include ours via a uniquely-named wrapper so the two never clash. */
#include "borai_main.h"
#else
#include "main.h"
#endif

#if defined(TRANSPOSED_WEIGHTS) || defined(VEC_SOFTMAX)
#include "layers.h"
#endif

#include "simple_setup.h"   /* init_test(): portable PLL + UART bring-up (bearly25 & dsp25) */

#ifdef KWS_LLAMA_COMBINED
/* --- Combined KWS + Llama dual-core demo (c2c-demos/bearly-kws-llama) integration shims. ---------
 * This TU is compiled into the combined binary alongside the TinySpeech KWS runtime, which ALSO
 * exports a non-static softmax(); rename ours to avoid a link-time collision. The single operating
 * frequency and the program entry points (main / __main / app_main) are owned by the combined
 * demo's main.c and by bmark-lib/hthread.c, so those definitions below are #ifdef'd out. The
 * exposed llama_build()/llama_run_forever() entry points are added at the end of this file.
 * None of this affects the standalone `boraiq` build (macro undefined there). */
#define softmax llama_softmax
extern uint64_t target_frequency;      /* owned by the combined demo's main.c */
extern volatile int g_llama_stop;      /* set by the control hart; abort generation between tokens */
#endif

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
// #define ENABLE_BORAVOICE_INTEG
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */
const unsigned char *ASCII_CRLF = (const unsigned char *) "\r\n";
const unsigned char *ASCII_BEL = (const unsigned char *) "\a";

// int32_t GS = 64; // group size global for quantization of the weights

#ifndef KWS_LLAMA_COMBINED
uint64_t target_frequency = 750000000l;   /* 750 MHz: 1 GHz didn't lock on dsp25 silicon (hung at
                                           * init_test). The combined KWS+Llama demo supplies its own
                                           * frequency, which is why boraiq only hung standalone. */
#endif

// #ifdef ENABLE_DMA_MATVEC
// int32_t GS_MATVEC_BOUND = 0;

// #endif

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */


/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN PUC */

// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    int8_t* q;    // quantized values
    float s; // scaling factors
} QuantizedTensor;

typedef struct {
    // token embedding table
    QuantizedTensor *q_tokens; // (vocab_size, dim)
    float* token_embedding_table; // same, but dequantized

    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    QuantizedTensor *wq; // (layer, dim, n_heads * head_size)
    QuantizedTensor *wk; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wv; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    QuantizedTensor *w1; // (layer, hidden_dim, dim)
    QuantizedTensor *w2; // (layer, dim, hidden_dim)
    QuantizedTensor *w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    QuantizedTensor *wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    QuantizedTensor xq; // quantized x (dim,)
    QuantizedTensor hq; // quantized hb (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
#ifdef TRANSPOSED_WEIGHTS
    TransformerWeightsT weights_t; // transposed B_pack copies for vectorized inference
#endif
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->xq = (QuantizedTensor) { .q = calloc(p->dim, sizeof(int8_t)), .s = 0.0f };
    s->hq = (QuantizedTensor) { .q = calloc(p->hidden_dim, sizeof(int8_t)), .s = 0.0f };
    s->q = calloc(p->dim, sizeof(float));
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->k || !s->v || !s->att || !s->logits || !s->key_cache
     || !s->value_cache) {
        printf("STDERR: malloc failed!\r\n");
        printf("size: %d\r\n", p->n_layers * p->seq_len * kv_dim * sizeof(float));
        printf("s->q: %x\r\n", s->q);
        printf("key_cache: %x\r\n", s->key_cache);

        // exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->xq.q);
    free(s->hq.q);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

// ----------------------------------------------------------------------------
// Quantization functions

void dequantize(QuantizedTensor *qx, float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = qx->q[i] * qx->s;
    }
}

void quantize(QuantizedTensor *qx, float* x, int n) {
    float Q_MAX = 127.0f;

    // find the max absolute value
    float wmax = 0.0;
    for (int i = 0; i < n; i++) {
        float val = fabs(x[i]);
        if (val > wmax) {
            wmax = val;
        }
    }

    // calculate and write the scaling factor
    float scale = wmax / Q_MAX;
    qx->s = scale;

    // calculate and write the quantized values
    for (int i = 0; i < n; i++) {
        float quant_value = x[i] / scale; // scale
        int8_t quantized = (int8_t) round(quant_value); // round and clamp
        qx->q[i] = quantized;
    }
}

/* initialize `n` x quantized tensor (with `size_each` elements), starting from memory pointed at *ptr */
QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each) {
    void *p = *ptr;
    QuantizedTensor *res = malloc(n * sizeof(QuantizedTensor));
    for(int i=0; i<n; i++) {
        /* map quantized int8 values*/
        res[i].q = (int8_t*)p;
        p = (int8_t*)p + size_each;
        /* map scale factors */
        res[i].s = *(float*)p;
        p = (float*)p + 1;
    }
    *ptr = p; // advance ptr to current position
    return res;
}

void memory_map_weights(TransformerWeights *w, Config* p, void* ptr, uint8_t shared_classifier) {
    int head_size = p->dim / p->n_heads;
    // first are the parameters that are kept in fp32 (the rmsnorm (1D) weights)
    float* fptr = (float*) ptr; // cast our pointer to float*
    w->rms_att_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_ffn_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_final_weight = fptr;
    fptr += p->dim;

    // now read all the quantized weights
    ptr = (void*)fptr; // now cast the pointer back to void*
    w->q_tokens = init_quantized_tensors(&ptr, 1, p->vocab_size * p->dim);
    // dequantize token embedding table
    w->token_embedding_table = malloc(p->vocab_size * p->dim * sizeof(float));
    dequantize(w->q_tokens, w->token_embedding_table, p->vocab_size * p->dim);

    w->wq = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_heads * head_size));
    w->wk = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wv = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wo = init_quantized_tensors(&ptr, p->n_layers, (p->n_heads * head_size) * p->dim);

    w->w1 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);
    w->w2 = init_quantized_tensors(&ptr, p->n_layers, p->hidden_dim * p->dim);
    w->w3 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);

    w->wcls = shared_classifier ? w->q_tokens : init_quantized_tensors(&ptr, 1, p->dim * p->vocab_size);
}


// void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
//                      int* fd, float** data, ssize_t* file_size) {
//     FILE *file = fopen(checkpoint, "rb");
//     if (!file) { printf("STDERR: Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
//     // read in the config header
//     if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
//     // negative vocab size is hacky way of signaling unshared weights. bit yikes.
//     int shared_weights = config->vocab_size > 0 ? 1 : 0;
//     config->vocab_size = abs(config->vocab_size);
//     // figure out the file size
//     fseek(file, 0, SEEK_END); // move file pointer to end of file
//     *file_size = ftell(file); // get the file size, in bytes
//     fclose(file);
//     // memory map the Transformer weights into the data pointer
//     *fd = open(checkpoint, O_RDONLY); // open in read only mode
//     if (*fd == -1) { printf("STDERR: open failed!\n"); exit(EXIT_FAILURE); }
//     *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
//     if (*data == MAP_FAILED) { printf("STDERR: mmap failed!\n"); exit(EXIT_FAILURE); }
//     float* weights_ptr = *data + sizeof(Config)/sizeof(float);
//     memory_map_weights(weights, config, weights_ptr, shared_weights);
// }

void read_checkpoint_from_header(Config* config, TransformerWeights* weights, float** data, size_t* file_size) {
  size_t cumulative_offset = 0;
  // Check magic number
  uint32_t magic = *((uint32_t*)(WEIGHTS + cumulative_offset));
  if (magic != MODEL_MAGIC_NUMBER) {
    printf("ERROR: Model magic number mismatch!\r\n");
  }
  cumulative_offset += sizeof(uint32_t);

  uint32_t version = *((uint32_t*)(WEIGHTS + cumulative_offset));
  if (version != MODEL_VERSION_INT8) {
    printf("ERROR: Model version is not Int8 Quantized. Version (hex) = %x\r\n", version);
  }
  cumulative_offset += sizeof(uint32_t);

  // load from weights.h WEIGHTS
  memcpy(config, (WEIGHTS + cumulative_offset), sizeof(Config));
  cumulative_offset += sizeof(Config);

  // Check shared classifier byte
  uint8_t shared_classifier = *((uint8_t*)(WEIGHTS + cumulative_offset));
  cumulative_offset += sizeof(uint8_t);

#ifdef ENABLE_QT_DOTPROD
  GS_QTDP_BOUND = GS / 8 * 8;
#endif

  // int shared_weights = config->vocab_size > 0 ? 1 : 0;
  config->vocab_size = abs(config->vocab_size);
  *file_size = sizeof(Config) + sizeof(WEIGHTS);

  // Point to WEIGHTS array as data
  *data = (float*)WEIGHTS;
  void* weights_ptr = ((char*)WEIGHTS) + MODEL_V2_HEADER_SIZE;
  memory_map_weights(weights, config, weights_ptr, shared_classifier);
}

void build_transformer(Transformer *t) {
    read_checkpoint_from_header(&t->config, &t->weights, &t->data, &t->file_size);
    malloc_run_state(&t->state, &t->config);
#ifdef TRANSPOSED_WEIGHTS
    alloc_and_transpose_weights_i8(&t->weights_t, &t->weights, &t->config);
#endif
}

void free_transformer(Transformer* t) {
#ifdef TRANSPOSED_WEIGHTS
    free_transposed_weights_i8(&t->weights_t);
#endif
  // Free quantized tensors
    free(t->weights.q_tokens);
    free(t->weights.token_embedding_table);
    free(t->weights.wq);
    free(t->weights.wk);
    free(t->weights.wv);
    free(t->weights.wo);
    free(t->weights.w1);
    free(t->weights.w2);
    free(t->weights.w3);
    if(t->weights.wcls != t->weights.q_tokens) { free(t->weights.wcls); }
    // close the memory mapping
    // if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    // if (t->fd != -1) { close(t->fd); }
    // free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // inputs to this function are both quantized

    int i;
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;

        // do the matmul
        int j;
        for (j = 0; j <= n; j++) {
            ival += ((int32_t) x->q[j]) * ((int32_t) w->q[in + j]);
        }

        xout[i] = ((float) ival) * w->s * x->s;
    }
}

#ifdef TRANSPOSED_WEIGHTS
/* ---------------------------------------------------------------------------
 * make_b_pack_i8 — build a B_pack from a quantized weight matrix.
 *
 * W is stored as W[n_out × n_in] row-major int8 (from QuantizedTensor.q).
 * Output B_pack is [(n_in+1) × n_out] bytes:
 *   Row 0       : n_out zero bytes  (zero bias)
 *   Rows 1..K   : rows of W_T as int8  (W_T[k][j] = W[j*n_in+k])
 *
 * The weights are already signed int8 from QuantizedTensor, so no bias-128
 * subtraction is needed (unlike uint8-biased formats).
 * ------------------------------------------------------------------------- */
static void make_b_pack_i8(
    unsigned char* pack,
    const int8_t* W,
    int n_out, int n_in)
{
    int8_t* ipack = (int8_t*)pack;
    /* bias row: zeros */
    for (int j = 0; j < n_out; j++)
        ipack[j] = 0;
    /* weight rows: W_T */
    for (int k = 0; k < n_in; k++)
        for (int j = 0; j < n_out; j++)
            ipack[(k + 1) * n_out + j] = W[j * n_in + k];
}

/* ---------------------------------------------------------------------------
 * alloc_and_transpose_weights_i8 — allocate and fill transposed B_packs.
 * Called once in build_transformer().
 * ------------------------------------------------------------------------- */
static void* checked_malloc(size_t size, const char* name) {
    void* ptr = malloc(size);
    if (!ptr) {
        printf("FATAL: malloc failed for %s (%u bytes)\r\n", name, (unsigned)size);
        while (1) {}
    }
    return ptr;
}

void alloc_and_transpose_weights_i8(
    TransformerWeightsT* wt,
    const TransformerWeights* w,
    const Config* p)
{
    int dim        = p->dim;
    int hidden_dim = p->hidden_dim;
    int n_layers   = p->n_layers;
    int kv_dim     = (p->dim * p->n_kv_heads) / p->n_heads;

    /* wq: W[dim × dim], B_pack [(dim+1) × dim] per layer */
    wt->wq_T = (unsigned char*)checked_malloc(n_layers * (size_t)(dim + 1) * dim, "wq_T");
    for (int l = 0; l < n_layers; l++)
        make_b_pack_i8(wt->wq_T + l * (size_t)(dim + 1) * dim,
                       w->wq[l].q, dim, dim);

    /* wk: W[kv_dim × dim], B_pack [(dim+1) × kv_dim] per layer */
    wt->wk_T = (unsigned char*)checked_malloc(n_layers * (size_t)(dim + 1) * kv_dim, "wk_T");
    for (int l = 0; l < n_layers; l++)
        make_b_pack_i8(wt->wk_T + l * (size_t)(dim + 1) * kv_dim,
                       w->wk[l].q, kv_dim, dim);

    /* wv: same layout as wk */
    wt->wv_T = (unsigned char*)checked_malloc(n_layers * (size_t)(dim + 1) * kv_dim, "wv_T");
    for (int l = 0; l < n_layers; l++)
        make_b_pack_i8(wt->wv_T + l * (size_t)(dim + 1) * kv_dim,
                       w->wv[l].q, kv_dim, dim);

    /* wo: W[dim × dim], same layout as wq */
    wt->wo_T = (unsigned char*)checked_malloc(n_layers * (size_t)(dim + 1) * dim, "wo_T");
    for (int l = 0; l < n_layers; l++)
        make_b_pack_i8(wt->wo_T + l * (size_t)(dim + 1) * dim,
                       w->wo[l].q, dim, dim);

    /* w1: W[hidden_dim × dim], B_pack [(dim+1) × hidden_dim] per layer */
    wt->w1_T = (unsigned char*)checked_malloc(n_layers * (size_t)(dim + 1) * hidden_dim, "w1_T");
    for (int l = 0; l < n_layers; l++)
        make_b_pack_i8(wt->w1_T + l * (size_t)(dim + 1) * hidden_dim,
                       w->w1[l].q, hidden_dim, dim);

    /* w2: W[dim × hidden_dim], B_pack [(hidden_dim+1) × dim] per layer */
    wt->w2_T = (unsigned char*)checked_malloc(n_layers * (size_t)(hidden_dim + 1) * dim, "w2_T");
    for (int l = 0; l < n_layers; l++)
        make_b_pack_i8(wt->w2_T + l * (size_t)(hidden_dim + 1) * dim,
                       w->w2[l].q, dim, hidden_dim);

    /* w3: same layout as w1 */
    wt->w3_T = (unsigned char*)checked_malloc(n_layers * (size_t)(dim + 1) * hidden_dim, "w3_T");
    for (int l = 0; l < n_layers; l++)
        make_b_pack_i8(wt->w3_T + l * (size_t)(dim + 1) * hidden_dim,
                       w->w3[l].q, hidden_dim, dim);

    /* wcls: W[vocab_size × dim], B_pack [(dim+1) × vocab_size] */
    wt->wcls_T = (unsigned char*)checked_malloc((size_t)(dim + 1) * p->vocab_size, "wcls_T");
    make_b_pack_i8(wt->wcls_T, w->wcls[0].q, p->vocab_size, dim);
}

void free_transposed_weights_i8(TransformerWeightsT* wt) {
    free(wt->wq_T);
    free(wt->wk_T);
    free(wt->wv_T);
    free(wt->wo_T);
    free(wt->w1_T);
    free(wt->w2_T);
    free(wt->w3_T);
    free(wt->wcls_T);
}

/* ---------------------------------------------------------------------------
 * matmul_t — vectorized matmul using pre-transposed B_pack.
 *
 * Computes xout(n_out,) = W(n_out,n_in) @ xq(n_in,) exactly like matmul(),
 * but w_t_pack is W stored as B_pack (transposed + bias row).
 *
 * Reformulated as: xout(1,n_out) = xq(1,n_in) @ W_T(n_in,n_out)
 *   → int8_qgemm_fout(M=1, N=n_out, K=n_in) — vectorises over n_out.
 * ------------------------------------------------------------------------- */
static void matmul_t(
    float* xout,
    QuantizedTensor* xq,
    const void* w_t_pack,
    float w_scale,
    int n_in, int n_out)
{
    float scale = xq->s * w_scale;
    quant_fully_connected_int8_t(
        (size_t)n_in, (size_t)n_out, 1,
        xq->q, w_t_pack, xout, scale);
}
#endif /* TRANSPOSED_WEIGHTS */

// ----------------------------------------------------------------------------
// Types needed by both single-core and multicore paths

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex* probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

// Forward declarations of functions used by generate paths
void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens);
char* decode(Tokenizer* t, int prev_token, int token);
int sample(Sampler* sampler, float* logits);

// ----------------------------------------------------------------------------
// Multicore prefill support
#ifdef PREFILL_MULTICORE

static Transformer  _mc_transformer;   /* shared transformer, built by hart 0 */
static volatile int _mc_token_val = 0;
static volatile int _mc_pos_val   = 0;
static volatile int _mc_fwd_rdy   = 0; /* hart 0 sets → hart 1 starts forward */
static volatile int _mc_all_done  = 0; /* hart 0 sets → hart 1 exits worker  */
static volatile int _mc_init_done = 0; /* hart 0 sets after init; hart 1 waits */

/* 2-hart sense-reversing barrier.
 * Hart 0: waits for hart 1 to arrive, then flips the shared sense to release.
 * Hart 1: signals arrival, then waits for sense to flip.
 *
 * fence instructions ensure cross-hart visibility on RISC-V RVWMO. */
static volatile int _bar_h1_arrived = 0;
static volatile int _bar_sense      = 0;

static void barrier2(int hartid) {
    int s = _bar_sense;
    if (hartid == 0) {
        while (!_bar_h1_arrived) { asm volatile("fence" ::: "memory"); }
        _bar_h1_arrived = 0;
        asm volatile("fence" ::: "memory");
        _bar_sense = !s;
        asm volatile("fence" ::: "memory");
    } else {
        _bar_h1_arrived = 1;
        asm volatile("fence" ::: "memory");
        while (_bar_sense == s) { asm volatile("fence" ::: "memory"); }
    }
}

/* Scalar matmul over output rows [row_start, row_end).
 * Computes xout[i] = W[i,:] dot x  for i in [row_start, row_end). */
static void matmul_rows(float* xout, QuantizedTensor* x, QuantizedTensor* w,
                        int n, int d, int row_start, int row_end) {
    (void)d;
    for (int i = row_start; i < row_end; i++) {
        int32_t ival = 0;
        int in = i * n;
        for (int j = 0; j < n; j++) {
            ival += (int32_t)x->q[j] * (int32_t)w->q[in + j];
        }
        xout[i] = (float)ival * w->s * x->s;
    }
}

/* forward_mc — parallel forward pass for two harts.
 *
 * Mutual-exclusion assignment: each independent matmul is owned exclusively
 * by one hart, so both harts run full-size (optionally vectorized) kernels
 * simultaneously rather than each computing half-rows of a single matrix.
 *
 * Assignment per layer:
 *   QKV  : hart 0 → wq + wk  |  hart 1 → wv
 *   Attn : both — each handles n_heads/2 heads
 *   FFN  : hart 0 → w1       |  hart 1 → w3  (key parallelism win)
 *   wcls : hart 0 vectorized (TRANSPOSED_WEIGHTS) or both split rows (scalar)
 *
 * All sequential work (rmsnorm, RoPE, kv-store, residual, SwiGLU, w2, wcls
 * when transposed) is done by hart 0; hart 1 spins at barriers.
 *
 * When TRANSPOSED_WEIGHTS is also defined, w1/w3/wq/wk/wv/wo all use the
 * vectorized matmul_t kernel on their full matrices.
 *
 * Returns logits (hart 0) or NULL (hart 1). */
static float* forward_mc(Transformer* transformer, int token, int pos, int hartid) {
    Config* p             = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s           = &transformer->state;
#ifdef TRANSPOSED_WEIGHTS
    TransformerWeightsT* wt = &transformer->weights_t;
#endif
    float* x        = s->x;
    int dim         = p->dim;
    int kv_dim      = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul      = p->n_heads / p->n_kv_heads;
    int hidden_dim  = p->hidden_dim;
    int head_size   = dim / p->n_heads;

    /* embedding copy (hart 0 only; hart 1 waits at first barrier in the loop) */
    if (hartid == 0) {
        memcpy(x, w->token_embedding_table + token * dim, dim * sizeof(float));
    }

    for (int l = 0; l < p->n_layers; l++) {

        /* rmsnorm + quantize input (hart 0) */
        if (hartid == 0) {
            rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);
            quantize(&s->xq, s->xb, dim);
        }
        barrier2(hartid);

        /* QKV projections — mutual exclusion:
         *   hart 0 → wq (s->q) + wk (s->k)
         *   hart 1 → wv (s->v)                              */
        if (hartid == 0) {
#ifdef TRANSPOSED_WEIGHTS
            matmul_t(s->q, &s->xq, wt->wq_T + l*(size_t)(dim+1)*dim,    w->wq[l].s, dim, dim);
            matmul_t(s->k, &s->xq, wt->wk_T + l*(size_t)(dim+1)*kv_dim, w->wk[l].s, dim, kv_dim);
#else
            matmul(s->q, &s->xq, w->wq + l, dim, dim);
            matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);
#endif
        } else {
#ifdef TRANSPOSED_WEIGHTS
            matmul_t(s->v, &s->xq, wt->wv_T + l*(size_t)(dim+1)*kv_dim, w->wv[l].s, dim, kv_dim);
#else
            matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);
#endif
        }
        barrier2(hartid);

        /* RoPE relative positional encoding + KV cache store (hart 0) */
        if (hartid == 0) {
            for (int i = 0; i < dim; i += 2) {
                int head_dim = i % head_size;
                float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
                float val  = pos * freq;
                float fcr  = cosf(val);
                float fci  = sinf(val);
                int rotn = i < kv_dim ? 2 : 1;
                for (int v = 0; v < rotn; v++) {
                    float* vec = v == 0 ? s->q : s->k;
                    float v0 = vec[i];
                    float v1 = vec[i + 1];
                    vec[i]     = v0 * fcr - v1 * fci;
                    vec[i + 1] = v0 * fci + v1 * fcr;
                }
            }
            int loff = l * p->seq_len * kv_dim;
            memcpy(s->key_cache   + loff + pos * kv_dim, s->k, kv_dim * sizeof(float));
            memcpy(s->value_cache + loff + pos * kv_dim, s->v, kv_dim * sizeof(float));
        }
        barrier2(hartid);

        /* multihead attention — each hart owns n_heads/2 heads exclusively */
        {
            int h_start = hartid * (p->n_heads / 2);
            int h_end   = h_start + p->n_heads / 2;
            int loff    = l * p->seq_len * kv_dim;
            for (int h = h_start; h < h_end; h++) {
                float* q   = s->q   + h * head_size;
                float* att = s->att + h * p->seq_len;
                for (int t = 0; t <= pos; t++) {
                    float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                    float score = 0.0f;
                    for (int i = 0; i < head_size; i++) score += q[i] * k[i];
                    att[t] = score / sqrtf(head_size);
                }
#ifdef VEC_SOFTMAX
                softmax_vec(att, att, 1, (size_t)(pos + 1));
#else
                softmax(att, pos + 1);
#endif
                float* xb = s->xb + h * head_size;
                memset(xb, 0, head_size * sizeof(float));
                for (int t = 0; t <= pos; t++) {
                    float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                    float a  = att[t];
                    for (int i = 0; i < head_size; i++) xb[i] += a * v[i];
                }
            }
        }
        barrier2(hartid);

        /* wo matmul + residual + FFN rmsnorm + quantize (hart 0) */
        if (hartid == 0) {
            quantize(&s->xq, s->xb, dim);
#ifdef TRANSPOSED_WEIGHTS
            matmul_t(s->xb2, &s->xq, wt->wo_T + l*(size_t)(dim+1)*dim, w->wo[l].s, dim, dim);
#else
            matmul(s->xb2, &s->xq, w->wo + l, dim, dim);
#endif
            for (int i = 0; i < dim; i++) x[i] += s->xb2[i];
            rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);
            quantize(&s->xq, s->xb, dim);
        }
        barrier2(hartid);

        /* FFN w1 + w3 — mutual exclusion (biggest parallelism win):
         *   hart 0 → w1 (s->hb)   hart 1 → w3 (s->hb2)
         * Both run simultaneously on their full output buffer. */
        if (hartid == 0) {
#ifdef TRANSPOSED_WEIGHTS
            matmul_t(s->hb,  &s->xq, wt->w1_T + l*(size_t)(dim+1)*hidden_dim, w->w1[l].s, dim, hidden_dim);
#else
            matmul(s->hb,  &s->xq, w->w1 + l, dim, hidden_dim);
#endif
        } else {
#ifdef TRANSPOSED_WEIGHTS
            matmul_t(s->hb2, &s->xq, wt->w3_T + l*(size_t)(dim+1)*hidden_dim, w->w3[l].s, dim, hidden_dim);
#else
            matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);
#endif
        }
        barrier2(hartid);

        /* SwiGLU + quantize hq + w2 matmul + residual (hart 0) */
        if (hartid == 0) {
            for (int i = 0; i < hidden_dim; i++) {
                float val = s->hb[i];
                val *= 1.0f / (1.0f + expf(-val));
                val *= s->hb2[i];
                s->hb[i] = val;
            }
            quantize(&s->hq, s->hb, hidden_dim);
#ifdef TRANSPOSED_WEIGHTS
            matmul_t(s->xb, &s->hq, wt->w2_T + l*(size_t)(hidden_dim+1)*dim, w->w2[l].s, hidden_dim, dim);
#else
            matmul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);
#endif
            for (int i = 0; i < dim; i++) x[i] += s->xb[i];
        }
        barrier2(hartid);
    }

    /* final rmsnorm + quantize (hart 0) */
    if (hartid == 0) {
        rmsnorm(x, x, w->rms_final_weight, dim);
        quantize(&s->xq, x, dim);
    }
    barrier2(hartid);

    /* wcls:
     *   TRANSPOSED_WEIGHTS → hart 0 runs full vectorized matmul_t; hart 1 idles.
     *   scalar             → both harts split vocab_size rows (mutual exclusion
     *                        via non-overlapping [0, vs_half) / [vs_half, end)). */
#ifdef TRANSPOSED_WEIGHTS
    if (hartid == 0) {
        matmul_t(s->logits, &s->xq, wt->wcls_T, w->wcls[0].s, dim, p->vocab_size);
    }
#else
    {
        int vs_half  = p->vocab_size / 2;
        int vs_start = hartid * vs_half;
        int vs_end   = vs_start + vs_half;
        matmul_rows(s->logits, &s->xq, w->wcls, dim, p->vocab_size, vs_start, vs_end);
    }
#endif
    barrier2(hartid);

    return hartid == 0 ? s->logits : NULL;
}

/* generate_mc — same as generate() but uses forward_mc(hartid=0) and
 * signals hart 1 before each token via _mc_fwd_rdy. */
void generate_mc(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
                 char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt) + 3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        printf("STDERR: something is wrong, expected at least 1 prompt token\r\n");
    }

    unsigned long start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos   = 0;
    while (pos < steps) {
        /* publish token/pos, then signal hart 1 to start its half */
        _mc_token_val = token;
        _mc_pos_val   = pos;
        asm volatile("fence" ::: "memory");
        _mc_fwd_rdy   = 1;
        asm volatile("fence" ::: "memory");
        float* logits = forward_mc(transformer, token, pos, 0);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, logits);
        }
        pos++;

        if (next == 1) { break; }

        char* piece = decode(tokenizer, token, next);
        safe_printf(piece);
        fflush(stdout);
        token = next;

        if (start == 0) { start = READ_CSR("mcycle"); }
    }
    printf("\r\n");

    _mc_all_done = 1;  /* signal hart 1 worker to exit */

    if (pos > 1) {
        unsigned long end = READ_CSR("mcycle");
        unsigned long elapsed = end - start;
        int toks = pos - 1;
        printf("\r\n--- %d tokens | %lu cycles | %.2f tok/s @ %u Hz ---\r\n",
               toks, elapsed,
               (double)toks / ((double)elapsed / target_frequency),
               target_frequency);
    }

    free(prompt_tokens);
}

#endif /* PREFILL_MULTICORE */

float* forward(Transformer* transformer, int token, int pos) {

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
#ifdef TRANSPOSED_WEIGHTS
    TransformerWeightsT* wt = &transformer->weights_t;
#endif
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    memcpy(x, w->token_embedding_table + token*dim, dim * sizeof(float));

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // qkv matmuls for this position
        quantize(&s->xq, s->xb, dim);
#ifdef TRANSPOSED_WEIGHTS
        matmul_t(s->q, &s->xq, wt->wq_T + l*(size_t)(dim+1)*dim,    w->wq[l].s, dim, dim);
        matmul_t(s->k, &s->xq, wt->wk_T + l*(size_t)(dim+1)*kv_dim, w->wk[l].s, dim, kv_dim);
        matmul_t(s->v, &s->xq, wt->wv_T + l*(size_t)(dim+1)*kv_dim, w->wv[l].s, dim, kv_dim);
#else
        matmul(s->q, &s->xq, w->wq + l, dim, dim);
        matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);
        matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);
#endif

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // save key,value at this time step (pos) to our kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        float* value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));

        // multihead attention. iterate over all heads
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
#ifdef VEC_SOFTMAX
            softmax_vec(att, att, 1, (size_t)(pos + 1));
#else
            softmax(att, pos + 1);
#endif

            // weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        quantize(&s->xq, s->xb, dim);
#ifdef TRANSPOSED_WEIGHTS
        matmul_t(s->xb2, &s->xq, wt->wo_T + l*(size_t)(dim+1)*dim, w->wo[l].s, dim, dim);
#else
        matmul(s->xb2, &s->xq, w->wo + l, dim, dim);
#endif

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        quantize(&s->xq, s->xb, dim);
#ifdef TRANSPOSED_WEIGHTS
        matmul_t(s->hb,  &s->xq, wt->w1_T + l*(size_t)(dim+1)*hidden_dim,    w->w1[l].s, dim, hidden_dim);
        matmul_t(s->hb2, &s->xq, wt->w3_T + l*(size_t)(dim+1)*hidden_dim,    w->w3[l].s, dim, hidden_dim);
#else
        matmul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);
#endif

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        quantize(&s->hq, s->hb, hidden_dim);
#ifdef TRANSPOSED_WEIGHTS
        matmul_t(s->xb, &s->hq, wt->w2_T + l*(size_t)(hidden_dim+1)*dim, w->w2[l].s, hidden_dim, dim);
#else
        matmul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);
#endif

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    quantize(&s->xq, x, dim);
#ifdef TRANSPOSED_WEIGHTS
    matmul_t(s->logits, &s->xq, wt->wcls_T, w->wcls[0].s, dim, p->vocab_size);
#else
    matmul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
#endif
    return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer_from_header(Tokenizer* t, int vocab_size) {
  // Potential point of improvement: Write the vocab size into the tokenizer file.
  t->vocab_size = vocab_size;
  // malloc space to hold the scores and the strings
  t->vocab = (char**)malloc(vocab_size * sizeof(char*));
  t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
  t->sorted_vocab = NULL; // initialized lazily
  for (int i = 0; i < 256; i++) {
    t->byte_pieces[i * 2] = (unsigned char)i;
    t->byte_pieces[i * 2 + 1] = '\0';
  }

  void *tok_ptr = TOKENIZER;
  // read in from TOKENIZER in tokenizer.h
  t->max_token_length = *(int *)tok_ptr;
  tok_ptr += sizeof(int);

  int len;
  for (int i = 0; i < vocab_size; i++) {
    // read into (t->vocab_scores + i)
    memcpy(t->vocab_scores + i, tok_ptr, sizeof(float));
    tok_ptr += sizeof(float);

    // read into len
    memcpy(&len, tok_ptr, sizeof(int));
    tok_ptr += sizeof(int);
    
    t->vocab[i] = (char *)malloc(len + 1);
    memcpy(t->vocab[i], tok_ptr, len);
    tok_ptr += len;
    t->vocab[i][len] = '\0'; // add the string null terminator
  }
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte

    // sscanf replaced with not sscanf
    unsigned char byte_val;
    if (piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' && piece[5] == '>') {
        char hex[3];
        hex[0] = piece[3];
        hex[1] = piece[4];
        hex[2] = '\0';
        byte_val = (unsigned char) strtol(hex, NULL, 16);
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { printf("STDERR: cannot encode NULL text\r\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time

#define CURRENT_TIME_IN_SECONDS (READ_CSR("mcycle") / MTIME_FREQ)

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    return (READ_CSR("mcycle") / MTIME_FREQ);
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        printf("STDERR: something is wrong, expected at least 1 prompt token\r\n");
        // exit(EXIT_FAILURE);
    }

    // start the main loop
    unsigned long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {

#ifdef KWS_LLAMA_COMBINED
        /* Combined demo: abort mid-generation when the control hart raises the STOP flag (a
         * no/off/stop keyword was recognized). Checked once per token. */
        if (g_llama_stop) { break; }
#endif

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = READ_CSR("mcycle"); }
    }
    printf("\r\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        unsigned long end = READ_CSR("mcycle");
        unsigned long elapsed = end - start;
        int toks = pos - 1;
        printf("\r\n--- %d tokens | %lu cycles | %.2f tok/s @ %u Hz ---\r\n",
               toks, elapsed,
               (double)toks / ((double)elapsed / target_frequency),
               target_frequency);
    }

    free(prompt_tokens);
}


void read_stdin(const char* guide, char* buffer, size_t bufsize) {
  // read a line from stdin, up to but not including \n
  // printf("%s", guide);
  // if (fgets(buffer, bufsize, stdin) != NULL) {
  //     size_t len = strlen(buffer);
  //     if (len > 0 && buffer[len - 1] == '\n') {
  //         buffer[len - 1] = '\0'; // strip newline
  //     }
  // }
  size_t char_offset = 0;
  size_t upper_bufsize_bound = bufsize - 1;
  printf("%s", guide);
  fflush(stdout);

  while(1) {
    unsigned char input_char = '\0';
    uart_receive(UART0, &input_char, 1, 100);

    if (input_char == '\b' && char_offset > 0) {
      // Backspace handling
      uart_transmit(UART0, &input_char, 1, 100);
      char_offset--;
      *(buffer + char_offset) = '\0';
    } else if (input_char == '\r') {
      // Newline (submit) handling
      uart_transmit(UART0, ASCII_CRLF, 2, 100);
      *(buffer + char_offset) = '\0';
      break;
    } else if (input_char > 31 && input_char < 127 && char_offset < upper_bufsize_bound) {
      // Any printable ascii character
      uart_transmit(UART0, &input_char, 1, 100);
      *(buffer + char_offset) = input_char;
      char_offset++;
    } else {
      // Send a bell to the terminal if some other condition.
      uart_transmit(UART0, ASCII_BEL, 1, 100);
    }
  }
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {

    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are soomewhat haphazardly and unsafely set atm
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx;

    // start the main loop
    int8_t user_turn = 1; // user starts
    unsigned long start = 0;  // used to time our code, only initialized after first iteration

    int next;        // will store the next token in the sequence
    int token;       // stores the current token to feed into the transformer
    int prev_token;
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            // get the (optional) system prompt at position 0
            if (pos == 0) {
                // at position 0, the user can also contribute a system prompt
                if (cli_system_prompt == NULL) {
                    // system prompt was not passed in, attempt to get it from stdin
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    // system prompt was passed in, use it
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            // get the user prompt
            if (pos == 0 && cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt, cli_user_prompt);
            } else {
                // otherwise get user prompt from stdin
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            // render user/system prompts into the Llama 2 Chat schema
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\r\n%s\r\n<</SYS>>\r\n\r\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            // encode the rendered prompt into tokens
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = 0;
            printf("Assistant: ");
        }

        // determine the token to pass into the transformer next
        if (user_idx < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[user_idx++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }
        // EOS (=2) token ends the Assistant turn
        // if (token == 2) {
        //     unsigned long end = READ_CSR("mcycle");
        //     printf("\r\nBENCHMARK: Total cycles: %lu\r\n", end-start);
        //     printf("BENCHMARK: Total tokens:\t%d\r\n", pos-1);
        //     printf("BENCHMARK: Cycles per token:\t%lu\r\n", (unsigned long)(end-start)/(pos-1));
        //     printf("BENCHMARK: Seconds per token:\t%lu\r\n", (unsigned long)((end-start)/SYS_CLK_FREQ)/(pos-1));
        //     printf("BENCHMARK: Seconds per token (float):\t%f\r\n", ((float)(end-start)/(float)SYS_CLK_FREQ)/(float)(pos-1));

        //     printf("BENCHMARK: MTIME Frequency:\t%lu\r\n", MTIME_FREQ);
        //     user_turn = 1;
        //     start = 0;
        // }

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;
        
        if (user_idx >= num_prompt_tokens && next != 2) {
            // the Assistant is responding, so print its output
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
            if (start == 0) { start = READ_CSR("mcycle"); }
        }
        if (next == 2) { printf("\r\n"); }
    }

    if (pos > 1) {
        unsigned long end = READ_CSR("mcycle");
        unsigned long elapsed = end - start;
        int toks = pos - 1;
        printf("\r\n--- %d tokens | %lu cycles | %.2f tok/s @ %u Hz ---\r\n",
               toks, elapsed,
               (double)toks / ((double)elapsed / SYS_CLK_FREQ),
               SYS_CLK_FREQ);
    }
    free(prompt_tokens);
}


#ifndef KWS_LLAMA_COMBINED
void app_main() {
  uint64_t mhartid = READ_CSR("mhartid");
  printf("Started BorAIq (Int8 Quantized) Inference Engine on hart ID %lu\r\n", mhartid);

  // Parameters //
  float temperature = 0.8f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
  float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
  int steps = 512;           // number of steps to run for
  char *prompt = NULL;        // prompt string
  unsigned long long rng_seed = CLINT->MTIME; // seed rng with time by default
  GenMode mode = GENERATE;    // generate|chat
  char *system_prompt = NULL; // the (optional) system prompt to use in chat mode (I have it set up to ask screen if not given)

  // Parameter validation and overrides
  // if (rng_seed <= 0) rng_seed = CLINT->MTIME;
  if (temperature < 0.0) temperature = 0.0;
  if (topp < 0.0 || 1.0 < topp) topp = 0.9;
  if (steps < 0) steps = 0;

  // Import from transformer binary
#ifdef PREFILL_MULTICORE
  Transformer* p_tfm = &_mc_transformer;
#else
  Transformer _tfm_buf;
  Transformer* p_tfm = &_tfm_buf;
#endif
  build_transformer(p_tfm);
  if (steps == 0 || steps > p_tfm->config.seq_len) steps = p_tfm->config.seq_len;

  // Import the tokenizer binary
  Tokenizer tokenizer;
  build_tokenizer_from_header(&tokenizer, p_tfm->config.vocab_size);

  // build the Sampler
  Sampler sampler;
  build_sampler(&sampler, p_tfm->config.vocab_size, temperature, topp, rng_seed);

#ifdef PREFILL_MULTICORE
  // Signal hart 1 that initialization is complete and it's safe to poll
  asm volatile("fence" ::: "memory");
  _mc_init_done = 1;
  asm volatile("fence" ::: "memory");
#endif

  while (1) {
    sampler.rng_state = CLINT->MTIME;

    // run!
    if (mode == GENERATE) {
#ifdef PREFILL_MULTICORE
        _mc_all_done = 0;
        generate_mc(p_tfm, &tokenizer, &sampler, prompt, steps);
#else
        generate(p_tfm, &tokenizer, &sampler, prompt, steps);
#endif
    } else {
        chat(p_tfm, &tokenizer, &sampler, NULL, NULL, steps);
    }

    printf("========================================\r\n");
    //msleep(1000);
  }
}
#endif /* !KWS_LLAMA_COMBINED (app_main) */

/**
  * @brief  The application entry point.
  * @retval int
  */
#ifndef KWS_LLAMA_COMBINED
int main(int argc, char **argv) {
  /* MCU Configuration: PLL + clocks + UART0. Use the portable init_test() (bmark-lib/simple_setup)
   * so this builds on both bearly25 and dsp25 — the raw configure_pll/set_all_clocks(CLOCK_SELECTOR)
   * path was bearly-specific (dsp25 names it RCC_CLOCK_SELECTOR). init_test does the same PLL ratio
   * (target/SYS_CLK_FREQ) + 115200 UART setup. */
  init_test(target_frequency);
  app_main();
}
#endif /* !KWS_LLAMA_COMBINED (main) */


// Alternative HART runner. Multithreading, anyone?
#ifndef KWS_LLAMA_COMBINED
void __attribute__((noreturn)) __main(void) {
#ifdef PREFILL_MULTICORE
  unsigned long long hartid;
  asm volatile("csrr %0, mhartid" : "=r"(hartid));
  if (hartid == 1) {
    /* Wait for hart 0 to finish BSS init and transformer setup.
     * Without this gate, hart 1 can read _mc_fwd_rdy before BSS is
     * zeroed, enter forward_mc with garbage, and trap-hang. */
    while (!_mc_init_done) { asm volatile("fence" ::: "memory"); }

    /* Hart 1 worker: spin until hart 0 signals work, then execute the
     * hart-1 half of each forward pass alongside hart 0.
     * Outer loop re-arms after each generate_mc() round so hart 1
     * survives the app_main while(1) loop. */
    while (1) {
      asm volatile("fence" ::: "memory");
      if (_mc_fwd_rdy) {
        _mc_fwd_rdy = 0;
        asm volatile("fence" ::: "memory");
        forward_mc(&_mc_transformer, _mc_token_val, _mc_pos_val, 1);
      }
    }
  }
#endif
  while (1) {
    asm volatile ("wfi");
  }
}
#endif /* !KWS_LLAMA_COMBINED (__main) */

#ifdef KWS_LLAMA_COMBINED
/* ================================================================================================
 * Combined KWS + Llama demo entry points (see c2c-demos/bearly-kws-llama). hart 1 runs Llama
 * CONTINUOUSLY via llama_run_forever(); the control hart (hart 0) collects 3 recognized keywords,
 * writes a story prompt built from them into g_llama_prompt (seqlock-guarded by g_llama_prompt_ver),
 * and raises g_llama_stop to abort the in-flight story so hart 1 immediately picks up the new prompt.
 * ============================================================================================== */

#define KWS_LLAMA_PROMPT_MAX 256

/* Persistent model state (built once). Lives in shared DRAM so build-vs-run hart doesn't matter. */
static Transformer g_llama_tfm;
static Tokenizer   g_llama_tok;
static Sampler     g_llama_sampler;
static int         g_llama_built = 0;

/* STOP/abort flag: raised by the control hart when a new prompt is ready; polled per token. */
volatile int g_llama_stop = 0;

/* Current story prompt (hart0 writes, hart1 reads). Seqlock: g_llama_prompt_ver is odd while hart0
 * is mid-write, even when stable; hart1 re-copies if it changed during the read. Starts empty ->
 * the model free-runs a default story until the first 3 keywords arrive. */
char g_llama_prompt[KWS_LLAMA_PROMPT_MAX] = "";
volatile uint32_t g_llama_prompt_ver = 0;

/* Build the transformer/tokenizer/sampler once. Call from a single hart before generation starts
 * (the combined demo calls it at init on hart 0, before the KWS loop / hart 1 run — no malloc race). */
void llama_build(void) {
  if (g_llama_built) { return; }
  build_transformer(&g_llama_tfm);
  build_tokenizer_from_header(&g_llama_tok, g_llama_tfm.config.vocab_size);
  /* temperature 0.8, top-p 0.9 — borai's app_main defaults; seed re-rolled per story below. */
  build_sampler(&g_llama_sampler, g_llama_tfm.config.vocab_size, 0.8f, 0.9f,
                (unsigned long long)CLINT->MTIME);
  g_llama_built = 1;
  printf("[kws-llama] Llama model built (dim=%d layers=%d vocab=%d seq_len=%d)\r\n",
         g_llama_tfm.config.dim, g_llama_tfm.config.n_layers,
         g_llama_tfm.config.vocab_size, g_llama_tfm.config.seq_len);
}

/* Snapshot the shared prompt into a local buffer via the seqlock (retries on a concurrent write). */
static void llama_snapshot_prompt(char *dst, uint32_t n) {
  uint32_t v0;
  do {
    v0 = g_llama_prompt_ver;
    while (v0 & 1u) { v0 = g_llama_prompt_ver; }   /* hart0 mid-write -> wait for even */
    __sync_synchronize();
    uint32_t i = 0;
    for (; i < n - 1u && g_llama_prompt[i] != '\0'; ++i) { dst[i] = g_llama_prompt[i]; }
    dst[i] = '\0';
    __sync_synchronize();
  } while (g_llama_prompt_ver != v0);              /* changed during copy -> retry */
}

/* Run Llama forever. Each round: snapshot the current prompt, generate a story from it (aborting
 * early if the control hart raised g_llama_stop with a new prompt), then loop. So Llama is always
 * talking; when 3 new keywords land, the current story cuts off and the next one is about them. */
void llama_run_forever(void) {
  int steps = g_llama_tfm.config.seq_len;
  char local_prompt[KWS_LLAMA_PROMPT_MAX];
  printf("[kws-llama] === Llama running continuously (prompt updates every 3 keywords) ===\r\n");
  while (1) {
    llama_snapshot_prompt(local_prompt, sizeof(local_prompt));
    g_llama_stop = 0;
    __sync_synchronize();
    g_llama_sampler.rng_state = (unsigned long long)CLINT->MTIME;
    char *p = (local_prompt[0] != '\0') ? local_prompt : NULL;
    printf("[kws-llama] --- new story (prompt=\"%s\") ---\r\n", p ? local_prompt : "(default)");
    generate(&g_llama_tfm, &g_llama_tok, &g_llama_sampler, p, steps);
    printf("\r\n========================================\r\n");
  }
}
#endif /* KWS_LLAMA_COMBINED */