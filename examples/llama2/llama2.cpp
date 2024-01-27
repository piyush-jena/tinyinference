#include <tensor.h>
#include <embedding.h>
#include <mathlib.h>
#include <multi_head_attention.h>

#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <ctime>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

struct Config {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
};

struct TransformerWeights {
    // token embedding table
    embedding token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    tensor* rms_att_weight; // (layer, dim) rmsnorm weights
    tensor* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    multi_head_attention* attention;
    // (layer, dim, n_heads * head_size)
    // (layer, dim, n_kv_heads * head_size)
    // (layer, dim, n_kv_heads * head_size)

    tensor* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    tensor* w1; // (layer, hidden_dim, dim)
    tensor* w2; // (layer, dim, hidden_dim)
    tensor* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    tensor rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    tensor wcls;

    ~TransformerWeights() {
        delete[] rms_att_weight;
        delete[] rms_ffn_weight;

        delete[] attention;
        delete[] wo;

        delete[] w1;
        delete[] w2;
        delete[] w3;
    }
};

struct RunState {
    // current wave of activations
    tensor x; // activation at current time stamp (dim,)
    tensor xb; // same, but inside a residual branch (dim,)
    tensor xb2; // an additional buffer just for convenience (dim,)
    tensor hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    tensor hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    tensor q; // query (dim,)
    tensor k; // key (dim,)
    tensor v; // value (dim,)
    tensor att; // buffer for scores/attention values (n_heads, seq_len)
    tensor logits; // output logits
    // kv cache
    tensor* key_cache;   // (layer, seq_len, dim)
    tensor* value_cache; // (layer, seq_len, dim)

    RunState() {}

    RunState(Config config) {
        int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
        x = tensor{{1, config.dim}};
        xb = tensor{{1, config.dim}};
        xb2 = tensor{{1, config.dim}};

        hb = tensor{{1, config.hidden_dim}};
        hb2 = tensor{{1, config.hidden_dim}};

        q = tensor{{1, config.dim}};
        k = tensor{{1, config.dim}};
        v = tensor{{1, config.dim}};

        key_cache = new tensor[config.n_layers];
        value_cache = new tensor[config.n_layers];

        for (int i = 0 ; i < config.n_layers; i++) {
            key_cache[i] = tensor{{config.seq_len, kv_dim}};
            value_cache[i] = tensor{{config.seq_len, kv_dim}};
        }
    }

    ~RunState() {
        delete[] key_cache;
        delete[] value_cache;
    }
};

class llama2 {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass

    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes

    llama2();
    void build(char* checkpoint_path) {
        // read in the Config and the Weights from the checkpoint
        read_checkpoint(checkpoint_path, &config, &weights, &fd, &data, &file_size);
        // allocate the RunState buffers
        state = RunState(config);
    }

    void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
        FILE *file = fopen(checkpoint, "rb");
        if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
        // read in the config header
        if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
        // negative vocab size is hacky way of signaling unshared weights. bit yikes.
        int shared_weights = config->vocab_size > 0 ? 1 : 0;
        config->vocab_size = abs(config->vocab_size);
        // figure out the file size
        fseek(file, 0, SEEK_END); // move file pointer to end of file
        *file_size = ftell(file); // get the file size, in bytes
        fclose(file);
        // memory map the Transformer weights into the data pointer
        *fd = open(checkpoint, O_RDONLY); // open in read only mode
        if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
        *data = (float *)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
        if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
        float* weights_ptr = *data + sizeof(Config)/sizeof(float);
        memory_map_weights(weights, config, weights_ptr, shared_weights);
    }

    void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
        int head_size = p->dim / p->n_heads;
        // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
        unsigned long long n_layers = p->n_layers;
        w->token_embedding_table = embedding(ptr, p->vocab_size, p->dim);
        ptr += p->vocab_size * p->dim;

        w->rms_att_weight = new tensor[n_layers];
        for (int i = 0 ; i < n_layers ; i++) {
            w->rms_att_weight[i] = tensor{ptr, {1, p->dim}};
            ptr += p->dim;
        }

        w->attention = new multi_head_attention[n_layers];
        for (int i = 0 ; i < n_layers ; i++) {
            w->attention[i] = multi_head_attention(p->dim, p->n_heads, p->n_kv_heads);
        }

        for (int i = 0 ; i < n_layers ; i++) {
            w->attention[i].set_query(ptr);
        }

        ptr += n_layers * p->dim * (p->n_heads * head_size);

        for (int i = 0 ; i < n_layers ; i++) {
            w->attention[i].set_key(ptr);
        }

        ptr += n_layers * p->dim * (p->n_kv_heads * head_size);

        for (int i = 0 ; i < n_layers ; i++) {
            w->attention[i].set_value(ptr);
        }

        ptr += n_layers * p->dim * (p->n_kv_heads * head_size);

        w->wo = new tensor[n_layers];
        for (int i = 0 ; i < n_layers ; i++) {
            w->wo[i] = tensor{ptr, {(p->n_heads * head_size), p->dim}};
            ptr += (p->n_heads * head_size) * p->dim;
        }
        
        w->rms_ffn_weight = new tensor[n_layers];
        for (int i = 0 ; i < n_layers ; i++) {
            w->rms_ffn_weight[i] = tensor{ptr, {1, p->dim}};
            ptr += p->dim;
        }

        w->w1 = new tensor[n_layers];
        for (int i = 0 ; i < n_layers ; i++) {
            w->w1[i] = tensor{ptr, {p->dim, p->hidden_dim}};
            ptr += p->dim * p->hidden_dim;
        }
        w->w2 = new tensor[n_layers];
        for (int i = 0 ; i < n_layers ; i++) {
            w->w2[i] = tensor{ptr, {p->hidden_dim, p->dim}};
            ptr += p->dim * p->hidden_dim;
        }
        w->w3 = new tensor[n_layers];
        for (int i = 0 ; i < n_layers ; i++) {
            w->w3[i] = tensor{ptr, {p->dim, p->hidden_dim}};
            ptr += p->dim * p->hidden_dim;
        }

        w->rms_final_weight = tensor{ptr, {1, p->dim}};
        ptr += p->dim;

        ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
        ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
        
        w->wcls = shared_weights ? w->token_embedding_table : tensor{ptr, {p->vocab_size, p->dim}};
    }

    ~llama2() {
        // close the memory mapping
        if (data != MAP_FAILED) { munmap(data, file_size); }
        if (fd != -1) { close(fd); }
    }

    tensor forward(int token, int pos) {
        Config* p = &config;
        TransformerWeights* w = &weights;
        RunState* s = &state;
        s->x = w->token_embedding_table(token);
        int dim = p->dim;
        int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
        int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
        int hidden_dim =  p->hidden_dim;
        int head_size = dim / p->n_heads;

        tensor x = s->x;

        for(unsigned long long l = 0; l < p->n_layers; l++) {
            s->xb = rms_norm(x, w->rms_att_weight[l]);
            s->k = s->key_cache[l][pos];
            s->v = s->value_cache[l][pos];

            s->q = s->xb * w->wq[l];
            s->k = s->xb * w->wk[l];
            s->v = s->xb * w->wv[l];

            for (int i = 0 ; i < dim ; i += 2) {

            }

            for (int h = 0 ; h < p->n_heads ; h++) {
                tensor q = s->q[h];
                tensor att = s->att[h];

                for (int t = 0 ; t <= pos ; t++) {
                    tensor k = s->key_cache[l][t] + ...;
                    float score = 0.0f;
                    for (int i = 0 ; i < head_size ; i++) {
                        score += q(i) * k(i);
                    }
                    score /= sqrtf(head_size);
                    att(t) = score;
                }

                att = softmax(att); //verify

                tensor xb = s->xb[h];
                for (int i = 0 ; i < head_size ; i++) {
                    xb(i) = 0;
                }

                for (int t = 0 ; t <= pos ; t++) {
                    float* v = s->value_cache[l][t] + ...;
                    float a = att(t);
                    for (int i = 0 ; i < head_size ; i++) {
                        xb(i) += a * v(i);
                    }
                }
                
            }

            s->xb2 = s->xb * w->wo[l];
            x = x + s->xb2;

            s->xb = rms_norm(x, w->rms_ffn_weight[l]);
            s->hb = s->xb * w->w1[l];
            s->hb2 = s->xb * w->w3[l];

            // SwiGLU non-linearity
            s->hb = silu(s->hb);

            // final matmul to get the output of the ffn
            s->xb = s->hb * w->w2[l];
            x = x + s->xb;
        }

        x = rms_norm(x, w->rms_final_weight);
        s->logits = x * w->wcls;
        return s->logits;
    }
};