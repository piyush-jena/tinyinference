#include "mathlib.h"
#include "tensor.h"
#include "nn/embedding.h"

#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <ctime>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

struct Config {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
};

class attention {
    //config
    Config config;

    //weights
    tensor rms_att_weight;

    tensor query;
    tensor key;
    tensor value;
    tensor weight_o;

    tensor weight1;
    tensor weight2;
    tensor weight3;

    tensor rms_ffn_weight;

    //kv_cache
    tensor key_cache;
    tensor value_cache;

    public:
        attention() {}
        attention (Config config) : config{config} {}

        ssize_t set_rms_att_weight(float* w) {
            rms_att_weight = tensor{w, {1, config.dim}};
            return rms_att_weight.size();
        }

        ssize_t set_query(float* q) {
            int head_size = config.dim / config.n_heads;

            query = tensor(q, {config.dim, config.n_heads * head_size});
            return query.size();
        }

        ssize_t set_key(float* k) {
            int head_size = config.dim / config.n_heads;

            key = tensor(k, {config.dim, config.n_kv_heads * head_size});
            return key.size();
        }

        ssize_t set_value(float* v) {
            int head_size = config.dim / config.n_heads;

            value = tensor(v, {config.dim, config.n_kv_heads * head_size});
            return value.size();
        }

        ssize_t set_weight_o(float* w) {
            int head_size = config.dim / config.n_heads;

            weight_o = tensor(w, {config.n_heads * head_size, config.dim});
            return weight_o.size();
        }

        ssize_t set_ffn_weights1(float* w1) {
            weight1 = tensor{w1, {config.dim, config.hidden_dim}};
            return weight1.size();
        }

        ssize_t set_ffn_weights2(float* w2) {
            weight2 = tensor{w2, {config.hidden_dim, config.dim}};
            return weight2.size();
        }

        ssize_t set_ffn_weights3(float* w3) {
            weight3 = tensor{w3, {config.dim, config.hidden_dim}};
            return weight3.size();
        }

        ssize_t set_rms_ffn_weight(float* w) {
            rms_ffn_weight = tensor{w, {1, config.dim}};
            return rms_ffn_weight.size();
        }

        tensor forward(tensor& x, int pos) {
            int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
            int head_size = config.dim / config.n_heads;
            int kv_mul = config.n_heads / config.n_kv_heads;

            x = x * rms_att_weight;

            tensor k = key_cache.slice(pos * kv_dim, {1, config.n_kv_heads * head_size});
            tensor v = value_cache.slice(pos * kv_dim, {1, config.n_kv_heads * head_size});

            // qkv matmuls for this position
            tensor q = x * query;
            k = x * key;
            v = x * value;

            q.reshape({config.n_heads, head_size});
            k.reshape({config.n_kv_heads, head_size});
            v.reshape({config.n_kv_heads, head_size});

            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            for (int i = 0 ; i < config.dim ; i += 2) {
                int h = i % head_size;
                float freq = 1.0f / powf(10000.0f, h / (float)head_size);
                float val = pos * freq;
                float fcr = cosf(val);
                float fci = sinf(val);
                int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
                for (int v = 0 ; v < rotn ; v++) {
                    if (v == 0) {
                        float v0 = q(i);
                        float v1 = q(i+1);

                        q(i) = v0 * fcr - v1 * fci;
                        q(i+1) = v0 * fci + v1 * fcr;
                    } else {
                        float v0 = k(i);
                        float v1 = k(i+1);

                        k(i) = v0 * fcr - v1 * fci;
                        k(i+1) = v0 * fci + v1 * fcr;
                    }
                }
            }

            tensor att{{1, pos+1}};
            tensor xb{{1, head_size}, 0.0f};

            for (int h = 0 ; h < config.n_heads ; h++) {
                tensor _q = q[h];
                
                for (int t = 0 ; t <= pos ; t++) {
                    tensor _k = key_cache.slice(t * kv_dim + (h / kv_mul) * head_size, {});

                    float score = 0.0f;
                    for (int i = 0 ; i < head_size ; i++) {
                        score += _q(i) * _k(i);
                    }

                    score /= sqrtf(head_size);
                    // save the score to the attention buffer
                    att(t) = score;
                }

                att = softmax(att);

                for (int t = 0 ; t <= pos ; t++) {
                    tensor _v = value_cache.slice(t * kv_dim + (h / kv_mul) * head_size, {}); // from value_cache
                    float a = att(t);

                    for (int i = 0 ; i < head_size ; i++) {
                        xb(i) = xb(i) + a * v(i);
                    }
                }
            }

            tensor xb2 = xb * weight_o;

            x = x + xb2;
            xb = rms_norm(x, rms_ffn_weight);

            tensor hb = xb * weight1;
            tensor hb2 = xb * weight3;

            hb = silu(hb) * hb2;
            xb = hb * weight2;
            x = x + xb;

            return x;
        }

        tensor operator() (tensor& x, int pos) {
            return forward(x, pos);
        }
};

class llama2 {
    Config config; // the hyperparameters of the architecture (the blueprint)

    embedding token_embedding_table;
    attention* multi_head_attention;
    tensor wcls;
    tensor rms_final_weight;

    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes

    llama2();

    llama2(char* checkpoint_path) {
        read_checkpoint(checkpoint_path);
    }

    ~llama2() {
        // close the memory mapping
        if (data != MAP_FAILED) { munmap(data, file_size); }
        if (fd != -1) { close(fd); }
    }

    void read_checkpoint(char* checkpoint_path) {
        FILE *file = fopen(checkpoint_path, "rb");
        if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint_path); exit(EXIT_FAILURE); }
        // read in the config header
        if (fread(&config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
        // negative vocab size is hacky way of signaling unshared weights. bit yikes.
        int shared_weights = config.vocab_size > 0 ? 1 : 0;
        config.vocab_size = abs(config.vocab_size);
        // figure out the file size
        fseek(file, 0, SEEK_END); // move file pointer to end of file
        ssize_t file_size = ftell(file); // get the file size, in bytes
        fclose(file);

        // file backed mapping of Transformer weights to memory
        fd = open(checkpoint_path, O_RDONLY);
        if (fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
        data = (float *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if ((void *)data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
        float* weights = data + sizeof(Config)/sizeof(float);

        // memory map the Transformer weights into the data pointers
        multi_head_attention = new attention[config.n_layers];
        for (int i = 0 ; i < config.n_layers; i++) {
            multi_head_attention[i] = attention(config);
        }

        int head_size = config.dim / config.n_heads;

        token_embedding_table = embedding(weights, config.vocab_size, config.dim);
        weights += token_embedding_table.size();

        for (int i = 0 ; i < config.n_layers; i++) {
            auto sz = multi_head_attention[i].set_rms_att_weight(weights);
            weights += sz;
        }
        for (int i = 0 ; i < config.n_layers; i++) {
            auto sz = multi_head_attention[i].set_query(weights);
            weights += sz;
        }
        for (int i = 0 ; i < config.n_layers; i++) {
            auto sz = multi_head_attention[i].set_key(weights);
            weights += sz;
        }
        for (int i = 0 ; i < config.n_layers; i++) {
            auto sz = multi_head_attention[i].set_value(weights);
            weights += sz;
        }
        for (int i = 0 ; i < config.n_layers; i++) {
            auto sz = multi_head_attention[i].set_weight_o(weights);
            weights += sz;
        }
        for (int i = 0 ; i < config.n_layers; i++) {
            auto sz = multi_head_attention[i].set_rms_ffn_weight(weights);
            weights += sz;
        }
        for (int i = 0 ; i < config.n_layers; i++) {
            auto sz = multi_head_attention[i].set_ffn_weights1(weights);
            weights += sz;
        }
        for (int i = 0 ; i < config.n_layers; i++) {
            auto sz = multi_head_attention[i].set_ffn_weights2(weights);
            weights += sz;
        }
        for (int i = 0 ; i < config.n_layers; i++) {
            auto sz = multi_head_attention[i].set_ffn_weights3(weights);
            weights += sz;
        }

        rms_final_weight = tensor{weights, {1, config.dim}};
        weights += config.dim;

        weights += config.seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
        weights += config.seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
        
        wcls = shared_weights ? token_embedding_table : tensor{weights, {config.vocab_size, config.dim}};
    }

    tensor forward(int token, int pos) {
        tensor x = token_embedding_table(token);
        
        for (int l = 0 ; l < config.n_layers ; l++) {
            x = multi_head_attention[l].forward(x, pos);
        }

        x = rms_norm(x, rms_final_weight);
        tensor logits = x * wcls;
        return logits;
    }
};

int main() {
    return 0;
}