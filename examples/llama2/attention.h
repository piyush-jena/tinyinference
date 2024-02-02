#include "config.h"
#include "tensor.h"
#include <iostream>

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
        attention (Config config) : config{config} {
            int head_size = config.dim / config.n_heads;
            key_cache = tensor{{config.seq_len, config.n_kv_heads * head_size}};
            value_cache = tensor{{config.seq_len, config.n_kv_heads * head_size}};
        }

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
            weight1 = tensor{w1, {config.hidden_dim, config.dim}};
            return weight1.size();
        }

        ssize_t set_ffn_weights2(float* w2) {
            weight2 = tensor{w2, {config.dim, config.hidden_dim}};
            return weight2.size();
        }

        ssize_t set_ffn_weights3(float* w3) {
            weight3 = tensor{w3, {config.hidden_dim, config.dim}};
            return weight3.size();
        }

        ssize_t set_rms_ffn_weight(float* w) {
            rms_ffn_weight = tensor{w, {1, config.dim}};
            return rms_ffn_weight.size();
        }

        tensor forward(tensor x, int pos) {
            int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
            int head_size = config.dim / config.n_heads;
            int kv_mul = config.n_heads / config.n_kv_heads;

            tensor xb = rms_norm(x, rms_att_weight);

            tensor q = xb * query;
            tensor k = xb * key;
            tensor v = xb * value;

            for (int i = 0 ; i < k.size() ; i++) {
                key_cache[{pos, i}] = k[{0,i}];
                value_cache[{pos, i}] = v[{0,i}];
            }

            q.set_shape({config.n_heads, head_size});
            k.set_shape({config.n_kv_heads, head_size});
            v.set_shape({config.n_kv_heads, head_size});
            
            // RoPE relative positional encoding: complex-valued rotate q and k in each head
            for (int i = 0 ; i < config.dim ; i += 2) {
                int head_dim = i % head_size;
                int head_no = i / head_size;

                float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
                float val = pos * freq;
                float fcr = cosf(val);
                float fci = sinf(val);
                int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only

                for (int v = 0 ; v < rotn ; v++) {
                    if (v == 0) {
                        float v0 = q[{head_no, head_dim}];
                        float v1 = q[{head_no, head_dim+1}];

                        q[{head_no, head_dim}] = v0 * fcr - v1 * fci;
                        q[{head_no, head_dim+1}] = v0 * fci + v1 * fcr;
                    } else {
                        float v0 = k[{head_no, head_dim}];
                        float v1 = k[{head_no, head_dim+1}];

                        k[{head_no, head_dim}] = v0 * fcr - v1 * fci;
                        k[{head_no, head_dim+1}] = v0 * fci + v1 * fcr;
                    }
                }
            }
            
            tensor att{{1, pos+1}};
            for (int h = 0 ; h < config.n_heads ; h++) {
                tensor _q = q[h];
                
                for (int t = 0 ; t <= pos ; t++) {
                    tensor _k = key_cache[t + h/config.n_heads];

                    float score = 0.0f;
                    for (int i = 0 ; i < head_size ; i++) {
                        score += _q[{0,i}] * _k[{0,i}];
                    }

                    score /= sqrtf(head_size);
                    // save the score to the attention buffer
                    att[{0, t}] = score;
                }

                att = softmax(att);

                int idx = h * head_size;
                for (int t = 0 ; t <= pos ; t++) {
                    tensor _v = value_cache[t];
                    float a = att[{0, t}];

                    for (int i = 0 ; i < head_size ; i++) {
                        xb[{0, idx + i}] = a * _v[{0, (idx / kv_mul) + i}];
                    }
                }
            }
            
            //some calculation went wrong above here
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
