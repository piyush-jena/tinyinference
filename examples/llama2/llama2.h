#include "mathlib.h"
#include "tensor.h"

#include "nn/embedding.h"
#include "attention.h"

#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <ctime>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <iostream>

class llama2 {
    embedding token_embedding_table;
    attention* multi_head_attention;
    tensor wcls;
    tensor rms_final_weight;

    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
public:
    Config config; // the hyperparameters of the architecture (the blueprint)
    llama2() {};

    llama2(char* checkpoint_path) {
        read_checkpoint(checkpoint_path);
    }

    ~llama2() {
        delete[] multi_head_attention;
        // close the memory mapping
        if ((void *)data != MAP_FAILED) { munmap(data, file_size); }
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
        file_size = ftell(file); // get the file size, in bytes
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
        weights = nullptr;
    }

    tensor forward(int token, int pos) {
        tensor emb = token_embedding_table(token);
        tensor x = emb.copy();

        for (int l = 0 ; l < config.n_layers ; l++) {
            x = multi_head_attention[l].forward(x, pos);
        }

        x = rms_norm(x, rms_final_weight);
        return (x * wcls);
    }
};