#include "llama2.h"
#include "encoder/bpe.h"
#include <iostream>
#include <vector>
#include "sampler.h"
#include <ctime>

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

int main (int argc, char *argv[]) {
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    unsigned long long rng_seed = 0; // seed rng with time by default

    //char *path = argv[1];
    char *path = argv[1];
    llama2 model{path};
    Sampler sampler{model.config.vocab_size, temperature, topp, rng_seed};

    std::string prompt = "Once upon a time";
    int vocab_size = model.config.vocab_size;

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    bpe tokenizer("tokenizer.bin", vocab_size);
    std::vector<int> prompt_tokens = tokenizer.encode(prompt, 1, 0);
    num_prompt_tokens = prompt_tokens.size();

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {
        // forward the transformer to get logits for the next token
        tensor logits = model.forward(token, pos);
        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sampler.sample(logits);
        }
        
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }
        // print the token as string, decode it with the Tokenizer object
        std::string piece = tokenizer.decode(token, next);
        tokenizer.safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;
        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
    }

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }

    return 0;
}