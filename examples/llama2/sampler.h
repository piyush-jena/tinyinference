#include <vector>
#include <algorithm>
#include <cassert>

#include "tensor.h"
#include "mathlib.h"

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

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

class Sampler {
    int vocab_size;
    std::vector<std::pair<float, int>> prob_index;  // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;

    public:
    Sampler(int vocab_size, float temperature, float topp, unsigned long long rng_seed)
    : vocab_size{vocab_size}, temperature{temperature}, topp{topp}, rng_state{rng_seed}
    {
        // buffer only used with nucleus sampling; may not need but it's ~small
        prob_index = std::vector<std::pair<float, int>>(vocab_size);
    }

    int sample_argmax(tensor probabilities) {
        assert((probabilities.size() == vocab_size) && (probabilities.rows() == 1));
        // return the index that has the highest probability
        int max_i = 0;
        float max_p = probabilities[{0, 0}];
        for (int i = 1; i < vocab_size; i++) {
            if (probabilities[{0, i}] > max_p) {
                max_i = i;
                max_p = probabilities[{0, i}];
            }
        }
        return max_i;
    }

    int sample_mult(tensor& probabilities, float coin) {
        // sample index from probabilities (they must sum to 1!)
        // coin is a random number in [0, 1), usually from random_f32()
        float cdf = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            cdf += probabilities[{0, i}];

            if (coin < cdf) {
                return i;
            }
        }

        return vocab_size - 1; // in case of rounding errors
    }

    int sample_topp(tensor& probabilities, float topp, float coin) {
        // top-p sampling (or "nucleus sampling") samples from the smallest set of
        // tokens that exceed probability topp. This way we never sample tokens that
        // have very low probabilities and are less likely to go "off the rails".
        // coin is a random number in [0, 1), usually from random_f32()

        int n0 = 0;
        // quicksort indices in descending order of probabilities
        // values smaller than (1 - topp) / (n - 1) cannot be part of the result
        // so for efficiency we crop these out as candidates before sorting
        const float cutoff = (1.0f - topp) / (vocab_size - 1);
        for (int i = 0; i < vocab_size; i++) {
            if (probabilities[{0, i}] >= cutoff) {
                prob_index[n0].second = i;
                prob_index[n0].first = probabilities[{0, i}];
                n0++;
            }
        }

        sort(prob_index.begin(), prob_index.begin() + n0, std::greater<std::pair<float, int>>());

        // truncate the list where cumulative probability exceeds topp
        float cumulative_prob = 0.0f;
        int last_idx = n0 - 1; // in case of rounding errors consider all elements
        for (int i = 0; i < n0; i++) {
            cumulative_prob += prob_index[i].first;
            if (cumulative_prob > topp) {
                last_idx = i;
                break; // we've exceeded topp by including last_idx
            }
        }

        // sample from the truncated list
        float r = coin * cumulative_prob;
        float cdf = 0.0f;
        for (int i = 0; i <= last_idx; i++) {
            cdf += prob_index[i].first;
            if (r < cdf) {
                return prob_index[i].second;
            }
        }

        return prob_index[last_idx].second; // in case of rounding errors
    }

    int sample(tensor& logits) {
        // sample the token given the logits and some hyperparameters
        int next;
        if (temperature == 0.0f) {
            // greedy argmax sampling: take the token with the highest probability
            next = sample_argmax(logits);
        } else {
            // apply the temperature to the logits
            logits = logits * (1.0f / temperature);
            // apply softmax to the logits to get the probabilities for next token
            logits = softmax(logits);
            
            // flip a (float) coin (this is our source of entropy for sampling)
            float coin = random_f32(&rng_state);
            // we sample from this distribution to get the next token
            if (topp <= 0 || topp >= 1) {
                // simply sample from the predicted probability distribution
                next = sample_mult(logits, coin);
            } else {
                // top-p (nucleus) sampling, clamping the least likely tokens to zero
                next = sample_topp(logits, topp, coin);
            }
        }

        return next;
    }
};
