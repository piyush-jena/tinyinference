#ifndef __tinyinference_bpe_h
#define __tinyinference_bpe_h

#include <map>
#include <utility>
#include "encoder.h"

class bpe : public encoder {
    std::vector<std::string> vocab;
    std::map<std::string, std::pair<int, float>> vocab_scores;

    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings

    public:
        bpe (std::string tokenizer_path, int vocab_size);
        std::vector<int> encode(std::string text, bool bos, bool eos);
        std::string decode(int prev_token, int token);
        void safe_printf(std::string text);
};

#endif