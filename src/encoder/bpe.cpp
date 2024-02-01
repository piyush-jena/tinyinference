#include <iostream>
#include <fstream>
#include <algorithm>

#include "encoder/bpe.h"

bpe::bpe(std::string tokenizer_path, int vocab_size) : encoder(vocab_size) {
    vocab = std::vector<std::string>(vocab_size);
    int len;
    float score;
    char buf[max_token_length * 2 + 3];

    for (int i = 0 ; i < 256 ; i++) {
        byte_pieces[i*2] = (unsigned char)i;
        byte_pieces[i*2 + 1] = '\0';
    }

    
    std::fstream file{tokenizer_path};
    if (!file) { throw std::runtime_error("Unable to open the tokenizer file tokenizer.bin!"); }
    file.read((char *)&max_token_length, sizeof(int));
    for (int i = 0 ; i < vocab_size; i++) {
        file.read((char *)&score, sizeof(float));
        file.read((char *)&len, sizeof(int));     
        file.read(buf, len * sizeof(char));
        std::string temp{buf, static_cast<unsigned long>(len)};
        vocab_scores[temp] = score;
        vocab[i] = temp;
    }

    file.close();
    sort(vocab.begin(), vocab.end());
}

std::vector<int> bpe::encode(std::string text, bool bos, bool eos) {
    std::vector<int> tokens;
    int token_count = 0;

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    //char str_buffer[max_token_length*2 + 3];
    std::string str_buffer = "";
    size_t str_len = 0;

    // add optional BOS (=1) token, if desired
    if (bos) {
        tokens.push_back(1);
        token_count++;
    }

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        auto itr = lower_bound(vocab.begin(), vocab.end(), " ");
        tokens.push_back(itr - vocab.begin());
        token_count++;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (int i = 0 ; i < text.size() ; i++) {
        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((text[i] & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer.push_back(text[i]); // ++ is post-increment, incremented after this line

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((text[i+1] & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        if (vocab_scores.find(str_buffer) != vocab_scores.end()) {
            auto itr = lower_bound(vocab.begin(), vocab.end(), str_buffer);
            tokens.push_back(itr - vocab.begin());
            token_count++;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens.push_back((unsigned char)str_buffer[i] + 3);
                token_count++;
            }
        }

        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0 ; i < token_count-1 ; i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            str_buffer = vocab[tokens[i]] + vocab[tokens[i+1]];
            auto itr = vocab_scores.find(str_buffer);
            if (itr != vocab_scores.end() && itr->second > best_score) {
                best_score = itr->second;
                best_id = lower_bound(vocab.begin(), vocab.end(), str_buffer) - vocab.begin();
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (token_count-1); i++) {
            tokens[i] = tokens[i+1];
        }

        token_count--; // token length decreased
    }

    if (eos) {
        tokens.push_back(2);
        token_count++;
    }
    return tokens;
}

std::string bpe::decode(int prev_token, int token) {
    char* piece = (char *)malloc((vocab[token].size()) * sizeof(char));
    strcpy(piece, vocab[token].c_str());
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)byte_pieces + byte_val * 2;
    }

    std::string result{piece};
    return result;
}

void bpe::safe_printf(std::string text) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (text.size() > 0) {
        if (text[0] != '\0') {
            if (text[1] == '\0') {
                unsigned char byte_val = text[0];
                if (!(isprint(byte_val) || isspace(byte_val))) {
                    return; // bad byte, don't print it
                }
            }

            std::cout << text;
        }
    }
}