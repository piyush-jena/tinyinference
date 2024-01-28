#include <vector>
#include <string>

class encoder {
    protected:
        int vocab_size;

    public:
        encoder(int vocab_size) : vocab_size{vocab_size} {};
        virtual std::vector<int> encode(std::string text, bool bos, bool eos) = 0;
        virtual std::string decode(int prev_token, int token) = 0;
        virtual void safe_printf(std::string text) = 0; //to be removed in the long run.
};