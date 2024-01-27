#include <tensor.h>

class embedding : public tensor {
    public:
        embedding (float* data, int vocab_size, int embd_dim);
        embedding (int vocab_size, int embd_dim);
        
        int embedding_size();
        int vocab_size();

        tensor operator() (size_t token) const;
};