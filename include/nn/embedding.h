#ifndef __tinyinference_embedding_h
#define __tinyinference_embedding_h

#include "tensor.h"

class embedding : public tensor {
    public:
        embedding ();
        embedding (float* data, int vocab_size, int embd_dim);
        embedding (int vocab_size, int embd_dim);
        
        int embedding_size();
        int vocab_size();

        tensor operator() (size_t token) const;
};

#endif