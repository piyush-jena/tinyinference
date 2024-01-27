#include <embedding.h>

embedding::embedding (float* data, int vocab_size, int embd_dim) : tensor(data, {vocab_size, embd_dim}) {}
embedding::embedding (int vocab_size, int embd_dim) : tensor({vocab_size, embd_dim}) {}

int embedding::embedding_size() {
    tensor::columns();
}

int embedding::vocab_size() {
    tensor::rows();
}

tensor embedding::operator() (size_t token) const {
    return tensor(m_data + token * dim.first, {1, dim.first});
}