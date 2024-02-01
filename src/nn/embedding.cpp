#include "nn/embedding.h"

embedding::embedding () : tensor() {}
embedding::embedding (float* data, int vocab_size, int embd_dim) : tensor{data, {vocab_size, embd_dim}, true} {}
embedding::embedding (int vocab_size, int embd_dim) : tensor({vocab_size, embd_dim}) {}

int embedding::embedding_size() {
    return tensor::columns();
}

int embedding::vocab_size() {
    return tensor::rows();
}

tensor& embedding::operator() (size_t token) const {
    tensor* res = new tensor(m_data + token * dim.second, {1, dim.second}, true);
    return *res;
}