#include <mathlib.h>
#include <cassert>

tensor rms_norm(const tensor& x, const tensor& weight, const float eps) {
    // calculate sum of squares
    float ss = eps;
    std::pair<int, int> dim = x.shape();
    assert(dim == weight.shape());

    for (int i = 0 ; i < (dim.first * dim.second) ; i++) {
        ss += (x(i) * x(i));
    }

    ss /= (dim.first * dim.second);
    ss = 1.0f / sqrtf(ss);

    //normalize and scale
    tensor result = weight * (x * ss);
    return result;
}

tensor softmax(tensor& x) {
    int r = x.rows();
    int c = x.columns();

    assert(r == 1 || c == 1);

    float* temp = new float[r * c];
    float max_val = x(0);

    for (int i = 1; i < (r * c) ; i++) {
        max_val = std::max(max_val, x(i));
    }

    float sum = 0.0f;
    for (int i = 0 ; i < (r * c) ; i++) {
        temp[i] = expf(x(i) - max_val);
        sum += temp[i];
    }

    for (int i = 0 ; i < (r * c) ; i++) {
        temp[i] /= sum;
    }

    tensor result{temp, x.shape()};
    return result;
}

tensor sigmoid(tensor& x) {
    for (int i = 0 ; i < x.size() ; i++) {
        x(i) = 1.0f / (1.0f + expf(-x(i)));
    }
}

tensor silu(tensor& x) {
    return x * sigmoid(x);
}