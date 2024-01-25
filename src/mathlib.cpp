#include <mathlib.h>
#include <cassert>

tensor rms_norm(tensor& x, tensor& weight, const float eps) {
    // calculate sum of squares
    float ss = eps;
    std::pair<int, int> dim = x.shape();
    assert(dim == weight.shape());

    for (int i = 0 ; i < (dim.first * dim.second) ; i++) {
        ss += (x[i] * x[i]);
    }

    ss /= (dim.first * dim.second);
    ss = 1.0f / sqrtf(ss);

    //normalize and scale
    tensor result = weight * (x * ss);
    return result;
}

tensor softmax(tensor& x) {
    std::pair<int, int> dim = x.shape();
    assert(dim.first == 1 || dim.second == 1);

    std::vector<float> temp(dim.first * dim.second);
    float max_val = x[0];

    for (int i = 1; i < (dim.first * dim.second) ; i++) {
        max_val = std::max(max_val, x[i]);
    }

    float sum = 0.0f;
    for (int i = 0 ; i < (dim.first * dim.second) ; i++) {
        temp[i] = expf(x[i] - max_val);
        sum += temp[i];
    }

    for (int i = 0 ; i < (dim.first * dim.second) ; i++) {
        temp[i] /= sum;
    }

    tensor result{temp, dim};
    return result;
}