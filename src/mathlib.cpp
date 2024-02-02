#include <cassert>
#include <utility>
#include <cmath>

#include "mathlib.h"

tensor rms_norm(const tensor& x, const tensor& weight, const float eps) {
    // calculate sum of squares
    assert(x.shape() == weight.shape());

    float ss = eps;
    std::pair<int, int> dim = x.shape();
    
    for (int i = 0 ; i < dim.first ; i++) {
        for (int j = 0 ; j < dim.second ; j++) {
            ss += (x[{i,j}] * x[{i,j}]);
        }
    }

    ss /= (dim.first * dim.second);
    ss = 1.0f / sqrtf(ss);

    //normalize and scale
    tensor result = weight * (x * ss);
    return result;
}

tensor softmax(const tensor& x) {
    int r = x.rows();
    int c = x.columns();
    tensor res = x;

    for (int i = 0 ; i < r ; i++) {
        float max_val = x[{i, 0}];
        float sum = 0.0f;
        for (int j = 1 ; j < c ; j++) {
            max_val = std::max(max_val, x[{i, j}]);
        }

        for (int j = 0 ; j < c ; j++) {
            res[{i,j}] = expf(x[{i,j}] - max_val);
            sum += res[{i,j}];
        }

        for (int j = 0 ; j < c ; j++) {
            res[{i,j}] /= sum;
        }
    }

    return res;
}

tensor sigmoid(const tensor& t) {
    tensor x = t;
    for (int i = 0 ; i < x.shape().first; i++) {
        for (int j = 0 ; j < x.shape().second ; j++) {
            x[{i, j}] = 1.0f / (1.0f + expf(-x[{i, j}]));
        }
    }

    return x;
}

tensor silu(const tensor& x) {
    tensor res = x * sigmoid(x);
    return res;
}