#include <tensor.h>
#include <cassert>

tensor rms_norm(const tensor& x, const tensor& weight, const float eps = 1e-5f);
tensor softmax(tensor& x);
tensor sigmoid(tensor& x);
tensor silu(tensor& x);