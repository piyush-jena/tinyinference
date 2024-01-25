#include <tensor.h>
#include <cassert>

tensor rms_norm(tensor& x, tensor& weight, const float eps = 1e-5f);
tensor softmax(tensor& x);