#ifndef __tinyinference_mathlib_h
#define __tinyinference_mathlib_h

#include "tensor.h"

tensor rms_norm(const tensor& x, const tensor& weight, const float eps = 1e-5f);
tensor softmax(const tensor& x);
tensor sigmoid(const tensor& x);
tensor silu(const tensor& x);

#endif