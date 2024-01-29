#include "nn/linear.h"
#include "tensor.h"

linear::linear(tensor weight) : w{weight}, bias{false} {}
linear::linear(tensor weight, tensor bias) : w{weight}, b{bias}, bias{true} {}

tensor linear::forward(const tensor& x) {
    if (bias)
        return x * w + b;
    return x * w;
}

tensor linear::operator() (const tensor& x) const {
    if (bias)
        return x * w + b;
    return x * w;
}