#include <tensor.h>

class linear {
    tensor w;
    tensor b;
    bool bias;

public:
    linear(tensor weight) : w{weight}, bias{false} {}
    linear(tensor weight, tensor bias) : w{weight}, b{bias}, bias{true} {}

    tensor forward(const tensor& x) {
        if (bias)
            return x * w + b;
        return x * w;
    }

    tensor operator() (const tensor& x) const {
        if (bias)
            return x * w + b;
        return x * w;
    }
};