#ifndef __tinyinference_linear_h
#define __tinyinference_linear_h

#include "tensor.h"

class linear {
    tensor w;
    tensor b;
    bool bias;

public:
    linear(tensor weight);
    linear(tensor weight, tensor bias);

    tensor forward(const tensor& x);

    tensor operator() (const tensor& x) const;
};

#endif