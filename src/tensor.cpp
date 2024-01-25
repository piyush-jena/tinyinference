#include <iostream>
#include <cmath>
#include <cassert>
#include <sstream>
#include <iomanip>
#include <vector>
#include <tensor.h>

tensor::tensor(std::vector<float> mat, std::pair<int, int> dim) : mat(mat), dim(dim) {}

tensor::tensor(std::pair<int, int> dim) : dim{dim} {
    mat = std::vector<float>(dim.first * dim.second);
}

tensor::tensor(std::pair<int, int> dim, float val) : dim{dim} {
    mat = std::vector<float>(dim.first * dim.second, val);
}

tensor tensor::copy() {
    tensor copy{mat, dim};
    return copy;
}

std::vector<float> tensor::matrix() const {
    return mat;
}

std::pair<int,int> tensor::shape() const {
    return dim;
}

const float& tensor::operator [](int idx) const {
    return mat[idx];
}

tensor tensor::operator+(tensor const& obj) const {
    assert(this->shape() == obj.shape());
    tensor res{obj};
    for (int i = 0 ; i < dim.first * dim.second ; i++)
        res.mat[i] += this->mat[i];
    return res;
}

tensor tensor::operator*(tensor const& obj) const {
    if (this->shape() == obj.shape()) {
        //dot product
        tensor res{obj};
        for (int i = 0 ; i < dim.first * dim.second ; i++)
            res.mat[i] *= this->mat[i];
        return res;
    } else if (dim.second == obj.shape().first) {
        std::vector<float> temp(dim.first * obj.shape().second);
        for (int i = 0 ; i < dim.first ; i++) {
            for (int j = 0 ; j < obj.shape().second ; j++) {
                float val = 0.0f;
                for (int k = 0 ; k < dim.second ; k++) {
                    val += mat[i * dim.second + k] * obj[k * obj.shape().second + j];
                }

                temp[i * obj.shape().second + j] = val;
            }
        }

        tensor res{temp, {dim.first, obj.shape().second}};
        return res;
    } else {
        throw std::runtime_error("Matrix dimensions are not compatible.");
    }
}

tensor tensor::operator*(float const& obj) const {
    tensor res{mat, dim};
    for (int i = 0 ; i < dim.first * dim.second ; i++)
        res.mat[i] *= obj;
    return res;
}

std::string tensor::toString() const {
    std::stringstream ss;

    for(int i = 0; i < dim.first; i++) {
        for(int j = 0; j < dim.second; j++) {
            ss << std::setprecision(2) << mat[i*dim.second+j] << " ";
        }

        ss << std::endl;
    }

    return ss.str();
}