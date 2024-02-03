#include <cassert>
#include <cstring>
#include <iomanip>
#include <sstream>

#include "tensor.h"

tensor::tensor() : ref{true}, m_data{nullptr} {
    dim = {0,0};
}

tensor::tensor(float* data, std::pair<int, int> dim)
: ref{true}, m_data{data}, dim{dim} {
    //this->ref = true;
}

tensor::tensor(float* data, std::pair<int, int> dim, bool ref) 
: ref{ref}, m_data{data}, dim{dim} {
    //this->ref = ref;
}

tensor::tensor(std::pair<int, int> dim, float val)
: ref{false}, m_data{nullptr}, dim{dim}  {
    m_data = new float[dim.first * dim.second];
    std::fill_n(m_data, (dim.first * dim.second), val);
    //this->ref = false;
}

tensor::tensor(std::pair<int, int> dim) 
: ref{false}, m_data{nullptr}, dim{dim} {
    m_data = new float[dim.first * dim.second];
}

tensor::tensor(const tensor& t)
: ref{false}, m_data{nullptr} {
    m_data = new float[t.size()];
    dim = t.shape();
    memcpy(m_data, t.m_data, t.size() * sizeof(float));
    //this->ref = false;
}

tensor::tensor(tensor&& matrix) {
    m_data = matrix.get_data();
    dim = matrix.shape();
    matrix.ref = true;
}

tensor::~tensor() {
    if (!ref && m_data != nullptr) {
        delete[] m_data;
    }
}

void tensor::set_data(float* data, int size) {
    assert(this->size() == size);
    m_data = data;
    ref = true;
}

void tensor::set_ref(bool ref) {
    this->ref = ref;
}

void tensor::set_shape(std::pair<int,int> n_dim) {
    assert((dim.first * dim.second) == (n_dim.first * n_dim.second));
    dim = n_dim;
}

std::string tensor::toString() const {
    std::stringstream ss;
    for(int i = 0; i < dim.first; i++) {
        for(int j = 0; j < dim.second; j++) {
            ss << std::setprecision(2) << m_data[i*dim.second+j] << " ";
        }

        ss << std::endl;
    }

    return ss.str();
}

float& tensor::operator[] (std::pair<size_t, size_t> index) const {
    assert((index.first < dim.first) && (index.second < dim.second));
    return m_data[index.first * dim.second + index.second];
}

tensor& tensor::operator[] (size_t index) const {
    assert(index < dim.first);
    tensor* res = new tensor(m_data + (index * dim.second), {1, dim.second}, true);
    return *res;
}

float& tensor::operator() (std::pair<size_t, size_t> index) const {
    assert((index.first < dim.first) && (index.second < dim.second));
    return m_data[index.first * dim.second + index.second];
}

tensor& tensor::operator() (size_t index) const {
    assert(index < dim.first);
    tensor* res = new tensor(m_data + (index * dim.second), {1, dim.second}, true);
    return *res;
}

tensor tensor::operator+(const tensor& obj) const {
    assert(this->shape() == obj.shape());
    tensor res = *this;
    for (int i = 0 ; i < dim.first ; i++) {
        for (int j = 0 ; j < dim.second ; j++) {
            res[{i,j}] += obj[{i,j}];
        }
    }

    return res;
}

tensor tensor::operator*(const tensor& obj) const {
    assert((this->shape() == obj.shape()) || (dim.second == obj.shape().second));
    if (this->shape() == obj.shape()) {
        //dot product
        tensor res = *this;
        for (int i = 0 ; i < dim.first ; i++) {
            for (int j = 0 ; j < dim.second ; j++) {
                res[{i,j}] *= obj[{i,j}];
            }
        }

        return res;
    } else if (dim.second == obj.shape().second) {
        tensor res{{dim.first, obj.shape().first}};
        for (int i = 0 ; i < dim.first ; i++) {
            for (int j = 0 ; j < obj.shape().first ; j++) {
                float val = 0.0f;
                for (int k = 0 ; k < dim.second ; k++) {
                    val += m_data[i*dim.second + k] * obj.m_data[j*dim.second + k];
                }

                res[{i, j}] = val;
            }
        }

        return res;
    } else {
        throw std::runtime_error("Matrix dimensions are not compatible.");
    }
}

tensor tensor::operator*(const float& val) const {
    tensor res = *this;
    for (int i = 0 ; i < dim.first ; i++) {
        for (int j = 0 ; j < dim.second ; j++) {
            res[{i,j}] *= val;
        }
    }

    return res;
}

tensor& tensor::operator=(const tensor& matrix) {
	if (this != &matrix) {
		if (!ref && m_data != nullptr) {
            delete[] m_data;
        }
		
		m_data = matrix.m_data;
	    dim = matrix.dim;
	}
	
    return *this;
}

tensor& tensor::operator=(tensor&& matrix) {
    if (this != &matrix) {
        if (ref && this->size() == matrix.size()) {
            memcpy(m_data, matrix.m_data, sizeof(float) * dim.first * dim.second);
            ref = false;
        } else {
            if (!ref && m_data != nullptr) {
                delete[] m_data;
            }
            
            m_data = matrix.m_data;
            matrix.m_data = nullptr;
            ref = matrix.ref;
        }

        dim = matrix.dim;
    }

    return *this;
}

tensor tensor::copy() {
    float* m_data_copy = new float[this->size()];
    memcpy(m_data_copy, m_data, this->size() * sizeof(float));

    tensor new_t{m_data_copy, this->dim, false};
    return new_t;
}