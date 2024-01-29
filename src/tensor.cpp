#include <cassert>
#include <cstring>
#include <iomanip>
#include <sstream>

#include "tensor.h"

tensor::tensor(float* data, std::pair<int, int> dim) 
: ref(true), m_data(data), dim(dim) {
}

tensor::tensor(std::pair<int, int> dim, float val) 
: ref(false), m_data(new float[dim.first * dim.second]), dim{dim} {
    std::fill_n(m_data, (dim.first * dim.second), val);
}

tensor::tensor(std::pair<int, int> dim) 
: ref(false), m_data(new float[dim.first * dim.second]), dim{dim} {}

tensor::tensor(const tensor& t)
: ref(false), m_data(new float[t.size()]), dim(t.shape())
{
    memcpy(m_data, t.m_data, t.size() * sizeof(float));
}

tensor::~tensor() {
    if (!ref)
        delete[] m_data;
}

std::pair<int,int> tensor::shape() const {
    return dim;
}

void tensor::reshape(std::pair<int,int> n_dim) {
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

float& tensor::operator() (size_t index) {
    return m_data[index];
}

tensor tensor::slice(size_t index, std::pair<int,int> dim) {
    return tensor(m_data + index, dim);
}

const float& tensor::operator() (size_t index) const {
    return m_data[index];
}

tensor tensor::operator[] (size_t index) const {
    return tensor(m_data + (index * dim.second), {1, dim.second});
}

tensor tensor::operator+(const tensor& obj) const {
    assert(this->shape() == obj.shape());
    tensor res = obj;
    for (int i = 0 ; i < dim.first * dim.second ; i++)
        res(i) += m_data[i];
    return res;
}

tensor tensor::operator*(const tensor& obj) const {
    assert((this->shape() == obj.shape()) || (dim.second == obj.shape().first));
    if (this->shape() == obj.shape()) {
        //dot product
        tensor res = obj;
        for (int i = 0 ; i < dim.first * dim.second ; i++)
            res(i) *= m_data[i];
        return res;
    } else if (dim.second == obj.shape().first) {
        float* temp = new float[dim.first * obj.shape().second];
        for (int i = 0 ; i < dim.first ; i++) {
            for (int j = 0 ; j < obj.shape().second ; j++) {
                float val = 0.0f;
                for (int k = 0 ; k < dim.second ; k++) {
                    val += m_data[i * dim.second + k] * obj(k * obj.shape().second + j);
                }

                temp[i * obj.shape().second + j] = val;
            }
        }

        tensor res{temp, {dim.first, obj.shape().second}};
        delete[] temp;
        return res;
    } else {
        throw std::runtime_error("Matrix dimensions are not compatible.");
    }
}

tensor tensor::operator*(const float& obj) const {
    tensor res{m_data, dim};
    for (int i = 0 ; i < dim.first * dim.second ; i++)
        res(i) *= obj;
    return res;
}

tensor& tensor::operator=(const tensor& matrix) {
    float* data = new float[dim.first * dim.second];
    memcpy(data, matrix.m_data, sizeof(float) * dim.first * dim.second);
    delete[] m_data;
    m_data = data;
    dim = matrix.dim;
    return *this;
}

tensor& tensor::operator=(tensor&& matrix) {
    if (ref && this->size() == matrix.size()) {
	    memcpy(m_data, matrix.m_data, sizeof(float) * dim.first * dim.second);
	} else {
		delete[] m_data;
	    m_data = matrix.m_data;
        matrix.m_data = nullptr;
	}

    dim = matrix.dim;
    return *this;
}