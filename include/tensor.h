#ifndef __tinyinference_tensor_h
#define __tinyinference_tensor_h

#include <string>
#include <utility>

class tensor {
    protected:
        bool ref = true;
        float* m_data;
        std::pair<int, int> dim;

    public:
        tensor();
        tensor(float* data, std::pair<int, int> dim);
        tensor(float* data, std::pair<int, int> dim, bool ref);
        tensor(std::pair<int, int> dim, float val);
        tensor(std::pair<int, int> dim);
        tensor(const tensor& matrix); //copy
        tensor(tensor&& matrix);      //move
        ~tensor();

        void set_data(float* data, int size);
        void set_shape(std::pair<int,int> n_dim);
        void set_ref(bool ref);

        bool get_ref() const { return ref; }
        float* get_data() const { return m_data; }

        inline size_t size() const { return (dim.first * dim.second); }
		inline size_t rows() const { return dim.first; }
		inline size_t columns() const { return dim.second; }
        std::pair<int, int> shape() const { return dim; }
        
        std::string toString() const;
        tensor copy();

        //tensor& slice(size_t index, std::pair<int,int> dim);
        tensor& operator[] (size_t index) const;
        float& operator[] (std::pair<size_t, size_t> index) const;
        tensor& operator() (size_t index) const;
        float& operator() (std::pair<size_t, size_t> index) const;
        /*
        float& operator() (size_t index);
        const float& operator() (size_t index) const;
        tensor operator[] (size_t index) const;*/


        tensor operator+(const tensor& obj) const;
        tensor operator*(const tensor& obj) const;
        tensor operator*(const float& obj) const;

        tensor& operator=(const tensor& matrix); //copy
		tensor& operator=(tensor&& matrix);      //move
};

#endif