#include <iostream>
#include <vector>

class tensor {
        bool ref;
    protected:
        float* m_data;
        std::pair<int, int> dim;

    public:
        tensor();
        tensor(float* data, std::pair<int, int> dim);
        tensor(std::pair<int, int> dim, float val);
        tensor(std::pair<int, int> dim);
        tensor(const tensor& matrix);
        ~tensor();

        inline size_t size() const { return (dim.first * dim.second); }
		inline size_t rows() const { return dim.first; }
		inline size_t columns() const { return dim.second; }
        std::pair<int, int> shape() const;
        void reshape(std::pair<int,int> n_dim);
        std::string toString() const;
        float* matrix() const;

        float& operator() (size_t index); //should return float* array
        const float& operator() (size_t index) const;
        tensor operator[] (size_t index) const;
        tensor operator+(const tensor& obj) const;
        tensor operator*(const tensor& obj) const;
        tensor operator*(const float& obj) const;
        tensor& operator=(const tensor& matrix); //copy assignment
		tensor& operator=(tensor&& matrix);      //move assignment
};