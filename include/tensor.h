#include <iostream>
#include <vector>

class tensor {
    std::vector<float> mat; //dim1 x dim2 matrix
    std::pair<int, int> dim;

    public:
        tensor();
        tensor(std::vector<float> mat, std::pair<int, int> dim);
        tensor(std::pair<int, int> dim, float val);
        tensor(std::pair<int, int> dim);

        tensor copy();
        unsigned int rows() const;
        unsigned int cols() const;
        std::pair<int, int> shape() const;
        std::string toString() const;
        std::vector<float> matrix() const;

        const float& operator [](int idx) const;
        tensor operator+(tensor const& obj) const;
        tensor operator*(tensor const& obj) const;
        tensor operator*(float const& obj) const;
};