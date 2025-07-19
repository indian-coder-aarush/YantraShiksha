#pragma once

#include <iostream>
#include <vector>

// Utility functions
size_t offset(std::vector<size_t>& stride, std::vector<size_t>& index);
bool increment(std::vector<size_t>& index, std::vector<size_t>& shape);

// Main storage class
class storage {
private:
    void print_data(int depth, size_t &index);

public:
    std::vector<size_t> shape;
    std::shared_ptr<double[]> data;
    size_t size;
    std::vector<size_t> stride;

    storage();  // Default constructor
    storage(const std::vector<size_t> &dim, double default_value); // Param constructor
    storage(const storage& other);  // Copy constructor
    storage& operator=(const storage& other);  // Copy assignment
    ~storage();  // Destructor

    void set_stride(const std::vector<size_t> &shape);

    std::vector<size_t> dimensions();
    double access(const std::vector<size_t> &indices);
    void change_value(const std::vector<size_t> &indices, double &value);
    storage slice(std::vector<std::vector<size_t>>& slice);
    void print();
};

// Element-wise binary operations
storage operator+(storage& a, storage& b);
storage operator-(storage& a, storage& b);
storage operator*(storage& a, storage& b);
storage operator/(storage& a, storage& b);
storage operator^(storage &a, double power);

// Transpose
storage T_s(storage& a);
storage T_s(storage& a, std::vector<size_t>& order);

// Other operations
storage sqrt(storage &a);
storage s_sin(storage &a, size_t terms);
storage s_cos(storage &a, size_t terms);
storage s_matmul(storage &a, storage &b);
double dot(storage &a, storage &b);
