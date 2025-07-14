#include <iostream>
#include <vector>
#include <cmath>
#include"storage.h"

size_t offset(std::vector<size_t>& stride,std::vector<size_t>& index){
    size_t offset_index = 0;
    for(int i = 0;i<stride.size();i++){
        offset_index += stride[i]*index[i];
    }
    return offset_index;
}

bool increment(std::vector<size_t>& index,std::vector<size_t>& shape){
    for(int i = index.size()-1;i>=0;--i){
        if(index[i]+1<shape[i]){
            index[i]++;
            return true;
        }
        index[i] = 0;
    }
    return false;
}

void storage::print_data(int depth , size_t &index) {
        std::cout << '['<<' ';
        for (int i = 0; i < shape[depth]; ++i) {
            if (depth == shape.size() - 1) {
                std::cout << data[index++]<<' ';
            }
            else {
                print_data(depth + 1,index);
            }
        }
        std::cout <<']';
    }

storage::storage(const storage& other)
      : shape(other.shape), size(other.size), data(new double[other.size]) {
        std::copy(other.data, other.data + size, data);
        set_stride(shape);
    }

void storage::set_stride(const std::vector<size_t> &shape){
    stride.resize(shape.size());
    size_t prod = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        stride[i] = prod;
        prod *= shape[i];
    }
}

storage::storage():data(nullptr), size(0){}

storage& storage::operator=(const storage& other) {
        if (this != &other) {
            if(data != nullptr){
                delete[] data;
            }
            shape = other.shape;
            size  = other.size;
            data  = new double[size];
            std::copy(other.data, other.data + size, data);
        }
        set_stride(shape);
        return *this;
    }

storage::storage(const std::vector<size_t> &dim,double default_value) : shape(dim), data(nullptr), size(1) {
        for (const size_t &i:dim) {
            size *= i;
        }
        data = new double[size];
        for (size_t i = 0; i < size; i++) {
            data[i] = default_value;
        }
        set_stride(shape);
    }

storage::~storage() {
        if(data!=nullptr){
            delete[] data;
        }
    }

std::vector<size_t> storage::dimensions() {
        return shape;
    }

double storage::access(const std::vector<size_t> &indices) {
        size_t index = 0;
        for (size_t i = 0; i < indices.size(); i++) {
            index += stride[i]*indices[i];
        }
        return data[index];
    }

    void storage::change_value(const std::vector<size_t> &indices, double &value) {
        size_t index = 0;
        for (size_t i = 0; i < indices.size(); i++) {
            index += stride[i]*indices[i];
        }
        data[index] = value;
    }


    void storage::print() {
        size_t a = 0;
        print_data(0,a);
    }


// Element-wise addition
template<typename Op>
storage elementwise_op(storage &a,storage &b,Op op) {
    storage result(a.dimensions(), 0);
    std::vector<size_t> index(a.shape.size(),0);
    size_t a_offset,b_offset,result_offset;
    do{
        a_offset = offset(a.stride,index);
        b_offset = offset(b.stride,index);
        result_offset = offset(result.stride,index);
        result.data[result_offset] = op(a.data[a_offset],b.data[b_offset]);
    }while(increment(index,a.shape));
    return result;
}

storage operator+(storage& a, storage& b) {
    return elementwise_op(a, b, [](double x, double y) { return x + y; });
}

// Element-wise subtraction
storage operator-(storage& a, storage& b) {
    return elementwise_op(a, b, [](double x, double y) { return x - y; });
}


// Element-wise multiplication
storage operator*(storage& a, storage& b) {
    return elementwise_op(a, b, [](double x, double y) { return x * y; });
}


// Element-wise division
storage operator/(storage& a, storage& b) {
    return elementwise_op(a, b, [](double x, double y) { return x / y; });
}

storage T(storage& a){
    size_t change_variable = a.shape[0];
    a.shape[0] = a.shape[1];
    a.shape[1] = change_variable;
    change_variable = a.stride[0];
    a.stride[0] = a.stride[1];
    a.stride[1] = change_variable;
    return a;
}

storage T(storage& a, std::vector<size_t>& order){
    std::vector<size_t> new_shape(a.shape.size(),0);
    std::vector<size_t> new_strides(a.stride.size(),0);
    for(size_t i = 0; i<a.shape.size();i++){
        new_strides[i] = a.stride[order[i]];
        new_shape[i] = a.shape[order[i]];
    }
    a.shape = new_shape;
    a.stride = new_strides;
    return a;
}

// Raise each element to a power
storage operator ^(storage &a, double power) {
    storage result(a.dimensions(), 0);
    for (size_t i = 0; i < a.size; i++) {
        result.data[i] = pow(a.data[i],power);
    }
    return result;
}

// Element-wise square root
storage sqrt(storage &a) {
    storage result(a.dimensions(), 0);
    for (size_t i = 0; i < a.size; i++) {
        result.data[i] = pow(a.data[i],0.5);
    }
    return result;
}

// Matrix multiplication
storage s_matmul(storage &a , storage &b){
    storage result(a.dimensions(), 0);
    double result_i_j = 0;
    for (size_t i = 0; i < a.shape[0]; i++) {
        for (size_t j = 0; j < b.shape[1]; j++) {
            for (size_t k = 0; k < b.shape[0]; k++) {
                result_i_j += a.access({i,k}) * b.access({k,j});
            }
            result.change_value({i,j}, result_i_j);
            result_i_j = 0;
        }
    }
    return result;
}

// Dot product (1D tensors)
double dot(storage &a , storage &b) {
    double result = 0;
    for (size_t i = 0; i < a.shape[0]; i++) {
        result += a.data[i] * b.data[i];
    }
    return result;
}

// Element-wise sine using Taylor series
storage s_sin(storage &a,size_t terms) {
    storage return_variable = a;
    double result = 0;
    double fact = 1;
    size_t a_offset;
    std::vector<size_t> index(a.shape.size(),0);
    do {
        a_offset = offset(a.stride,index);
        for (int j = 0; j < terms ; j++) {
            fact = 1;
            for (int k = 1; k <= (2*j+1); k++) {
                fact *= k;
            }
            result += (pow((0-1),j)*pow((a.data[a_offset]),(2*j+1)))/fact;
        }
        return_variable.data[a_offset] = result;
        result = 0;
    }while(increment(index,a.shape));
    return return_variable;
}

// Element-wise cosine using Taylor series
storage s_cos(storage &a,size_t terms) {
    storage return_variable = a;
    double result = 0;
    float fact = 1;
    size_t a_offset;
    std::vector<size_t> index(a.shape.size(),0);
    do {
        a_offset = offset(a.stride,index);
        for (int j = 0; j < terms ; j++) {
            fact = 1;
            for (int k = 1; k <= (2*j); k++) {
                fact *= k;
            }
            result += (pow((0-1),j)*pow((a.data[a_offset]),(2*j)))/fact;
        }
        return_variable.data[a_offset] = result;
        result = 0;
    }while(increment(index,a.shape));
    return return_variable;
}
