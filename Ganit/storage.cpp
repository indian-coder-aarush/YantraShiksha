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

void storage::print_data(int depth , std::vector<size_t> &index) {
    std::cout << '[' << ' ';
    for (size_t i = 0; i < shape[depth]; ++i) {
        index[depth] = i;
        if (depth == shape.size() - 1) {
            std::cout << this->access(index) << ' ';
        } else {
            print_data(depth + 1, index);
        }
    }
    std::cout << ']';
}

storage::storage(const storage& other)
      : shape(other.shape), size(other.size), data(other.data) ,
      stride(other.stride){
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
            shape = other.shape;
            size  = other.size;
            data  = other.data;
        }
        set_stride(shape);
        return *this;
    }

storage::storage(const std::vector<size_t> &dim,double default_value) : shape(dim), data(nullptr), size(1) {
        for (const size_t &i:dim) {
            size *= i;
        }
        double* a = new double[size];
        data = a;
        for (size_t i = 0; i < size; i++) {
            *(data+i) = default_value;
        }
        set_stride(shape);
    }

storage::~storage() {
    }

std::vector<size_t> storage::dimensions() {
        return shape;
    }

double storage::access(const std::vector<size_t> &indices) {
        size_t index = 0;
        for (size_t i = 0; i < indices.size(); i++) {
            index += stride[i]*indices[i];
        }
        return *(data+index);
    }

    void storage::change_value(const std::vector<size_t> &indices, double &value) {
        size_t index = 0;
        for (size_t i = 0; i < indices.size(); i++) {
            index += stride[i]*indices[i];
        }
        *(data+index) = value;
    }


    void storage::print() {
        std::vector<size_t> a(shape.size(),0);
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
        *(result.data+result_offset) = op(*(a.data+a_offset),*(b.data+b_offset));
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

storage T_s(storage& a){
    storage b(a);
    b.shape[0] = a.shape[1];
    b.shape[1] = a.shape[0];
    b.stride[0] = a.stride[1];
    b.stride[1] = a.stride[0];
    return b;
}

storage T_s(storage& a, std::vector<size_t>& order){
    std::vector<size_t> new_shape(a.shape.size(),0);
    std::vector<size_t> new_strides(a.stride.size(),0);
    for(size_t i = 0; i<a.shape.size();i++){
        new_strides[i] = a.stride[order[i]];
        new_shape[i] = a.shape[order[i]];
    }
    storage b(a);
    b.shape = new_shape;
    b.stride = new_strides;
    return b;
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
        *(result.data+i) = pow(*(a.data+i),0.5);
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
            result += (pow((0-1),j)*pow((*(a.data+a_offset)),(2*j+1)))/fact;
        }
        *(return_variable.data+a_offset) = result;
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
            result += (pow((0-1),j)*pow((*(a.data+a_offset)),(2*j)))/fact;
        }
        *(return_variable.data+a_offset) = result;
        result = 0;
    }while(increment(index,a.shape));
    return return_variable;
}

storage storage::slice(std::vector<std::vector<size_t>>& slice){
    std::vector<size_t> new_shape;
    std::vector<size_t> new_stride;
    size_t index_offset = 0;
    for(int i=0;i<slice.size();i++){
        if(slice[i].size() != 1){
        new_shape.push_back(static_cast<size_t>((slice[i][1]-slice[i][0]+slice[i][2]-1)/slice[i][2]));
        new_stride.push_back(stride[i]*slice[i][2]);}
        index_offset += stride[i]*slice[i][0];
    }
    if(new_shape.size()!=shape.size()){
        for (size_t i = slice.size(); i < shape.size(); i++) {
        new_shape.push_back(shape[i]);
        new_stride.push_back(stride[i]);
    }
    }
    storage b;
    b.stride = new_stride;
    b.shape = new_shape;
    b.data = &data[index_offset];
    return b;
}

void storage::setslice(std::vector<std::vector<size_t>>& slice,storage& other){
    size_t index_offset = 0;
    std::vector<size_t> new_stride;
    for(int i=0;i<slice.size();i++){
        if(slice[i].size() != 1){
        new_stride.push_back(stride[i]*slice[i][2]);}
        index_offset += stride[i]*slice[i][0];
    }
    if(new_stride.size()!=stride.size()){
        for (size_t i = slice.size(); i < shape.size(); i++) {
        new_stride.push_back(stride[i]);
    }
    }
    size_t offset_assign;
    std::vector<size_t> index(other.shape.size(), 0);
    double* data_to_be_assigned = &data[index_offset];
    do{
        offset_assign = offset(new_stride, index);
        data[offset_assign] = other.access(index);
    }while(increment(index,other.shape));
}