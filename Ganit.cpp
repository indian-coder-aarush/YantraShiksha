// Include headers for I/O, containers, math, and pybind11
#include <iostream>
#include <vector>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <functional>
namespace py = pybind11;
class Tensor;

// Template class for storing multi-dimensional array-like data
template <typename T>
class storage {
private:
    // Recursive function to print data in multi-dimensional format
    void print_data(int depth , size_t &index) {
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

public:
    std::vector<size_t> shape; // Shape of the tensor
    T *data;                   // Pointer to data
    size_t size;              // Total size

    // Copy constructor
    storage(const storage& other)
      : shape(other.shape), size(other.size), data(new T[other.size]) {
        std::copy(other.data, other.data + size, data);
    }

    // Default constructor
    storage():data(nullptr), size(0){}

    // Assignment operator
    storage& operator=(const storage& other) {
        if (this != &other) {
            if(data != nullptr){
                delete[] data;
            }
            shape = other.shape;
            size  = other.size;
            data  = new T[size];
            std::copy(other.data, other.data + size, data);
        }
        return *this;
    }

    // Constructor with shape and default value
    storage(const std::vector<size_t> &dim,T default_value) : shape(dim), data(nullptr), size(1) {
        for (const size_t &i:dim) {
            size *= i;
        }
        data = new T[size];
        for (size_t i = 0; i < size; i++) {
            data[i] = default_value;
        }
    }

    // Destructor
    ~storage() {
        if(data!=nullptr){
            delete[] data;
        }
    }

    // Returns dimensions
    std::vector<size_t> dimensions() {
        return shape;
    }

    // Access data at given indices
    T access(const std::vector<size_t> &indices) {
        size_t indice = 0;
        size_t iterator = 1;
        for (size_t i = 0; i < indices.size(); i++) {
            for (size_t j = i+1; j < indices.size() ; j++) {
                iterator *= shape[j];
            }
            indice += iterator*indices[i];
            iterator = 1;
        }
        return data[indice];
    }

    // Change value at given indices
    void change_value(const std::vector<size_t> &indices, T &value) {
        size_t indice = 0;
        size_t iterator = 1;
        for (size_t i = 0; i < indices.size(); i++) {
            for (size_t j = i+1; j < indices.size() ; j++) {
                iterator *= shape[j];
            }
            indice += iterator*indices[i];
            iterator = 1;
        }
        data[indice] = value;
    }

    // Pretty print the data
    void print() {
        size_t a = 0;
        print_data(0,a);
    }
};

// Element-wise addition
template <typename T>
storage<T> operator +(storage<T> &a,storage<T> &b) {
    storage<T> result(a.dimensions(), 0);
    for (size_t i = 0; i < a.size; i++) {
        result.data[i] =  a.data[i] + b.data[i];
    }
    return result;
}

// Element-wise subtraction
template <typename T>
storage<T> operator -(storage<T> &a,storage<T> &b) {
    storage<T> result(a.dimensions(), 0);
    for (size_t i = 0; i < a.size; i++) {
        result.data[i] =  a.data[i] - b.data[i];
    }
    return result;
}

// Element-wise multiplication
template <typename T>
storage<T> operator *(storage<T> &a,storage<T> &b) {
    storage<T> result(a.dimensions(), 0);
    for (size_t i = 0; i < a.size; i++) {
        result.data[i] =  a.data[i] * b.data[i];
    }
    return result;
}

// Element-wise division
template <typename T>
storage<T> operator /(storage<T> &a,storage<T> &b) {
    storage<T> result(a.dimensions(), 0);
    for (size_t i = 0; i < a.size; i++) {
        result.data[i] =  a.data[i] / b.data[i];
    }
    return result;
}

// Raise each element to a power
template <typename T>
storage<T> operator ^(storage<T> &a, float power) {
    storage<T> result(a.dimensions(), 0);
    for (size_t i = 0; i < a.size; i++) {
        result.data[i] = pow(a.data[i],power);
    }
    return result;
}

// Element-wise square root
template <typename T>
storage<T> sqrt(storage<T> &a) {
    storage<T> result(a.dimensions(), 0);
    for (size_t i = 0; i < a.size; i++) {
        result.data[i] = pow(a.data[i],0.5);
    }
    return result;
}

// Matrix multiplication
template <typename T>
storage<T> matmul(storage<T> &a , storage<T> &b){
    storage<T> result(a.dimensions(), 0);
    T result_i_j = 0;
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
template <typename T>
T dot(storage<T> &a , storage<T> &b) {
    T result = 0;
    for (size_t i = 0; i < a.shape[0]; i++) {
        result += a.data[i] * b.data[i];
    }
    return result;
}

// Element-wise sine using Taylor series
template <typename T>
storage<T> s_sin(storage<T> &a,size_t terms) {
    storage<T> return_variable = a;
    T result = 0;
    double fact = 1;
    for (size_t i = 0; i < a.size; i++) {
        for (int j = 0; j < terms ; j++) {
            fact = 1;
            for (int k = 1; k <= (2*j+1); k++) {
                fact *= k;
            }
            result += (pow((0-1),j)*pow((a.data[i]),(2*j+1)))/fact;
        }
        return_variable.data[i] = result;
        result = 0;
    }
    return return_variable;
}

// Element-wise cosine using Taylor series
template <typename T>
storage<T> s_cos(storage<T> &a,size_t terms) {
    storage<T> return_variable = a;
    T result = 0;
    float fact = 1;
    for (size_t i = 0; i < a.size; i++) {
        for (int j = 0; j < terms ; j++) {
            fact = 1;
            for (int k = 1; k <= (2*j); k++) {
                fact *= k;
            }
            result += (pow((0-1),j)*pow((a.data[i]),(2*j)))/fact;
        }
        return_variable.data[i] = result;
        result = 0;
    }
    return return_variable;
}

// Base class for autodiff nodes
class Node {
public:
    Node(){}
    std::shared_ptr<Tensor> tensor;
    storage<float> gradient;
    void accumulate_gradient(storage<float>& grad) {
    if (gradient.data == nullptr)
        gradient = grad;
    else
        gradient = grad + gradient;
    }
    virtual void apply(storage<float> &grad){
        if(gradient.data==nullptr){
            gradient = grad;
        }
        else{
            gradient = grad+gradient;
        }
    }
};

// Wrapper class for Tensor and autodiff
class Tensor {
private:
    // Helper to flatten nested Python list to flat array
    void flatten(py::list &list, float *a,int &index){
        for(auto i: list){
            if(py::isinstance<py::list>(i)){
                flatten(i.cast<py::list>(),a,index);
            }
            else{
                a[index] = i.cast<float>();
                index++;
            }
        }
    }
    void get_shape(py::list& list, std::vector<size_t> &shape){
        if(py::isinstance<py::list>(list[0])){
            shape.push_back(list.size());
            get_shape(list[0].cast<py::list>(),shape);
        }
        else{
            shape.push_back(list.size());
        }
    }

public:
    storage<float> data;
    std::shared_ptr<Node> Tensor_Node;

    // Backpropagation entry point
    void backward(){
        storage<float> grad(data.shape,1);
        Tensor_Node->apply(grad);
    }

    // Get gradient Tensor
    Tensor grad(){
        return Tensor(Tensor_Node->gradient);
    }

    // Constructors
    Tensor(storage<float> &other):data(other),Tensor_Node(std::make_shared<Node>()){
        Tensor_Node->tensor = std::make_shared<Tensor>(*this);
    }
    Tensor(std::vector<size_t> &dim,float default_value):data(dim,default_value),Tensor_Node(std::make_shared<Node>()){
        Tensor_Node->tensor = std::make_shared<Tensor>(*this);
    }
    Tensor(py::list& list):Tensor_Node(std::make_shared<Node>()){
        std::vector<size_t> shape;
        get_shape(list,shape);
        size_t size = 1;
        data.shape = shape;
        for(int i = 0; i<shape.size();i++){
            size*= shape[i];
        }
        data.size = size;
        float* a = new float[size];
        int index = 0;
        flatten(list,a,index);
        data.data = a;
        Tensor_Node->tensor = std::make_shared<Tensor>(*this);
    }

    // Access and modify elements
    float access(std::vector<size_t> &idx) {
        return data.access(idx);}
    void change_value(py::list &idx, float value) {
        std::vector<size_t> index;
        for (const auto &i:idx) {
            index.push_back(i.cast<size_t>());
        }
        data.change_value(index,value);
    }

    // Print tensor
    void print() {
        data.print();
    }
};

// Addition node for autodiff
class AddNode: public Node {
public:
    std::shared_ptr<Node> a,b;
    void apply(storage<float> &grad) override {
        accumulate_gradient(grad);
        a->apply(grad);
        b->apply(grad);
    }
};

// Subtraction node for autodiff
class SubNode: public Node {
public:
    std::shared_ptr<Node> a,b;
    void apply(storage<float> &grad) override {
        accumulate_gradient(grad);
        a->apply(grad);
        storage<float> minus_ones(grad.shape,-1);
        storage<float> grad_b = grad * minus_ones;
        b->apply(grad_b);
    }
};

class MulNode: public Node {
    public:
    std::shared_ptr<Node> a,b;
    void apply(storage<float> &grad) override {
        accumulate_gradient(grad);
        storage<float> grad_a = b->tensor->data * grad;
        storage<float> grad_b = a->tensor->data * grad;
        a->apply(grad_a);
        b->apply(grad_b);
    }
};

class DivNode : public Node {
    public:
    std::shared_ptr<Node> a,b;
    void apply(storage<float> &grad) override {
        accumulate_gradient(grad);
        storage<float> minus_ones(grad.shape, -1);
        storage<float> grad_a = grad / b->tensor->data;
        storage<float> grad_b =  (grad * minus_ones * a->tensor->data)/(b->tensor->data^2);
        a->apply(grad_a);
        b->apply(grad_b);
    }
};

class SinNode : public Node {
    public:
    std::shared_ptr<Node> a;
    void apply(storage<float> &grad) override {
        accumulate_gradient(grad);
        storage<float> grad_a = s_cos(a->tensor->data,10) * grad;
        a->apply(grad_a);
    }
};

class CosNode : public Node {
    public:
    std::shared_ptr<Node> a;
    void apply(storage<float> &grad) override {
        accumulate_gradient(grad);
        storage<float> minus_ones(grad.shape, -1);
        storage<float> grad_a = s_sin(a->tensor->data,10) * grad * minus_ones;
        a->apply(grad_a);
    }
};

class ReshapeNode : public Node{
    public:
    std::shared_ptr<Node> a;
    std::vector<size_t> initial_shape;
    void apply(storage<float> &grad){
        accumulate_gradient(grad);
        grad.shape = initial_shape;
        a->apply(grad);
    }
};

// Overloaded tensor operations with autodiff
Tensor add(Tensor &a, Tensor &b){
    Tensor c(a.data.shape, 0);
    c.data = a.data+b.data;
    std::shared_ptr<AddNode> c_Node = std::make_shared<AddNode>();
    c_Node->tensor  = std::make_shared<Tensor>(c);
    c_Node->b = b.Tensor_Node;
    c_Node->a = a.Tensor_Node;
    c.Tensor_Node = c_Node;
    return c;
}

Tensor sub(Tensor &a, Tensor &b){
    Tensor c(a.data.shape, 0);
    c.data = a.data-b.data;
    std::shared_ptr<SubNode> c_Node  = std::make_shared<SubNode>();
    c_Node->tensor  = std::make_shared<Tensor>(c);
    c_Node->b = b.Tensor_Node;
    c_Node->a = a.Tensor_Node;
    c.Tensor_Node = c_Node;
    return c;
}

Tensor division(Tensor &a, Tensor &b){
    Tensor c(a.data.shape, 0);
    c.data = a.data/b.data;
    std::shared_ptr<DivNode> c_Node  = std::make_shared<DivNode>();
    c_Node->tensor  = std::make_shared<Tensor>(c);
    c_Node->b = b.Tensor_Node;
    c_Node->a = a.Tensor_Node;
    c.Tensor_Node = c_Node;
    return c;
}

Tensor mul(Tensor &a, Tensor &b){
    Tensor c(a.data.shape, 0);
    c.data = a.data*b.data;
    std::shared_ptr<MulNode> c_Node  = std::make_shared<MulNode>();
    c_Node->tensor  = std::make_shared<Tensor>(c);
    c_Node->b = b.Tensor_Node;
    c_Node->a = a.Tensor_Node;
    c.Tensor_Node = c_Node;
    return c;
}

Tensor sin(Tensor &a){
    Tensor c(a.data.shape, 0);
    c.data = s_sin(a.data,10);
    std::shared_ptr<SinNode> c_Node  = std::make_shared<SinNode>();
    c_Node->tensor =  std::make_shared<Tensor>(c);
    c_Node->a = a.Tensor_Node;
    c.Tensor_Node = c_Node;
    return c;
}

Tensor cos(Tensor &a){
    Tensor c(a.data.shape, 0);
    c.data = s_cos(a.data,10);
    std::shared_ptr<CosNode> c_Node  = std::make_shared<CosNode>();
    c_Node->tensor =  std::make_shared<Tensor>(c);
    c_Node->a = a.Tensor_Node;
    c.Tensor_Node = c_Node;
    return c;
}

Tensor sec(Tensor &a){
    Tensor b(a.data.shape, 1);
    Tensor c = division(b,cos(a));
    return c;
}

Tensor csc(Tensor &a){
    Tensor b(a.data.shape, 1);
    Tensor c = division(b,sin(a)).data;
    return c;
}

Tensor tan(Tensor &a){
    Tensor c = division(sin(a),cos(a));
    return c;
}

Tensor cot(Tensor &a){
    Tensor c = division(cos(a),sin(a));
    return c;
}

Tensor matmul(Tensor &a, Tensor &b){
    storage<float> c = matmul(a.data, b.data);
    Tensor d(c.shape, 0);
    d.data = c;
    return d;
}

Tensor reshape(Tensor &a , std::vector<size_t> &shape){
    Tensor b(a.data);
    std::shared_ptr<ReshapeNode> b_Node  = std::make_shared<ReshapeNode>();
    b_Node->tensor =  std::make_shared<Tensor>(b);
    b_Node->a = a.Tensor_Node;
    b_Node->initial_shape = a.data.shape;
    b.data.shape = shape;
    b.Tensor_Node = b_Node;
    return b;
}

// Bind everything with pybind11
PYBIND11_MODULE(Ganit, m) {
    pybind11::class_<Tensor>(m, "Tanitra")
        .def(pybind11::init<py::list&>())
        .def("__getitem__", &Tensor::access)
        .def("__setitem__", &Tensor::change_value)
        .def("print", &Tensor::print)
        .def("reshape",&reshape)
        .def("backward",py::overload_cast<>(&Tensor::backward))
        .def("grad",&Tensor::grad)
        .def("__add__",&add)
        .def("__sub__", &sub)
        .def("__truediv__",&division)
        .def("__mul__",&mul);


    m.def("sin",&sin)
    .def("cos",&cos)
    .def("tan",&tan)
    .def("sec",&sec)
    .def("csc",&csc)
    .def("cot", &cot)
    .def("__matmul__",&matmul);
}