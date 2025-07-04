#include <iostream>
#include <vector>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include<functional>
namespace py = pybind11;

template <typename T>
class storage {
    private:

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
    std::vector<size_t> shape;
    T *data;
    size_t size;
    storage(const storage& other)
  : shape(other.shape),
    size(other.size),
    data(new T[other.size])
    {
        std::copy(other.data, other.data + size, data);
    }

    storage():data(nullptr), size(0){}
    storage& operator=(const storage& other) {
        if (this != &other) {
            if(data != nullptr){
            delete[] data;}
            shape = other.shape;
            size  = other.size;
            data  = new T[size];
            std::copy(other.data, other.data + size, data);
        }
        return *this;
    }

        storage(const std::vector<size_t> &dim,T default_value) : shape(dim), data(nullptr), size(1) {
            for (const size_t &i:dim) {
                size *= i;
            }
            data = new T[size];
            for (size_t i = 0; i < size; i++) {
                data[i] = default_value;
            }
        }
        ~storage() {\
            if(data!=nullptr){
            delete[] data;}
        }
        std::vector<size_t> dimensions() {
            return shape;
        }
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

        void print() {
            size_t a = 0;
            print_data(0,a);
        }
};

template <typename T>
storage<T> operator +(storage<T> &a,storage<T> &b) {
    storage<T> result(a.dimensions(), 0);
    for (size_t i = 0; i < a.size; i++) {
        result.data[i] =  a.data[i] + b.data[i];
    }
    return result;
}

template <typename T>
storage<T> operator -(storage<T> &a,storage<T> &b) {
    storage<T> result(a.dimensions(), 0);
    for (size_t i = 0; i < a.size; i++) {
        result.data[i] =  a.data[i] - b.data[i];
    }
    return result;
}
template <typename T>
storage<T> operator *(storage<T> &a,storage<T> &b) {
    storage<T> result(a.dimensions(), 0);
    for (size_t i = 0; i < a.size; i++) {
        result.data[i] =  a.data[i] * b.data[i];
    }
    return result;
}

template <typename T>
storage<T> operator /(storage<T> &a,storage<T> &b) {
    storage<T> result(a.dimensions(), 0);
    for (size_t i = 0; i < a.size; i++) {
        result.data[i] =  a.data[i] / b.data[i];
    }
    return result;
}

template <typename T>
storage<T> operator ^(storage<T> &a, T power) {
    storage<T> result(a.dimensions(), 0);
    for (size_t i = 0; i < a.size; i++) {
        result.data[i] = pow(a.data[i],power);
    }
    return result;
}

template <typename T>
storage<T> sqrt(storage<T> &a) {
    storage<T> result(a.dimensions(), 0);
    for (size_t i = 0; i < a.size; i++) {
        result.data[i] = pow(a.data[i],0.5);
    }
    return result;
}

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

template <typename T>
T dot(storage<T> &a , storage<T> &b) {
    T result = 0;
    for (size_t i = 0; i < a.shape[0]; i++) {
        result += a.data[i] * b.data[i];
    }
    return result;
}

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

template <typename T>
storage<T> s_tan(storage<T> &a,size_t terms) {
    storage<T> s = s_sin(a,terms);
    storage<T> c = s_cos(a,terms);
    storage<T> result = s/c;
    return result;
}

template <typename T>
storage<T> s_sec(storage<T> &a,size_t terms) {
    storage<T> ones (a.shape,1.0f);
    storage<T> c = s_cos(a,terms);
    storage<T> result = ones/c;
    return result;
}

template <typename T>
storage<T> s_csc(storage<T> &a,size_t terms) {
    storage<T> ones (a.shape,1.0f);
    storage<T> s = s_sin(a,terms);
    storage<T> result = ones/s;
    return result;
}

template <typename T>
storage<T> s_cot(storage<T> &a,size_t terms) {
    storage<T> ones (a.shape,1.0f);
    storage<T> t = s_tan(a,terms);
    storage<T> result = ones/t;
    return result;
}

class Node{
    public:
    Node(){}
    storage<float> gradient;
    virtual void apply(storage<float> &grad){
        if(gradient.data==nullptr){
            gradient = grad;
        }
        else{
            gradient = grad+gradient;
        }
    }
};

class AddNode: public Node{
public:
    Node* a,*b;
    void apply(storage<float> &grad) override{
        if(gradient.data==nullptr){
            gradient = grad;
        }
        else{
            gradient = grad+gradient;
        }
        a->apply(grad);
        b->apply(grad);
    }
};

class SubNode: public Node{
public:
    Node* a,*b;
    void apply(storage<float> &grad) override{
        if(gradient.data==nullptr){
            gradient = grad;
        }
        else{
            gradient = grad+gradient;
        }
        a->apply(grad);
        storage<float> grad_b = grad*storage<float>(grad.shape,-1);
        b->apply(grad_b);
    }
};

class Tensor {
    private:
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
    public:
        storage<float> data;
        Node* Tensor_Node = new Node();

        void backward(){
            storage<float> grad(data.shape,1);
            Tensor_Node->apply(grad);
        }

        Tensor grad(){
            return Tensor(Tensor_Node->gradient);
        }

        Tensor(storage<float> &other):data(other){}

        Tensor(std::vector<size_t> &dim,float default_value):data(dim,default_value){}

        void assign (py::list &list) {
            float *a = new float[data.size];
            int index = 0;
            flatten(list,a,index);
            if(data.data != nullptr){
            delete[] data.data;}
            data.data = a;
            }

        float access(std::vector<size_t> &idx) {
            return data.access(idx);}

        void change_value(py::list &idx, float value) {
            std::vector<size_t> index;
            for (const auto &i:idx) {
                index.push_back(i.cast<size_t>());
            }
            data.change_value(index,value);
        }

        void print() {
            data.print();
        }
};

Tensor add(Tensor &a, Tensor &b){
    Tensor c(a.data.shape, 0);
    c.data = a.data+b.data;
    AddNode* c_Node = new AddNode();
    c_Node->a = a.Tensor_Node;
    c_Node->b = b.Tensor_Node;
    c.Tensor_Node = c_Node;
    return c;
}

Tensor sub(Tensor &a, Tensor &b){
    Tensor c(a.data.shape, 0);
    c.data = a.data-b.data;
    SubNode* c_Node = new SubNode();
    c_Node->a = a.Tensor_Node;
    c_Node->b = b.Tensor_Node;
    c.Tensor_Node = c_Node;
    return c;
}

Tensor division(Tensor &a, Tensor &b){
    Tensor c(a.data.shape, 0);
    c.data = a.data/b.data;
    return c;
}

Tensor mul(Tensor &a, Tensor &b){
    Tensor c(a.data.shape, 0);
    c.data = a.data*b.data;
    return c;
}

Tensor sin(Tensor &a){
    Tensor c(a.data.shape, 0);
    c.data = s_sin(a.data,10);
    return c;
}

Tensor cos(Tensor &a){
    Tensor c(a.data.shape, 0);
    c.data = s_cos(a.data,10);
    return c;
}

Tensor sec(Tensor &a){
    Tensor c(a.data.shape, 0);
    c.data = s_sec(a.data,10);
    return c;
}

Tensor csc(Tensor &a){
    Tensor c(a.data.shape, 0);
    c.data = s_csc(a.data,10);
    return c;
}

Tensor tan(Tensor &a){
    Tensor c(a.data.shape, 0);
    c.data = s_tan(a.data,10);
    return c;
}

Tensor cot(Tensor &a){
    Tensor c(a.data.shape, 0);
    c.data = s_cot(a.data,10);
    return c;
}

Tensor matmul(Tensor &a, Tensor &b){
    storage<float> c = matmul(a.data, b.data);
    Tensor d(c.shape, 0);
    d.data = c;
    return d;
}

PYBIND11_MODULE(Ganit, m) {
    pybind11::class_<Tensor>(m, "Tanitra")
        .def(pybind11::init<std::vector<size_t>, float>())
        .def("__getitem__", &Tensor::access)
        .def("__setitem__", &Tensor::change_value)
        .def("print", &Tensor::print)
        .def("assign",&Tensor::assign)
        .def("backward",py::overload_cast<>(&Tensor::backward))
        .def("grad",&Tensor::grad);

    m.def("__add__",&add)
    .def("__sub__", &sub)
    .def("__div__",&division)
    .def("__mul__",&mul)
    .def("sin",&sin)
    .def("cos",&cos)
    .def("tan",&tan)
    .def("sec",&sec)
    .def("csc",&csc)
    .def("cot", &cot)
    .def("__matmul__",&matmul);
}