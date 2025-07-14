#include <iostream>
#include <vector>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "Tanitra.h"
#include "Autograd.h"
#include "storage.h"

namespace py = pybind11;

void Tensor::flatten(py::list &list, double *a,int &index){
        for(auto i: list){
            if(py::isinstance<py::list>(i)){
                flatten(i.cast<py::list>(),a,index);
            }
            else{
                a[index] = i.cast<double>();
                index++;
            }
        }
    }
    void Tensor::get_shape(py::list& list, std::vector<size_t> &shape){
        if(py::isinstance<py::list>(list[0])){
            shape.push_back(list.size());
            get_shape(list[0].cast<py::list>(),shape);
        }
        else{
            shape.push_back(list.size());
        }
    }

void Tensor::backward(){
        storage grad(data.shape,1);
        Tensor_Node->apply(grad);
    }

    // Get gradient Tensor
    Tensor Tensor::grad(){
        return Tensor(Tensor_Node->gradient);
    }

    // Constructors
    Tensor::Tensor(storage &other):data(other),Tensor_Node(std::make_shared<Node>()){
        Tensor_Node->tensor = std::make_shared<Tensor>(*this);
    }
    Tensor::Tensor(std::vector<size_t> &dim,double default_value):data(dim,default_value),Tensor_Node(std::make_shared<Node>()){
        Tensor_Node->tensor = std::make_shared<Tensor>(*this);
    }
    Tensor::Tensor(py::list& list):Tensor_Node(std::make_shared<Node>()){
        std::vector<size_t> shape;
        get_shape(list,shape);
        size_t size = 1;
        data.shape = shape;
        for(int i = 0; i<shape.size();i++){
            size*= shape[i];
        }
        data.size = size;
        double* a = new double[size];
        int index = 0;
        flatten(list,a,index);
        data.data = a;
        data.set_stride(data.shape);
        Tensor_Node->tensor = std::make_shared<Tensor>(*this);
    }

    // Access and modify elements
    double Tensor::access(std::vector<size_t> &idx) {
        return data.access(idx);}
    void Tensor::change_value(py::list &idx, double value) {
        std::vector<size_t> index;
        for (const auto &i:idx) {
            index.push_back(i.cast<size_t>());
        }
        data.change_value(index,value);
    }

    // Print tensor
    void Tensor::print() {
        data.print();
    }


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
    Tensor c(a.data.shape, 0);
    c.data = s_matmul(a.data,b.data);
    std::shared_ptr<MatmulNode> c_Node = std::make_shared<MatmulNode>();
    c_Node->tensor  = std::make_shared<Tensor>(c);
    c_Node->b = b.Tensor_Node;
    c_Node->a = a.Tensor_Node;
    c.Tensor_Node = c_Node;
    return c;
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