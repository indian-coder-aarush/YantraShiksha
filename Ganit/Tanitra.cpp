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
std::vector<size_t> slice_to_triplet(py::slice& slice, size_t& dim_size){
    py::ssize_t start, stop, step, slice_length;
    slice.compute(dim_size, &start, &stop, &step, &slice_length);
    std::vector<size_t> a = {static_cast<size_t>(start),static_cast<size_t>(stop),static_cast<size_t>(step)};
    return a;
}

std::vector<size_t> int_to_triplet(py::object& index, size_t& dim_size){
    size_t start = py::cast<size_t>(index), stop = py::cast<size_t>(index), step = 1;
    std::vector<size_t> a = {start};
    return a;
}

std::vector<std::vector<size_t>> slice_to_vector(py::tuple& slice, std::vector<size_t>& a_shape){
    std::vector<std::vector<size_t>> slice_vector;
    for (size_t i = 0; i < slice.size(); i++){
        if(py::isinstance<py::slice>(slice[i])){
            slice_vector.push_back(slice_to_triplet(py::cast<py::slice>(slice[i]),a_shape[i]));
        }
        else{
            slice_vector.push_back(int_to_triplet(py::cast<py::object>(slice[i]),a_shape[i]));
        }
    }
    return slice_vector;
}

void Tensor::flatten(py::list &list, double* a,int &index){
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
    Tensor::Tensor():data(){}
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
    Tensor Tensor::access(py::object& slice) {
        std::vector<std::vector<size_t>> slice_vector;
        if(py::isinstance<py::slice>(slice)){
            slice_vector.push_back(slice_to_triplet(py::cast<py::slice>(slice),data.shape[0]));
        }
        else if(py::isinstance<py::tuple>(slice)){
            slice_vector = slice_to_vector(py::cast<py::tuple>(slice),data.shape);
        }
        else{
            slice_vector.push_back(int_to_triplet(slice,data.shape[0]));
        }

        Tensor return_tensor(data.slice(slice_vector));
    std::shared_ptr<GetItemNode> c_Node = std::make_shared<GetItemNode>();
    c_Node->tensor  = std::make_shared<Tensor>(return_tensor);
    c_Node->a = Tensor_Node;
    c_Node->slice = slice_vector;
    return_tensor.Tensor_Node = c_Node;
    return return_tensor;
    }

    void change_value(Tensor& a, py::object& slice, Tensor& replace) {
        std::vector<std::vector<size_t>> slice_vector;
        if(py::isinstance<py::slice>(slice)){
            slice_vector.push_back(slice_to_triplet(py::cast<py::slice>(slice),a.data.shape[0]));
        }
        else if(py::isinstance<py::tuple>(slice)){
            slice_vector = slice_to_vector(py::cast<py::tuple>(slice),a.data.shape);
        }
        else{
            slice_vector.push_back(int_to_triplet(slice,a.data.shape[0]));
        }
        a.data.setslice(slice_vector,replace.data);
        Tensor a2;
        a2.data = a.data;
        std::shared_ptr<SetItemNode> a2_node = std::make_shared<SetItemNode>();
        a2_node->tensor = std::make_shared<Tensor>(a2);
        a2_node->b = a.Tensor_Node;
        a2_node->a = replace.Tensor_Node;
        a2_node->slice = slice_vector;
        a2.Tensor_Node = a2_node;
        a = a2;
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

Tensor T(Tensor &a){
    storage c = T_s(a.data);
    Tensor b(c);
    std::shared_ptr<TNode> b_Node  = std::make_shared<TNode>();
    b_Node->tensor =  std::make_shared<Tensor>(b);
    b_Node->a = a.Tensor_Node;
    b.Tensor_Node = b_Node;
    return b;
}

Tensor convolution(Tensor &a, Tensor &b, size_t stride){
    storage c = convolution_s(a.data,b.data,stride);
    Tensor d(c);
    return d;
}