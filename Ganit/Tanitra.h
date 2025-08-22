#pragma once


#include <vector>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "storage.h"
#include "Autograd.h"
#include "Tanitra.h"


namespace py = pybind11;


class Node;


class Tensor {
private:
   void flatten(pybind11::list &list, double* a, int &index);
   void get_shape(pybind11::list &list, std::vector<int> &shape);


public:
   storage data;
   std::shared_ptr<Node> Tensor_Node;


   // Constructors
   Tensor();
   Tensor(storage &other);
   Tensor(std::vector<int> &dim, double default_value);
   Tensor(pybind11::list &list);


   // Autograd
   void backward();
   Tensor grad();


   // Access and mutation
   Tensor access(py::object& slice);


   // Print
   void print();
};


// Tensor operations
Tensor add(Tensor &a, Tensor &b);
Tensor sub(Tensor &a, Tensor &b);
Tensor mul(Tensor &a, Tensor &b);
Tensor division(Tensor &a, Tensor &b);
Tensor matmul(Tensor &a, Tensor &b);
Tensor reshape(Tensor &a, std::vector<int> &shape);
Tensor T(Tensor &a);
Tensor convolution(Tensor &a, Tensor &b, int stride);
// Trigonometric functions
Tensor sin(Tensor &a);
Tensor cos(Tensor &a);
Tensor tan(Tensor &a);
Tensor sec(Tensor &a);
Tensor csc(Tensor &a);
Tensor cot(Tensor &a);
Tensor log(Tensor &a);
Tensor relu(Tensor &a);


void change_value(Tensor& a, py::object& slice, Tensor& replace);