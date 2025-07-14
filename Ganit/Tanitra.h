#pragma once

#include <vector>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "storage.h"
#include "Autograd.h"
#include "Tanitra.h"

class Node;

class Tensor {
private:
    void flatten(pybind11::list &list, double *a, int &index);
    void get_shape(pybind11::list &list, std::vector<size_t> &shape);

public:
    storage data;
    std::shared_ptr<Node> Tensor_Node;

    // Constructors
    Tensor(storage &other);
    Tensor(std::vector<size_t> &dim, double default_value);
    Tensor(pybind11::list &list);

    // Autograd
    void backward();
    Tensor grad();

    // Access and mutation
    double access(std::vector<size_t> &idx);
    void change_value(pybind11::list &idx, double value);

    // Print
    void print();
};

// Tensor operations
Tensor add(Tensor &a, Tensor &b);
Tensor sub(Tensor &a, Tensor &b);
Tensor mul(Tensor &a, Tensor &b);
Tensor division(Tensor &a, Tensor &b);
Tensor matmul(Tensor &a, Tensor &b);
Tensor reshape(Tensor &a, std::vector<size_t> &shape);

// Trigonometric functions
Tensor sin(Tensor &a);
Tensor cos(Tensor &a);
Tensor tan(Tensor &a);
Tensor sec(Tensor &a);
Tensor csc(Tensor &a);
Tensor cot(Tensor &a);