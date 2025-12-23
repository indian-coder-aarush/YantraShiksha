#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "storage.h"
#include "Autograd.h"
#include "Tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(Math, m) {
    pybind11::class_<Tensor>(m, "Tensor")
            .def(pybind11::init<py::list &>())
            .def(pybind11::init<py::array_t<double> &>())
            .def("__getitem__", &Tensor::access)
            .def("__setitem__", &change_value)
            .def("reshape", &reshape)
            .def("backward", py::overload_cast<>(&Tensor::backward))
            .def_property_readonly("grad", &Tensor::grad)
            .def("__add__", &add)
            .def("__sub__", &sub)
            .def("__truediv__", &division)
            .def("__mul__", &mul)
            .def("__matmul__", &matmul)
            .def("T", &T)
            .def("__str__", &Tensor::print)
            .def("zero_grad", &Tensor::zero_grad);


    m.def("sin", &sin)
            .def("cos", &cos)
            .def("tan", &tan)
            .def("sec", &sec)
            .def("csc", &csc)
            .def("cot", &cot)
            .def("convolution", &convolution)
            .def("log", &log)
            .def("relu", &relu);
}
