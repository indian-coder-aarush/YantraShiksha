#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "storage.h"
#include "Autograd.h"
#include "Tanitra.h"

namespace py = pybind11;

PYBIND11_MODULE(Ganit, m) {
    pybind11::class_<Tensor>(m, "Tanitra")
        .def(pybind11::init<py::list&>())
        .def("__getitem__", &Tensor::access)
        .def("__setitem__", &change_value)
        .def("reshape",&reshape)
        .def("backward",py::overload_cast<>(&Tensor::backward))
        .def("grad",&Tensor::grad)
        .def("__add__",&add)
        .def("__sub__", &sub)
        .def("__truediv__",&division)
        .def("__mul__",&mul)
        .def("__matmul__",&matmul)
        .def("T",&T);


    m.def("sin",&sin)
    .def("cos",&cos)
    .def("tan",&tan)
    .def("sec",&sec)
    .def("csc",&csc)
    .def("cot", &cot)
    .def("print",&Tensor::print)
    .def("convolution",&convolution);
}