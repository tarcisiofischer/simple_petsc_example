#include <solver.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_PLUGIN(_simple_petsc_example) {
    py::module m("_simple_petsc_example");
    m.def("init_petsc", &init_petsc);
    m.def("finalize_petsc", &finalize_petsc);
    m.def("run_solver", &run_solver);
    return m.ptr();
}
