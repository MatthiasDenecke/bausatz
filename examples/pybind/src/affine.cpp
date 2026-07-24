// A tiny C++ compute kernel exposed to Python with pybind11.
//
// It stands in for "the C++ class you already have and want to drive from Python": a plain object
// constructed from a few scalars, with a scalar method and a batched one. The pybind11 module at
// the bottom is the entire bridge — no hand-written CPython boilerplate, no manual refcounting.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>   // <-- automatic std::vector <-> list, std::optional <-> None, etc.

#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

// The "engine": y = scale * x + bias. Pure C++, knows nothing about Python or pydantic.
class AffineKernel {
public:
    AffineKernel(double scale, double bias) : scale_(scale), bias_(bias) {
        if (scale == 0.0) {
            // Thrown from C++, surfaced in Python as ValueError (see the translation note below).
            throw std::invalid_argument("AffineKernel: scale must be non-zero");
        }
    }

    double apply(double x) const { return scale_ * x + bias_; }

    // Batched path: takes a Python list (converted to std::vector by the stl.h caster), loops in
    // pure C++, returns a vector (converted back to a list). See the GIL note on the binding.
    std::vector<double> apply_many(const std::vector<double> &xs) const {
        std::vector<double> out;
        out.reserve(xs.size());
        for (double x : xs) out.push_back(scale_ * x + bias_);
        return out;
    }

    double scale() const { return scale_; }
    double bias() const { return bias_; }

private:
    double scale_;
    double bias_;
};

PYBIND11_MODULE(_affine, m) {
    m.doc() = "pybind11 bridge for AffineKernel — the native engine behind the Affine component";

    py::class_<AffineKernel>(m, "AffineKernel")
        // Constructor: Python AffineKernel(scale=..., bias=...) -> C++ ctor. Keyword args + a
        // default make it call cleanly from the component's _make_engine.
        .def(py::init<double, double>(), py::arg("scale"), py::arg("bias") = 0.0)
        .def("apply", &AffineKernel::apply, py::arg("x"),
             "Apply the affine map to a single value.")
        // py::call_guard releases the GIL for the duration of the C++ loop, so other Python threads
        // keep running during a big batch. Arg/return conversion happens outside the guard, so this
        // is safe. This is what makes a heavy C++ kernel a good citizen in threaded/async Python.
        .def("apply_many", &AffineKernel::apply_many, py::arg("xs"),
             py::call_guard<py::gil_scoped_release>(),
             "Apply the affine map to a batch, looping in C++ with the GIL released.")
        .def_property_readonly("scale", &AffineKernel::scale)
        .def_property_readonly("bias", &AffineKernel::bias)
        .def("__repr__", [](const AffineKernel &k) {
            return "AffineKernel(scale=" + std::to_string(k.scale()) +
                   ", bias=" + std::to_string(k.bias()) + ")";
        });

    // C++ exceptions are auto-translated: std::invalid_argument -> Python ValueError,
    // std::out_of_range -> IndexError, std::runtime_error -> RuntimeError, etc. So a bad-argument
    // failure from the kernel surfaces as an ordinary Python exception. Use py::register_exception
    // to map a custom C++ exception type to a custom Python one.
}
