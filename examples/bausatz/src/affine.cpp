// pybind11 bridge demonstrating:
//
//   1. Using the library's C++ base: bausatz::Component (from <bausatz/component.hpp>, located via
//      bausatz.get_include()). It has no compute method — the application defines one — and carries
//      an immutable `name` set at construction.
//   2. An app-level compute interface `Transform : bausatz::Component` (forward lives HERE, not in
//      the library base), with a leaf (AffineKernel) and a native composite (Sequential) that runs
//      a whole chain in C++.
//   3. Constructors that take the config fields (and the name) as keyword arguments, so the Python
//      `CppComponent` builds them generically from the validated config — no per-field marshalling.
//
// Boilerplate stays small: bausatz::bind_component / bind_component_base (from <bausatz/pybind.hpp>)
// hide the base + shared_ptr-holder wiring, and the base supplies a default __repr__ that a class
// can replace with its own `.def("__repr__", ...)` (Sequential does). The batched `forward_many`
// lives once on Transform, not on every leaf.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>   // std::vector <-> list, etc.

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <bausatz/component.hpp>
#include <bausatz/pybind.hpp>

namespace py = pybind11;

// (2) The application's compute interface. It — not the library base — declares `forward`, and
// provides a default batched `forward_many` so leaves don't each reimplement it.
class Transform : public bausatz::Component {
public:
    virtual double forward(double x) const = 0;

    // Loops forward() in C++; bound once (with the GIL released) on Transform, inherited by all.
    std::vector<double> forward_many(const std::vector<double> &xs) const {
        std::vector<double> out;
        out.reserve(xs.size());
        for (double x : xs) out.push_back(forward(x));
        return out;
    }

protected:
    explicit Transform(std::string name) : bausatz::Component(std::move(name)) {}
};

// An internal typed config — not part of the Python boundary (the ctor takes the fields directly).
struct AffineConfig {
    double scale = 1.0;
    double bias = 0.0;
};

// A leaf component: y = scale * x + bias. Its ctor takes the config fields plus the name as keyword
// arguments (see the binding), so `CppComponent` can build it generically from the pydantic config.
class AffineKernel : public Transform {
public:
    AffineKernel(double scale, double bias, std::string name)
        : Transform(std::move(name)), cfg_{scale, bias} {
        if (scale == 0.0) {
            // Thrown from C++, surfaced in Python as ValueError (auto-translated by pybind11).
            throw std::invalid_argument("AffineKernel: scale must be non-zero");
        }
    }

    double forward(double x) const override { return cfg_.scale * x + cfg_.bias; }

private:
    AffineConfig cfg_;
};

// (2) A native composite: holds child transforms and runs the chain entirely in C++.
class Sequential : public Transform {
public:
    Sequential(std::vector<std::shared_ptr<Transform>> stages, std::string name)
        : Transform(std::move(name)), stages_(std::move(stages)) {
        // pybind11 converts a Python None in the list to a null shared_ptr. Reject it here, where it
        // surfaces as a Python ValueError, rather than dereference it in forward() and crash.
        for (std::size_t i = 0; i < stages_.size(); ++i) {
            if (!stages_[i]) {
                throw std::invalid_argument("Sequential: stages[" + std::to_string(i) + "] is None");
            }
        }
    }

    double forward(double x) const override {
        for (const auto &stage : stages_) x = stage->forward(x);
        return x;
    }

    std::size_t size() const { return stages_.size(); }

private:
    std::vector<std::shared_ptr<Transform>> stages_;
};

PYBIND11_MODULE(_affine, m) {
    m.doc() = "pybind11 bridge over the bausatz::Component base: an app Transform interface, a leaf "
              "kernel, and a native composite";

    // Register the library base (exposes the read-only `name` and a default __repr__ every class
    // inherits) and the app's compute interface. Only Sequential overrides the default repr below.
    bausatz::bind_component_base(m);
    bausatz::bind_component<Transform>(m, "Transform")
        .def("forward", &Transform::forward, py::arg("x"), "Run the transform on a single value.")
        .def("forward_many", &Transform::forward_many, py::arg("xs"),
             py::call_guard<py::gil_scoped_release>(),
             "Apply to a batch, looping in C++ with the GIL released.");

    bausatz::bind_component<AffineKernel, Transform>(m, "AffineKernel")
        .def(py::init<double, double, std::string>(),
             py::arg("scale") = 1.0, py::arg("bias") = 0.0, py::arg("name") = "");

    bausatz::bind_component<Sequential, Transform>(m, "Sequential")
        // Python list[Transform] -> std::vector<std::shared_ptr<Transform>> via stl.h.
        .def(py::init<std::vector<std::shared_ptr<Transform>>, std::string>(),
             py::arg("stages"), py::arg("name") = "")
        .def_property_readonly("size", &Sequential::size)
        // Replaces the default __repr__ inherited from the base binding.
        .def("__repr__", [](const Sequential &s) {
            std::string r = "Sequential(";
            if (!s.name().empty()) r += "name=" + py::repr(py::str(s.name())).cast<std::string>() + ", ";
            return r + "size=" + std::to_string(s.size()) + ")";
        });
}
