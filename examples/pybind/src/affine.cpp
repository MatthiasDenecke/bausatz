// pybind11 bridge demonstrating two things at once:
//
//   1. A C++ abstract base `bausatz::Component` with a native composite. AffineKernel (a leaf) and
//      Sequential (a container) both derive from it; Sequential holds its children by base pointer
//      and runs the whole chain in C++ — no bounce back to Python between stages.
//   2. Config-struct passing. AffineConfig mirrors the pydantic Config as a plain C++ struct; the
//      Python side builds it from the validated config and hands this one typed object to the ctor.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>   // std::vector <-> list, etc.

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "bausatz/component.hpp"

namespace py = pybind11;

// (2) Affine-specific config, mirrored as a plain C++ struct. The Python component fills this from
// its validated pydantic config; the C++ side never sees pydantic.
struct AffineConfig {
    double scale = 1.0;
    double bias = 0.0;
};

// A leaf engine: y = scale * x + bias. IS-A bausatz::Component.
class AffineKernel : public bausatz::Component {
public:
    explicit AffineKernel(const AffineConfig &cfg) : cfg_(cfg) {
        if (cfg.scale == 0.0) {
            // Thrown from C++, surfaced in Python as ValueError (auto-translated by pybind11).
            throw std::invalid_argument("AffineKernel: scale must be non-zero");
        }
    }

    double forward(double x) const override { return cfg_.scale * x + cfg_.bias; }

    // Batched path: a Python list crosses via the stl.h caster, the loop runs in C++ with the GIL
    // released (safe here — forward() is pure C++, no Python re-entry).
    std::vector<double> forward_many(const std::vector<double> &xs) const {
        std::vector<double> out;
        out.reserve(xs.size());
        for (double x : xs) out.push_back(forward(x));
        return out;
    }

    const AffineConfig &config() const { return cfg_; }

private:
    AffineConfig cfg_;
};

// (1) A native composite: holds child components by base pointer and runs the chain entirely in C++.
class Sequential : public bausatz::Component {
public:
    explicit Sequential(std::vector<std::shared_ptr<bausatz::Component>> stages)
        : stages_(std::move(stages)) {}

    double forward(double x) const override {
        for (const auto &stage : stages_) x = stage->forward(x);
        return x;
    }

    std::size_t size() const { return stages_.size(); }

private:
    std::vector<std::shared_ptr<bausatz::Component>> stages_;
};

PYBIND11_MODULE(_affine, m) {
    m.doc() = "pybind11 bridge: a bausatz::Component ABC, a leaf kernel, and a native composite";

    // The abstract base. A shared_ptr holder lets a C++ parent (Sequential) and the Python wrapper
    // share ownership of the same engine. No __init__ — it is abstract; `forward` dispatches
    // virtually, so it works on any derived instance.
    py::class_<bausatz::Component, std::shared_ptr<bausatz::Component>>(m, "Component")
        .def("forward", &bausatz::Component::forward, py::arg("x"),
             "Run the component's compute on a single value.");

    // The mirrored config struct (Q2 option 2): a typed C++ config built on the Python side.
    py::class_<AffineConfig>(m, "AffineConfig")
        .def(py::init<double, double>(), py::arg("scale") = 1.0, py::arg("bias") = 0.0)
        .def_readwrite("scale", &AffineConfig::scale)
        .def_readwrite("bias", &AffineConfig::bias)
        .def("__repr__", [](const AffineConfig &c) {
            return "AffineConfig(scale=" + std::to_string(c.scale) +
                   ", bias=" + std::to_string(c.bias) + ")";
        });

    // Derived types declare the base AND the shared_ptr holder so instances upcast to
    // shared_ptr<bausatz::Component> when handed to Sequential.
    py::class_<AffineKernel, bausatz::Component, std::shared_ptr<AffineKernel>>(m, "AffineKernel")
        .def(py::init<const AffineConfig &>(), py::arg("config"))
        .def("forward_many", &AffineKernel::forward_many, py::arg("xs"),
             py::call_guard<py::gil_scoped_release>(),
             "Apply to a batch, looping in C++ with the GIL released.")
        .def_property_readonly("scale", [](const AffineKernel &k) { return k.config().scale; })
        .def_property_readonly("bias", [](const AffineKernel &k) { return k.config().bias; })
        .def("__repr__", [](const AffineKernel &k) {
            return "AffineKernel(scale=" + std::to_string(k.config().scale) +
                   ", bias=" + std::to_string(k.config().bias) + ")";
        });

    py::class_<Sequential, bausatz::Component, std::shared_ptr<Sequential>>(m, "Sequential")
        // Python list[Component] -> std::vector<std::shared_ptr<bausatz::Component>> via stl.h.
        .def(py::init<std::vector<std::shared_ptr<bausatz::Component>>>(), py::arg("stages"))
        .def_property_readonly("size", &Sequential::size)
        .def("__repr__", [](const Sequential &s) {
            return "Sequential(size=" + std::to_string(s.size()) + ")";
        });
}
