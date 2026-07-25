// pybind11 bridge demonstrating:
//
//   1. Using the library's C++ base: bausatz::Component (from <bausatz/component.hpp>, located via
//      bausatz.get_include()). It has no compute method — the application defines one.
//   2. An app-level compute interface `Transform : bausatz::Component` (forward lives HERE, not in
//      the library base), with a leaf (AffineKernel) and a native composite (Sequential) that runs
//      a whole chain in C++.
//   3. Constructors that take the config fields as keyword arguments, so the Python `CppComponent`
//      builds them generically from the validated config — no per-field marshalling glue.
//
// bausatz::bind_component / bind_component_base (from <bausatz/pybind.hpp>) hide the base +
// shared_ptr-holder boilerplate.

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

// (2) The application's compute interface. It — not the library base — declares `forward`.
class Transform : public bausatz::Component {
public:
    virtual double forward(double x) const = 0;
};

// (3) An internal typed config. It is NOT part of the Python boundary — the constructor takes the
// fields directly, and the Python component marshals its validated config in as keyword arguments.
// A typed struct like this is still handy for carrying config deep into C++ code.
struct AffineConfig {
    double scale = 1.0;
    double bias = 0.0;
};

// A leaf component: y = scale * x + bias. Its constructor takes the config fields as keyword arguments
// (see the binding), so `CppComponent` can build it generically from the pydantic config.
class AffineKernel : public Transform {
public:
    AffineKernel(double scale, double bias) : cfg_{scale, bias} {
        if (scale == 0.0) {
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

// (2) A native composite: holds child transforms and runs the chain entirely in C++.
class Sequential : public Transform {
public:
    explicit Sequential(std::vector<std::shared_ptr<Transform>> stages) : stages_(std::move(stages)) {}

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

    // Register the library base (exposes its `name` plumbing) and the app's compute interface.
    bausatz::bind_component_base(m);
    bausatz::bind_component<Transform>(m, "Transform")
        .def("forward", &Transform::forward, py::arg("x"),
             "Run the transform on a single value.");

    // AffineKernel's constructor takes the config fields as keyword arguments, so a Python
    // CppComponent can build it generically by splatting the validated config: AffineKernel(**fields).
    bausatz::bind_component<AffineKernel, Transform>(m, "AffineKernel")
        .def(py::init<double, double>(), py::arg("scale") = 1.0, py::arg("bias") = 0.0)
        .def("forward_many", &AffineKernel::forward_many, py::arg("xs"),
             py::call_guard<py::gil_scoped_release>(),
             "Apply to a batch, looping in C++ with the GIL released.")
        .def_property_readonly("scale", [](const AffineKernel &k) { return k.config().scale; })
        .def_property_readonly("bias", [](const AffineKernel &k) { return k.config().bias; })
        .def("__repr__", [](const AffineKernel &k) {
            return "AffineKernel(scale=" + std::to_string(k.config().scale) +
                   ", bias=" + std::to_string(k.config().bias) + ")";
        });

    bausatz::bind_component<Sequential, Transform>(m, "Sequential")
        // Python list[Transform] -> std::vector<std::shared_ptr<Transform>> via stl.h.
        .def(py::init<std::vector<std::shared_ptr<Transform>>>(), py::arg("stages"))
        .def_property_readonly("size", &Sequential::size)
        .def("__repr__", [](const Sequential &s) {
            return "Sequential(size=" + std::to_string(s.size()) + ")";
        });
}
