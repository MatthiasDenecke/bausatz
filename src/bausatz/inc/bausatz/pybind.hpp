// Optional pybind11 plumbing for binding native component classes. Include it only from a
// translation unit that already uses pybind11; ``bausatz/component.hpp`` itself stays pybind-free,
// so the base is usable without pybind. Header-only — links nothing.
#pragma once

#include <memory>
#include <string>

#include <pybind11/pybind11.h>

#include "bausatz/component.hpp"

namespace bausatz {

// Register ``bausatz::Component`` as a Python base, exposing its shared plumbing (the read-only
// ``name`` property). Call once per module before binding component classes that derive from it.
inline pybind11::class_<Component, std::shared_ptr<Component>> bind_component_base(
        pybind11::module_ &m, const char *name = "Component") {
    return pybind11::class_<Component, std::shared_ptr<Component>>(m, name)
        .def_property_readonly("name", &Component::name);
}

// Bind a component class ``T`` (deriving from ``Base``, default ``bausatz::Component``) with a
// ``shared_ptr`` holder — so instances upcast to ``shared_ptr<Component>`` when handed to a native
// container — and a default ``__repr__`` (``TypeName`` / ``TypeName(name='…')``). Returns the
// ``class_`` for chaining ``.def(...)``; a later ``.def("__repr__", …)`` overrides the default.
template <typename T, typename Base = Component>
pybind11::class_<T, Base, std::shared_ptr<T>> bind_component(pybind11::module_ &m, const char *name) {
    pybind11::class_<T, Base, std::shared_ptr<T>> cls(m, name);
    std::string type_name(name);
    cls.def("__repr__", [type_name](const T &self) {
        const std::string &n = self.name();
        return n.empty() ? type_name + "()" : type_name + "(name='" + n + "')";
    });
    return cls;
}

}  // namespace bausatz
