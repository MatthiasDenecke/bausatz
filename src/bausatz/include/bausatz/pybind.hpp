// Optional pybind11 plumbing for binding native engines as bausatz components. Include it only from
// a translation unit that already uses pybind11; ``bausatz/component.hpp`` itself stays pybind-free,
// so the base is usable without pybind. Header-only — links nothing.
#pragma once

#include <memory>

#include <pybind11/pybind11.h>

#include "bausatz/component.hpp"

namespace bausatz {

// Register ``bausatz::Component`` as a Python base, exposing its shared plumbing (the ``name``
// property). Call once per module before binding engine classes that derive from it.
inline pybind11::class_<Component, std::shared_ptr<Component>> bind_component_base(
        pybind11::module_ &m, const char *name = "Component") {
    return pybind11::class_<Component, std::shared_ptr<Component>>(m, name)
        .def_property("name", &Component::name, &Component::set_name);
}

// Bind an engine class ``T`` (deriving from ``Base``, default ``bausatz::Component``) with a
// ``shared_ptr`` holder — so instances upcast to ``shared_ptr<Component>`` when handed to a native
// container. Returns the ``class_`` for chaining ``.def(...)``, hiding the base + holder incantation.
template <typename T, typename Base = Component>
pybind11::class_<T, Base, std::shared_ptr<T>> bind_component(pybind11::module_ &m, const char *name) {
    return pybind11::class_<T, Base, std::shared_ptr<T>>(m, name);
}

}  // namespace bausatz
