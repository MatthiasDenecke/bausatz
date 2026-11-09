// Optional pybind11 plumbing for binding native component classes. Include it only from a
// translation unit that already uses pybind11; ``bausatz/component.hpp`` itself stays pybind-free,
// so the base is usable without pybind. Header-only — links nothing.
#pragma once

#include <memory>
#include <string>

#include <pybind11/pybind11.h>

#include "bausatz/component.hpp"

namespace bausatz {

// Register ``bausatz::Component`` as a Python base, exposing its shared plumbing: the read-only
// ``name`` property and a default ``__repr__`` (``TypeName()`` / ``TypeName(name='…')``, using the
// instance's actual Python type). Call once per module before binding component classes that
// derive from it.
//
// The default ``__repr__`` lives here, on the base, so a derived class *inherits* it rather than
// defining its own. A derived class's ``.def("__repr__", …)`` therefore replaces it. If
// ``bind_component`` defined the repr on every class instead, a later ``.def`` on the same class
// would only append an overload behind the existing ``(const T &)`` one, which matches every
// instance first, so the custom repr would never run.
inline pybind11::class_<Component, std::shared_ptr<Component>> bind_component_base(
        pybind11::module_ &m, const char *name = "Component") {
    namespace py = pybind11;
    return py::class_<Component, std::shared_ptr<Component>>(m, name)
        .def_property_readonly("name", &Component::name)
        .def("__repr__", [](py::handle self) {
            const std::string &n = self.cast<const Component &>().name();
            std::string type_name = py::type::of(self).attr("__name__").cast<std::string>();
            if (n.empty()) return type_name + "()";
            return type_name + "(name=" + py::repr(py::str(n)).cast<std::string>() + ")";
        });
}

// Bind a component class ``T`` (deriving from ``Base``, default ``bausatz::Component``) with a
// ``shared_ptr`` holder, so instances upcast to ``shared_ptr<Component>`` when handed to a native
// container. ``T`` inherits the default ``__repr__`` from the base (see ``bind_component_base``).
// Returns the ``class_`` for chaining ``.def(...)``; a chained ``.def("__repr__", …)`` replaces the
// default.
template <typename T, typename Base = Component>
pybind11::class_<T, Base, std::shared_ptr<T>> bind_component(pybind11::module_ &m, const char *name) {
    return pybind11::class_<T, Base, std::shared_ptr<T>>(m, name);
}

}  // namespace bausatz
