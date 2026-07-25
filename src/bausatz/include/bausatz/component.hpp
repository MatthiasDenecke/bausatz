// bausatz::Component — the C++ root for native components that back a Python ``bausatz.CppComponent``.
//
// Part of the bausatz library: it ships with the package, and a downstream pybind11 extension finds
// it with ``bausatz.get_include()`` and ``#include <bausatz/component.hpp>``. Header-only and
// inline — including it links nothing.
//
// The base declares NO compute method on purpose. What a component *computes* (forward, run, encode,
// …) is application-specific and is added by a subclass. ``bausatz::Component`` only provides the
// shared, owner-agnostic root, plus small integration plumbing, so components can be:
//   * owned uniformly as ``std::shared_ptr<Component>`` and passed across the pybind11 boundary as
//     one type, and
//   * carried by library-level tooling that shouldn't need to know any concrete compute signature.
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace bausatz {

class Component {
public:
    virtual ~Component() = default;

    // Components are shared by pointer, never copied (copying a polymorphic base slices).
    Component(const Component &) = delete;
    Component &operator=(const Component &) = delete;

    // Optional label, mirroring the Python ``Component.Config.name`` — plumbing for logging, repr,
    // or lookup, carried onto the component without teaching C++ anything about pydantic.
    const std::string &name() const noexcept { return name_; }
    void set_name(std::string name) { name_ = std::move(name); }

protected:
    Component() = default;  // a base: constructed only through a subclass

private:
    std::string name_;
};

// A list of components held polymorphically — e.g. the children a native container owns.
using ComponentList = std::vector<std::shared_ptr<Component>>;

}  // namespace bausatz
