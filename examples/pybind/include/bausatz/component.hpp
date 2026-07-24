// The C++ mirror of the Python ``bausatz.Component``: an abstract base every native engine
// implements. Keeping the interface here (rather than in one .cpp) is what lets a native container
// hold heterogeneous children by base pointer and run a whole subtree in C++.
#pragma once

namespace bausatz {

// Abstract base for native components. Concrete engines override ``forward``; a composite engine
// (see Sequential) holds children as ``std::shared_ptr<Component>`` and dispatches through it.
class Component {
public:
    virtual ~Component() = default;

    // Run the component's compute on a single value.
    virtual double forward(double x) const = 0;
};

}  // namespace bausatz
