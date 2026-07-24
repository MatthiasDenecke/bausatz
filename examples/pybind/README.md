# Example: C++ classes as bausatz components (pybind11)

Drives compiled C++ classes from Python as ordinary config-driven `Component`s. Python keeps the
config/validation/registry/factory; C++ keeps the compute. Two shapes:

```
# Composition — Python owns the tree, a C++ leaf does the compute:
spec ─▶ ComponentFactory ─▶ Affine(CppComponent)            ← Python: pydantic Config, tag "affine"
                                │  _make_engine: AffineConfig(scale, bias)   ← config struct
                                ▼
                           AffineKernel : bausatz::Component ← C++ (extension _affine)

# Native subtree — a C++ composite runs the whole chain, no per-stage bounce to Python:
spec ─▶ ComponentFactory ─▶ NativeSequential(CppComponent)
                                │  _build_children → child Affines
                                │  _make_engine: Sequential([child.engine, ...])
                                ▼
                           Sequential : bausatz::Component   ← chains children in C++
```

## The C++ `bausatz::Component` base

`include/bausatz/component.hpp` declares an abstract base — the C++ mirror of the Python
`Component` — with a virtual `forward`. `AffineKernel` (a leaf) and `Sequential` (a composite) both
derive from it. Because `Sequential` holds its children as `std::shared_ptr<bausatz::Component>` and
dispatches through the base, a whole subtree runs in C++ with a single `forward` call from Python.

## Files

| file | role |
|---|---|
| `include/bausatz/component.hpp` | the C++ abstract base `bausatz::Component` (virtual `forward`) |
| `src/affine.cpp` | `AffineConfig` struct, `AffineKernel` (leaf), `Sequential` (native composite), and the `PYBIND11_MODULE` |
| `bausatz_affine/__init__.py` | `Affine` (config-struct passing) and `NativeSequential` (its engine is the C++ `Sequential`) |
| `setup.py` | builds the extension with `Pybind11Extension` + `build_ext` (`include_dirs=["include"]`) |
| `test_affine.py` | config-struct passing, polymorphism (`isinstance` of the base), native chaining + nesting, GIL/STL batch, and C++→Python exceptions |

## Build & run

```bash
pip install pybind11                         # build-time dependency
python setup.py build_ext --inplace          # compiles bausatz_affine/_affine*.so
python -m pytest test_affine.py
```

Or install it as a package: `pip install --no-build-isolation .`

## What pybind11 is doing for you

- **`py::class_<AffineKernel, bausatz::Component, std::shared_ptr<AffineKernel>>`** — exposes the C++
  class as a Python type *and* records its base + a shared-ownership holder, so instances upcast to
  `shared_ptr<bausatz::Component>` when handed to the `Sequential` constructor.
- **Type casters (`<pybind11/stl.h>`)** — scalars cross as plain numbers; `list[Component]` ⇄
  `std::vector<std::shared_ptr<bausatz::Component>>` and `list[float]` ⇄ `std::vector<double>` are
  automatic. This is why handing a validated config into a C++ constructor needs *no* glue.
- **`py::call_guard<py::gil_scoped_release>()`** — releases the GIL around the C++ batch loop.
- **Exception translation** — `std::invalid_argument` surfaces in Python as `ValueError`, next to
  pydantic's `ValidationError`.

## How the config crosses the boundary

`Affine` uses **config-struct passing**: `_make_engine` builds a typed C++ `AffineConfig` from the
validated pydantic config and hands that one object to the kernel — the C++ side never sees pydantic.
For a handful of fields you can also just unpack them into positional constructor args; for a large
or dynamic config, pass `self.config.model_dump()` as a `dict`. In every case the pydantic `Config`
is the validated source of truth in Python, and only plain data crosses into C++.

## When to use which shape

- **Composition** (`Affine`) is the default: least friction, and the Python tree stays in charge.
- **Native subtree** (`NativeSequential`) pays off when the compute composition itself should run in
  C++ (tight loops over many stages, no per-node GIL crossing). Its children must themselves be
  C++-backed — their `engine` is a `bausatz::Component`. Mixing in a pure-Python child won't work,
  because the C++ `Sequential` only holds `bausatz::Component`s.
- **Trampolines** (`PYBIND11_OVERRIDE`, not shown) are the third shape — for overriding a C++ virtual
  *in Python*, or a C++ driver calling back into Python stages.
