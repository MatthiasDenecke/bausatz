# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). While on `0.x`, a minor bump may include
breaking changes.

## [0.2.0] - 2026-07-25

### Added

- **Native (C++) components.** `CppComponent` — a `Component` whose behavior is a native object
  (typically a [pybind11](https://pybind11.readthedocs.io)-exposed C++ class). It is declarative:
  name the native class with `class Foo(CppComponent, component=NativeFoo)` and the framework builds
  it from the validated config, marshalling the config fields in as keyword arguments (override
  `_create_component` for custom construction, e.g. containers). The native object is built lazily
  and cached; the pydantic `Config` still owns validation, the `class_name` tag, registration, and
  `to_dict`.
- **Shipped C++ headers.** The package now ships a header-only C++ base `bausatz::Component` plus
  pybind11 binding helpers; a downstream extension locates them with `bausatz.get_include()`. A
  worked, compiled example lives in [`examples/pybind`](examples/pybind) (config-struct passing and a
  native C++ composite).
- **Typed children.** Container child fields are typed as a `Config`
  (`list[Retriever.Config]`, `dict[str, Component.Config]`, an optional one, …). The whole config
  tree is validated in one eager pass — a malformed grandchild fails at `validate()` — and a
  family-base type rejects a child from another family.
- `ComponentFactory.build(config)` and `build_as(config, expected)` — construct a component from an
  already-validated `Config` (`create` is now `build(validate(spec))`).
- `Component.DISCRIMINATOR` class variable and `Registry.version` counter.

### Changed

- **Breaking:** `Component.to_json()` renamed to `to_dict()` (it returns a `dict`).
- **Breaking:** `Component._ensure_children()` is now the public `ensure_built()`. It is
  exception-safe (a failed build stays retriable instead of leaving a half-built object marked done)
  and detects cyclic component graphs.
- **Breaking:** container children are typed `Config`s rather than raw `dict`s, so a malformed
  grandchild is rejected during the parent's validation instead of later at build time.
- `ComponentFactory.config_adapter()` caches on the registry's version counter, fixing a stale
  adapter after a `clear()` + re-register to the same size.

### Removed

- **Breaking:** the decorative `abc.ABC` base on `Component` (it declared no abstract methods).

### Fixed

- `ensure_built()` no longer marks children as built before the build succeeds — a failed build is
  now retriable, and a cyclic graph raises a clear error instead of recursing forever.

## [0.1.0] - 2026-07-06

- Initial release: config-driven component trees — pydantic-validated classes that self-register by
  a `class_name` tag and assemble recursively from plain dict/JSON specs.

[0.2.0]: https://github.com/MatthiasDenecke/bausatz/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/MatthiasDenecke/bausatz/releases/tag/v0.1.0
