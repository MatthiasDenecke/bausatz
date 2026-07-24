"""
Class registry for components. The class registry owns the tag -> class mapping (a pure lookup table).
"""

from __future__ import annotations

from collections.abc import Iterator


class DuplicateTagError(ValueError):
    """Two classes tried to register under the same tag."""


class UnknownTagError(KeyError):
    """A spec referenced a tag that is not registered."""


# --------------------------------------------------------------------------- #
# Registry: the tag -> class mapping. No construction logic lives here.       #
# --------------------------------------------------------------------------- #
class Registry[T]:

    def __init__(self) -> None:
        self._map: dict[str, type[T]] = {}
        self._version = 0  # bumped whenever the set of registered classes changes

    @property
    def version(self) -> int:
        """A counter incremented on every mutation to the class set. Lets callers cache things
        derived from the registry (e.g. the factory's discriminated-union adapter) and rebuild them
        exactly when it changes — robust where a size check is not (clear + re-register to the same
        size leaves the size unchanged but the classes different)."""
        return self._version

    def register(self, tag: str, cls: type[T]) -> None:
        existing = self._map.get(tag)
        if existing is not None and existing is not cls:
            raise DuplicateTagError(
                f"tag {tag!r} is already registered to {existing.__qualname__}"
            )
        if existing is cls:
            return  # idempotent re-registration: nothing changed, keep the version stable
        self._map[tag] = cls
        self._version += 1

    def get(self, tag: str) -> type[T]:
        try:
            return self._map[tag]
        except KeyError:
            raise UnknownTagError(
                f"unknown tag {tag!r}; registered: {sorted(self._map)}"
            ) from None

    def __contains__(self, tag: object) -> bool:
        return tag in self._map

    def __iter__(self) -> Iterator[str]:
        return iter(self._map)

    def __len__(self) -> int:
        return len(self._map)

    def tags(self) -> list[str]:
        return list(self._map)

    def classes(self) -> list[type[T]]:
        return list(self._map.values())

    def items(self) -> list[tuple[str, type[T]]]:
        return list(self._map.items())

    def clear(self) -> None:
        if self._map:
            self._version += 1
        self._map.clear()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({sorted(self._map)!r})"
