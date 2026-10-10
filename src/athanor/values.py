"""Compact immutable values shared by data, declarations, and expression bodies."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import cast

from .errors import ValidationError


def _valid_text(value: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise ValidationError("Strings must contain valid Unicode code points") from exc


@dataclass(frozen=True, slots=True, order=True)
class Ref:
    """A local identity plus an exact revision; never an implicit live link."""

    id: str
    revision: int = 1
    namespace: str = "local"

    def __post_init__(self) -> None:
        if (
            type(self.id) is not str
            or len(self.id) != 9
            or self.id[0] != "s"
            or not self.id[1:].isascii()
            or not self.id[1:].isdigit()
            or int(self.id[1:]) < 1
            or type(self.revision) is not int
            or self.revision < 1
            or type(self.namespace) is not str
            or not self.namespace
        ):
            raise ValidationError("Reference requires a local s######## ID and positive revision")
        _valid_text(self.namespace)

    def __str__(self) -> str:
        return f"{self.namespace}:{self.id}@{self.revision}"


type Value = None | bool | str | Fraction | Ref | tuple[Value, ...] | Record


@dataclass(frozen=True, slots=True)
class Record(Mapping[str, Value]):
    """Sorted string-keyed immutable mapping. Construct with record()/freeze()."""

    entries: tuple[tuple[str, Value], ...] = ()

    def __post_init__(self) -> None:
        if type(self.entries) is not tuple:
            raise ValidationError("Record entries must be a tuple")
        previous: str | None = None
        for entry in self.entries:
            if type(entry) is not tuple or len(entry) != 2:
                raise ValidationError("Malformed record entry")
            key, value = entry
            if type(key) is not str or not key or (previous is not None and key <= previous):
                raise ValidationError("Record keys must be nonempty, unique, and sorted")
            _valid_text(key)
            _canonical(value)
            previous = key

    def __getitem__(self, key: str) -> Value:
        for name, value in self.entries:
            if name == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (name for name, _ in self.entries)

    def __len__(self) -> int:
        return len(self.entries)


def _canonical(value: object) -> None:
    pending: list[tuple[object, int]] = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if depth > 64:
            raise ValidationError("Canonical value nesting exceeds 64")
        if type(item) is str:
            _valid_text(item)
        elif item is None or type(item) in (bool, Fraction, Ref):
            continue
        elif isinstance(item, Record):
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is tuple:
            pending.extend((child, depth + 1) for child in item)
        else:
            raise ValidationError(f"Noncanonical internal value: {type(item).__name__}")


def freeze(value: object, *, _depth: int = 0) -> Value:
    """Copy structured input; normalize all exact numbers to Fraction; reject floats."""
    if _depth > 64:
        raise ValidationError("Input value nesting exceeds 64; cyclic input is not allowed")
    if value is None or type(value) in (bool, str, Fraction, Ref, Record):
        _canonical(value)
        return cast(Value, value)
    if type(value) is int:
        return Fraction(value)
    if isinstance(value, Mapping):
        raw = dict(value)
        if any(type(key) is not str or not key for key in raw):
            raise ValidationError("Record keys must be nonempty strings")
        return Record(
            tuple(
                (key, freeze(raw[key], _depth=_depth + 1))
                for key in sorted(cast(dict[str, object], raw))
            )
        )
    if type(value) in (list, tuple):
        return tuple(
            freeze(item, _depth=_depth + 1)
            for item in cast(list[object] | tuple[object, ...], value)
        )
    raise ValidationError(f"Unsupported value type: {type(value).__name__}; use exact values")


def record(values: Mapping[str, object] | None = None, **kwargs: object) -> Record:
    raw = dict(values or {})
    raw.update(kwargs)
    if any(type(key) is not str or not key for key in raw):
        raise ValidationError("Record keys must be nonempty strings")
    return Record(tuple((key, freeze(raw[key])) for key in sorted(raw)))


def references(value: Value) -> Iterator[Ref]:
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, Ref):
            yield current
        elif isinstance(current, Record):
            pending.extend(reversed(tuple(current.values())))
        elif isinstance(current, tuple):
            pending.extend(reversed(current))


def exact_equal(left: Value, right: Value) -> bool:
    """Avoid Python's bool/number equality while using canonical rational equality."""
    if type(left) is not type(right):
        return False
    if isinstance(left, Record) and isinstance(right, Record):
        return tuple(left) == tuple(right) and all(exact_equal(left[k], right[k]) for k in left)
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            exact_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def as_record(value: Value, context: str = "value") -> Record:
    if not isinstance(value, Record):
        raise ValidationError(f"{context} must be a record")
    return value


def as_tuple(value: Value, context: str = "value") -> tuple[Value, ...]:
    if not isinstance(value, tuple):
        raise ValidationError(f"{context} must be a sequence")
    return value


def as_ref(value: Value, context: str = "value") -> Ref:
    if not isinstance(value, Ref):
        raise ValidationError(f"{context} must be a reference")
    return value


def as_string(value: Value, context: str = "value") -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{context} must be a nonempty string")
    return value


def as_index(value: Value, context: str = "index") -> int:
    if not isinstance(value, Fraction) or value.denominator != 1 or value < 0:
        raise ValidationError(f"{context} must be a nonnegative integer")
    return value.numerator
