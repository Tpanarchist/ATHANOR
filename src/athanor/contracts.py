"""Public immutable contracts; declarations become ordinary structural records."""

from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

from .errors import ValidationError
from .values import Record, Ref, Value, as_record, exact_equal, freeze, record


class Role(StrEnum):
    DATA = "data"
    CATEGORY = "category"
    RELATION = "relation"
    OPERATION = "operation"
    CLAIM = "claim"


@dataclass(frozen=True, slots=True)
class Structure:
    ref: Ref
    role: Role
    category: Ref | None
    fields: Record
    members: tuple[Ref, ...] = ()


@dataclass(frozen=True, slots=True)
class FieldSpec:
    kind: str = "any"
    required: bool = True
    minimum: Fraction | int | None = None
    maximum: Fraction | int | None = None

    def to_record(self) -> Record:
        result = record(
            kind=self.kind, required=self.required, minimum=self.minimum, maximum=self.maximum
        )
        validate_field_spec(result)
        return result


KINDS = frozenset({"any", "number", "string", "bool", "ref", "sequence", "record"})


def validate_field_spec(spec: Record) -> None:
    if set(spec) != {"kind", "required", "minimum", "maximum"}:
        raise ValidationError("Malformed field constraint")
    if spec["kind"] not in KINDS or type(spec["required"]) is not bool:
        raise ValidationError("Unsupported field type or required flag")
    low, high = spec["minimum"], spec["maximum"]
    if low is not None or high is not None:
        if spec["kind"] != "number" or any(
            bound is not None and not isinstance(bound, Fraction) for bound in (low, high)
        ):
            raise ValidationError("Numeric bounds require exact number fields")
        if isinstance(low, Fraction) and isinstance(high, Fraction) and low > high:
            raise ValidationError("Minimum exceeds maximum")


def validate_fields(fields: Record, schema: Record) -> None:
    if set(fields) - set(schema):
        raise ValidationError(f"Undeclared fields: {sorted(set(fields) - set(schema))}")
    for name, raw in schema.items():
        spec = as_record(raw, "field specification")
        validate_field_spec(spec)
        if name not in fields:
            if spec["required"]:
                raise ValidationError(f"Required field missing: {name}")
            continue
        value = fields[name]
        kind = spec["kind"]
        correct = {
            "any": True,
            "number": isinstance(value, Fraction),
            "string": isinstance(value, str),
            "bool": type(value) is bool,
            "ref": isinstance(value, Ref),
            "sequence": isinstance(value, tuple),
            "record": isinstance(value, Record),
        }[str(kind)]
        if not correct:
            raise ValidationError(f"Field {name} requires {kind}")
        if isinstance(value, Fraction):
            low, high = spec["minimum"], spec["maximum"]
            if isinstance(low, Fraction) and value < low:
                raise ValidationError(f"Field {name} is below its minimum")
            if isinstance(high, Fraction) and value > high:
                raise ValidationError(f"Field {name} exceeds its maximum")


@dataclass(frozen=True, slots=True)
class Endpoint:
    role: Role | None = None
    category: Ref | None = None

    def to_record(self) -> Record:
        return record(role=None if self.role is None else self.role.value, category=self.category)


@dataclass(frozen=True, slots=True)
class Evidence:
    source: str
    stance: str = "positive"
    confidence: Fraction | int | None = None
    note: str = ""

    def to_record(self) -> Record:
        result = record(
            source=self.source, stance=self.stance, confidence=self.confidence, note=self.note
        )
        validate_evidence(result)
        return result


def validate_evidence(evidence: Record) -> None:
    if set(evidence) != {"source", "stance", "confidence", "note"}:
        raise ValidationError("Malformed evidence")
    if not isinstance(evidence["source"], str) or not evidence["source"]:
        raise ValidationError("Evidence requires an attributed source")
    if evidence["stance"] not in {"positive", "negative", "uncertain"}:
        raise ValidationError("Evidence stance must be positive, negative, or uncertain")
    if not isinstance(evidence["note"], str):
        raise ValidationError("Evidence note must be a string")
    confidence = evidence["confidence"]
    if confidence is not None and (
        not isinstance(confidence, Fraction) or not 0 <= confidence <= 1
    ):
        raise ValidationError("Confidence must be an exact number between zero and one")


@dataclass(frozen=True, slots=True)
class Receipt:
    sequence: int
    action: str
    result: Ref
    inputs: tuple[Ref, ...]
    operation: Ref | None
    causes: tuple[int, ...]
    details: Record


@dataclass(frozen=True, slots=True)
class Inspection:
    structure: Structure
    receipt: Receipt


@dataclass(frozen=True, slots=True)
class Difference:
    same_identity: bool
    same_revision: bool
    same_state: bool
    changed_fields: tuple[str, ...]
    same_members: bool
    same_category: bool
    same_role: bool


def compare(left: Structure, right: Structure) -> Difference:
    keys = sorted(set(left.fields) | set(right.fields))
    changed = tuple(
        key
        for key in keys
        if key not in left.fields
        or key not in right.fields
        or not exact_equal(left.fields[key], right.fields[key])
    )
    members = left.members == right.members
    category = left.category == right.category
    role = left.role == right.role
    return Difference(
        left.ref.id == right.ref.id,
        left.ref == right.ref,
        not changed and members and category and role,
        changed,
        members,
        category,
        role,
    )


@dataclass(frozen=True, slots=True)
class ResolvedClaim:
    claim: Ref
    source: Ref
    via: tuple[Ref, ...]


class Assessment(StrEnum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    CONFLICT = "conflict"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Limits:
    max_steps: int = 100_000
    max_depth: int = 64

    def __post_init__(self) -> None:
        if any(type(v) is not int or v < 1 for v in (self.max_steps, self.max_depth)):
            raise ValidationError("Execution limits must be positive integers")


def fields_record(fields: dict[str, object] | Record | None) -> Record:
    value: Value = freeze({} if fields is None else fields)
    return as_record(value, "fields")
