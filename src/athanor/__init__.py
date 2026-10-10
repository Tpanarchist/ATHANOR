"""ATHANOR Goal 001: a local, teachable, deterministic structural kernel."""

from .contracts import (
    Assessment,
    Difference,
    Endpoint,
    Evidence,
    FieldSpec,
    Inspection,
    Limits,
    Receipt,
    ResolvedClaim,
    Role,
    Structure,
)
from .errors import (
    AthanorError,
    ExecutionError,
    PersistenceError,
    ResourceLimitError,
    ValidationError,
)
from .kernel import Athanor
from .programs import Procedure
from .values import Record, Ref, Value, freeze, record

__all__ = [
    "Assessment",
    "Athanor",
    "AthanorError",
    "Difference",
    "Endpoint",
    "Evidence",
    "ExecutionError",
    "FieldSpec",
    "Inspection",
    "Limits",
    "PersistenceError",
    "Procedure",
    "Receipt",
    "Record",
    "Ref",
    "ResolvedClaim",
    "ResourceLimitError",
    "Role",
    "Structure",
    "ValidationError",
    "Value",
    "freeze",
    "record",
]

__version__ = "0.1.0"
