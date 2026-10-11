"""Structured teaching language: portable expression data, never Python source."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction

from .contracts import Endpoint, Limits, Structure
from .errors import ExecutionError, ResourceLimitError, ValidationError
from .values import (
    Record,
    Ref,
    Value,
    as_index,
    as_record,
    as_ref,
    as_string,
    as_tuple,
    exact_equal,
    record,
)


def literal(value: object) -> Record:
    return record(op="literal", value=value)


def argument(index: int) -> Record:
    return record(op="argument", index=index)


def field(of: Record, name: str) -> Record:
    return record(op="field", of=of, name=name)


def has_field(of: Record, name: str) -> Record:
    return record(op="has_field", of=of, name=name)


def members(of: Record) -> Record:
    return record(op="members", of=of)


def category(of: Record) -> Record:
    """Reflect the exact category reference, or None for an uncategorized participant."""
    return record(op="category", of=of)


def item(of: Record, index: int) -> Record:
    return record(op="item", of=of, index=index)


def binary(op: str, left: Record, right: Record) -> Record:
    return record(op=op, left=left, right=right)


def choose(condition: Record, then: Record, otherwise: Record) -> Record:
    return record(op="if", condition=condition, then=then, otherwise=otherwise)


def sequence(*expressions: Record) -> Record:
    return record(op="sequence", items=expressions)


def record_expression(fields: Mapping[str, Record]) -> Record:
    """Construct immutable native records from validated child expressions."""
    return record(op="record", fields=fields)


def variable(name: str) -> Record:
    """The fold body supplies item and acc; there are no hidden global variables."""
    return record(op="variable", name=name)


def fold(items: Record, initial: Record, body: Record) -> Record:
    return record(op="fold", items=items, initial=initial, body=body)


def call(operation: Ref | str, *arguments: Record) -> Record:
    """Only 'self' is a symbolic operation reference. All others are exact Refs."""
    return record(op="call", operation=operation, arguments=arguments)


@dataclass(frozen=True, slots=True)
class Procedure:
    inputs: tuple[Endpoint, ...]
    result: Ref | None
    fields: Mapping[str, Record]
    members: Record | None = None
    update: int | None = None

    def to_record(self) -> Record:
        return record(
            inputs=tuple(endpoint.to_record() for endpoint in self.inputs),
            result=self.result,
            fields=self.fields,
            members=self.members,
            update=self.update,
        )


_BINARY = frozenset({"add", "sub", "mul", "div", "eq", "lt", "le"})
_KEYS = {
    "literal": {"op", "value"},
    "argument": {"op", "index"},
    "field": {"op", "of", "name"},
    "has_field": {"op", "of", "name"},
    "members": {"op", "of"},
    "category": {"op", "of"},
    "item": {"op", "of", "index"},
    "if": {"op", "condition", "then", "otherwise"},
    "sequence": {"op", "items"},
    "record": {"op", "fields"},
    "variable": {"op", "name"},
    "fold": {"op", "items", "initial", "body"},
    "call": {"op", "operation", "arguments"},
}


def validate_expression(
    expression: Record,
    arity: int,
    check_operation: Callable[[Ref], None],
    *,
    in_fold: bool = False,
    depth: int = 0,
) -> None:
    if depth > 64:
        raise ValidationError("Expression nesting exceeds the installation limit (64)")
    op = as_string(expression.get("op"), "expression opcode")
    expected = {"op", "left", "right"} if op in _BINARY else _KEYS.get(op)
    if expected is None or set(expression) != expected:
        raise ValidationError(f"Unknown or malformed expression: {op}")

    def child(raw: Value, folded: bool = in_fold) -> None:
        validate_expression(
            as_record(raw, "expression"), arity, check_operation, in_fold=folded, depth=depth + 1
        )

    if op == "literal":
        return
    if op == "argument":
        if as_index(expression["index"]) >= arity:
            raise ValidationError("Argument index exceeds procedure arity")
    elif op in {"field", "has_field", "members", "category", "item"}:
        child(expression["of"])
        if op in {"field", "has_field"}:
            as_string(expression["name"], "field name")
        if op == "item":
            as_index(expression["index"])
    elif op in _BINARY:
        child(expression["left"])
        child(expression["right"])
    elif op == "if":
        for key in ("condition", "then", "otherwise"):
            child(expression[key])
    elif op == "sequence":
        for raw in as_tuple(expression["items"]):
            child(raw)
    elif op == "record":
        for raw in as_record(expression["fields"]).values():
            child(raw)
    elif op == "variable":
        if not in_fold or expression["name"] not in {"item", "acc"}:
            raise ValidationError("Only item/acc variables inside fold bodies are available")
    elif op == "fold":
        child(expression["items"])
        child(expression["initial"])
        child(expression["body"], True)
    elif op == "call":
        target = expression["operation"]
        if target != "self":
            check_operation(as_ref(target, "called operation"))
        for raw in as_tuple(expression["arguments"]):
            child(raw)


class Evaluator:
    """Transient invocation state. The kernel provides staged reads and calls."""

    def __init__(self, get: Callable[[Ref], Structure], limits: Limits) -> None:
        self.get = get
        self.limits = limits
        self.steps = 0

    def tick(self) -> None:
        self.steps += 1
        if self.steps > self.limits.max_steps:
            raise ResourceLimitError("Invocation exceeded its expression-step budget")

    def evaluate(
        self,
        expression: Record,
        arguments: tuple[Ref, ...],
        invoke: Callable[[Ref, tuple[Ref, ...]], Ref],
        self_ref: Ref,
        variables: Mapping[str, Value] | None = None,
    ) -> Value:
        self.tick()
        op = as_string(expression["op"])
        scope = {} if variables is None else variables

        def child(raw: Value, local: Mapping[str, Value] = scope) -> Value:
            return self.evaluate(as_record(raw), arguments, invoke, self_ref, local)

        if op == "literal":
            return expression["value"]
        if op == "argument":
            return arguments[as_index(expression["index"])]
        if op == "variable":
            return scope[as_string(expression["name"])]
        if op in {"field", "has_field"}:
            source = child(expression["of"])
            values = self.get(source).fields if isinstance(source, Ref) else as_record(source)
            name = as_string(expression["name"])
            if op == "has_field":
                return name in values
            if name not in values:
                raise ExecutionError(f"Field unavailable: {name}")
            return values[name]
        if op == "members":
            return self.get(as_ref(child(expression["of"]))).members
        if op == "category":
            return self.get(as_ref(child(expression["of"]))).category
        if op == "item":
            sequence_values = as_tuple(child(expression["of"]))
            index = as_index(expression["index"])
            if index >= len(sequence_values):
                raise ExecutionError("Sequence index out of range")
            return sequence_values[index]
        if op == "sequence":
            return tuple(child(raw) for raw in as_tuple(expression["items"]))
        if op == "record":
            return record({key: child(raw) for key, raw in as_record(expression["fields"]).items()})
        if op == "if":
            condition = child(expression["condition"])
            if type(condition) is not bool:
                raise ExecutionError("Conditional requires an exact boolean")
            return child(expression["then"] if condition else expression["otherwise"])
        if op == "fold":
            folded_values = as_tuple(child(expression["items"]))
            accumulator = child(expression["initial"])
            for value in folded_values:
                self.tick()
                accumulator = child(expression["body"], {"item": value, "acc": accumulator})
            return accumulator
        if op == "call":
            target = expression["operation"]
            operation = self_ref if target == "self" else as_ref(target)
            inputs = tuple(as_ref(child(raw)) for raw in as_tuple(expression["arguments"]))
            return invoke(operation, inputs)
        left, right = child(expression["left"]), child(expression["right"])
        if op == "eq":
            return exact_equal(left, right)
        if not isinstance(left, Fraction) or not isinstance(right, Fraction):
            raise ExecutionError(f"{op} requires exact numeric operands")
        if op == "lt":
            return left < right
        if op == "le":
            return left <= right
        if op == "add":
            return left + right
        if op == "sub":
            return left - right
        if op == "mul":
            return left * right
        if op == "div":
            if right == 0:
                raise ExecutionError("Division by zero")
            return left / right
        raise ExecutionError(f"Unavailable opcode: {op}")
