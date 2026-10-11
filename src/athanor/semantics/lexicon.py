"""Persistent, attributed expression conventions; spelling is not semantic identity."""

import re
import unicodedata
from dataclasses import dataclass

from athanor import (
    Assessment,
    Athanor,
    Endpoint,
    Evidence,
    Record,
    Ref,
    Role,
    ValidationError,
    record,
)
from athanor.values import as_record, as_ref, as_string, as_tuple, exact_equal


def normalize(expression: str) -> str:
    if type(expression) is not str or not expression.strip() or len(expression) > 256:
        raise ValidationError("Expressions must contain 1..256 characters")
    normalized = " ".join(unicodedata.normalize("NFKC", expression).casefold().split())
    if not re.fullmatch(r"[^\W\d_]+(?:[ -][^\W\d_]+)*", normalized):
        raise ValidationError("Taught expressions must contain bounded words, spaces or hyphens")
    return normalized


@dataclass(frozen=True, slots=True)
class Binding:
    ref: Ref
    symbol: Ref
    referent: Ref
    context: Record
    assessment: Assessment


class Lexicon:
    def __init__(self, knowledge: Athanor) -> None:
        self.knowledge = knowledge
        found = [
            node.ref
            for node in knowledge.structures
            if node.role == Role.RELATION and node.fields["name"] == "semantic denotes"
        ]
        if len(found) > 1:
            raise ValidationError(
                "Multiple semantic convention relations require explicit unification"
            )
        self.relation = (
            found[0]
            if found
            else knowledge.teach_relation(
                "semantic denotes",
                Endpoint(Role.DATA),
                Endpoint(Role.DATA),
                source="semantic interface initialization",
            )
        )

    def find(self, expression: str) -> Ref | None:
        normalized = normalize(expression)
        found = [
            node.ref
            for node in self.knowledge.structures
            if node.fields.get("kind") == "symbol" and node.fields.get("expression") == normalized
        ]
        if len(found) > 1:
            raise ValidationError("Duplicate symbol identities require explicit unification")
        return found[0] if found else None

    def symbol(self, expression: str, *, source: str = "lexical observation") -> Ref:
        normalized = normalize(expression)
        existing = self.find(normalized)
        return (
            existing
            if existing is not None
            else self.knowledge.create(
                fields={
                    "kind": "symbol",
                    "expression": normalized,
                    "tokens": tuple(normalized.split()),
                },
                source=source,
            )
        )

    def bind(
        self,
        expression: str,
        referent: Ref,
        *,
        context: Record | None = None,
        evidence: Evidence | None = None,
    ) -> Ref:
        if not isinstance(referent, Ref):
            raise ValidationError("Meaning referents require exact references")
        self.knowledge.get(referent)
        scope = record() if context is None else as_record(context)
        attribution = Evidence("structured meaning instructor") if evidence is None else evidence
        symbol = self.symbol(expression, source=attribution.source)
        binding = self.knowledge.create(
            fields={
                "kind": "meaning_binding",
                "symbol": symbol,
                "referent": referent,
                "context": scope,
            },
            source=attribution.source,
        )
        self.knowledge.bind(self.relation, symbol, binding, attribution)
        return binding

    def bindings(self, expression: str, context: Record | None = None) -> tuple[Binding, ...]:
        symbol = self.find(expression)
        if symbol is None:
            return ()
        actual = record() if context is None else as_record(context)
        superseded: set[Ref] = set()
        for node in self.knowledge.structures:
            if (
                node.fields.get("kind") == "meaning_correction"
                and node.fields.get("symbol") == symbol
            ):
                superseded.update(as_ref(value) for value in as_tuple(node.fields["supersedes"]))
        result: list[Binding] = []
        for resolved in self.knowledge.relations(symbol, self.relation):
            claim = self.knowledge.get(resolved.claim)
            ref = as_ref(claim.fields["target"])
            node = self.knowledge.get(ref)
            if ref in superseded or node.fields.get("kind") != "meaning_binding":
                continue
            scope = as_record(node.fields["context"])
            # Absent context keeps scoped alternatives live; supplied contradictions exclude them.
            if any(
                key in actual and not exact_equal(actual[key], value)
                for key, value in scope.items()
            ):
                continue
            result.append(
                Binding(
                    ref,
                    symbol,
                    as_ref(node.fields["referent"]),
                    scope,
                    self.knowledge.assess(symbol, self.relation, ref),
                )
            )
        return tuple({binding.ref: binding for binding in result}.values())

    def correct(
        self,
        expression: str,
        supersedes: tuple[Ref, ...],
        referent: Ref,
        *,
        reason: str,
        context: Record | None = None,
        source: str = "explicit meaning correction",
    ) -> Ref:
        if (
            not supersedes
            or type(reason) is not str
            or not reason.strip()
            or any(not isinstance(ref, Ref) for ref in supersedes)
        ):
            raise ValidationError("Correction requires exact prior bindings and an explanation")
        symbol = self.find(expression)
        if symbol is None:
            raise ValidationError("Cannot correct an unknown symbol")
        scope = record() if context is None else as_record(context)
        for ref in supersedes:
            old = self.knowledge.get(ref)
            if (
                old.fields.get("kind") != "meaning_binding"
                or old.fields.get("symbol") != symbol
                or not exact_equal(old.fields.get("context"), scope)
            ):
                raise ValidationError(
                    "Correction must identify bindings of this expression and scope"
                )
        replacement = self.bind(
            expression, referent, context=scope, evidence=Evidence(source, note=reason)
        )
        return self.knowledge.create(
            fields={
                "kind": "meaning_correction",
                "symbol": symbol,
                "supersedes": supersedes,
                "replacement": replacement,
                "reason": as_string(reason),
            },
            source=source,
        )
