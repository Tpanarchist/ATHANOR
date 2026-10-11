"""Validated teaching and attributed observations, independently usable without PyTorch."""

from collections.abc import Mapping
from dataclasses import dataclass

from athanor import (
    Athanor,
    Evidence,
    FieldSpec,
    Procedure,
    Record,
    Ref,
    ValidationError,
    record,
)
from athanor.values import exact_equal

from .lexicon import Lexicon


@dataclass(frozen=True, slots=True)
class Example:
    arguments: tuple[Ref, ...]
    expected: Record


class Teacher:
    def __init__(self, knowledge: Athanor) -> None:
        self.knowledge = knowledge
        self.lexicon = Lexicon(knowledge)

    def define(
        self,
        name: str,
        definition: str,
        procedure: Procedure,
        *,
        expressions: tuple[str, ...],
        examples: tuple[Example, ...] = (),
        roles: tuple[str, ...] = (),
        context: Record | None = None,
        general: tuple[Ref, ...] = (),
        source: str = "structured instructor",
    ) -> Ref:
        if (
            type(definition) is not str
            or not definition.strip()
            or not expressions
            or roles
            and len(roles) != len(procedure.inputs)
            or any(type(role) is not str or not role.strip() for role in roles)
        ):
            raise ValidationError(
                "A concept requires a definition, expressions and consistent roles"
            )

        def install(a: Athanor) -> Ref:
            operation = a.teach_operation(name, procedure, source=source)
            demonstrated: list[Ref] = []
            for example in examples:
                result = a.execute(operation, example.arguments)
                if not exact_equal(a.get(result).fields, example.expected):
                    raise ValidationError("Taught procedure fails its independent exact example")
                demonstrated.append(
                    a.create(
                        fields={
                            "kind": "teaching_example",
                            "operation": operation,
                            "arguments": example.arguments,
                            "result": result,
                            "expected": example.expected,
                        },
                        source=source,
                    )
                )
            for parent in general:
                if not isinstance(parent, Ref):
                    raise ValidationError("General concepts require exact references")
                a.get(parent)
            concept = a.create(
                fields={
                    "kind": "concept",
                    "definition": definition,
                    "operation": operation,
                    "roles": roles,
                    "general": general,
                    "examples": tuple(demonstrated),
                },
                source=source,
            )
            lexicon = Lexicon(a)
            for expression in expressions:
                lexicon.bind(expression, concept, context=context, evidence=Evidence(source))
            a.create(
                fields={
                    "kind": "capability_installation",
                    "concept": concept,
                    "operation": operation,
                    "installed": True,
                    "recognition_demonstrated": False,
                    "execution_examples": tuple(demonstrated),
                    "observer_success_demonstrated": False,
                },
                source=source,
            )
            return concept

        return self.knowledge.transaction(install)

    def specialize(
        self,
        name: str,
        parent: Ref,
        fields: Mapping[str, FieldSpec] | None = None,
        *,
        expressions: tuple[str, ...],
        source: str = "structured specialization",
    ) -> Ref:
        def install(a: Athanor) -> Ref:
            category = a.teach_category(name, fields, parents=(parent,), source=source)
            lexicon = Lexicon(a)
            for expression in expressions:
                lexicon.bind(expression, category, evidence=Evidence(source))
            return category

        return self.knowledge.transaction(install)

    def alias(
        self,
        expression: str,
        referent: Ref,
        *,
        context: Record | None = None,
        evidence: Evidence | None = None,
    ) -> Ref:
        return self.knowledge.transaction(
            lambda a: Lexicon(a).bind(expression, referent, context=context, evidence=evidence)
        )

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
        return self.knowledge.transaction(
            lambda a: Lexicon(a).correct(
                expression, supersedes, referent, reason=reason, context=context, source=source
            )
        )

    def observe(
        self,
        category: Ref,
        measurements: Record,
        *,
        expression: str,
        source: str,
        context: Record | None = None,
    ) -> tuple[Ref, Ref]:
        if not source.strip():
            raise ValidationError("Observations require an attributed independent source")

        def install(a: Athanor) -> tuple[Ref, Ref]:
            participant = a.create(category, measurements, source=f"observation:{source}")
            witness = a.create(
                fields={
                    "kind": "grounding_observation",
                    "participant": participant,
                    "measurements": measurements,
                    "context": record() if context is None else context,
                    "provider": source,
                },
                source=f"observation:{source}",
            )
            Lexicon(a).bind(
                expression,
                participant,
                context=context,
                evidence=Evidence(
                    f"observation:{source}",
                    note="named participant grounded in this attributed record",
                ),
            )
            return participant, witness

        return self.knowledge.transaction(install)

    def claim(
        self,
        relation: Ref,
        subject: Ref,
        target: Ref,
        *,
        source: str,
        stance: str = "positive",
        observation: Ref | None = None,
    ) -> Ref:
        # An assertion cannot mint its own observation support.
        if observation is not None:
            node = self.knowledge.get(observation)
            receipt = self.knowledge.inspect(observation).receipt
            if (
                node.fields.get("kind") != "relation_observation"
                or node.fields.get("relation") != relation
                or node.fields.get("subject") != subject
                or node.fields.get("target") != target
                or node.fields.get("stance") != stance
                or not str(receipt.details.get("source", "")).startswith("observation:")
            ):
                raise ValidationError("Claim support requires a matching independent observation")
            evidence = Evidence(
                source, stance, note=f"supported by attributed observation {observation}"
            )
        else:
            evidence = Evidence(
                source,
                "uncertain",
                note=f"unverified empirical assertion; proposed stance={stance}",
            )
        if stance not in {"positive", "negative", "uncertain"}:
            raise ValidationError("Invalid empirical assertion stance")

        def install(a: Athanor) -> Ref:
            claim = a.bind(relation, subject, target, evidence)
            if observation is not None:
                a.create(
                    fields={"kind": "claim_support", "claim": claim, "observation": observation},
                    source=source,
                )
            return claim

        return self.knowledge.transaction(install)

    def observe_relation(
        self, relation: Ref, subject: Ref, target: Ref, *, source: str, stance: str = "positive"
    ) -> Ref:
        def install(a: Athanor) -> Ref:
            # Validate the endpoints/evidence before recording an observation witness.
            a.bind(
                relation,
                subject,
                target,
                Evidence(
                    f"observation:{source}",
                    "uncertain",
                    note="observation scope validated; support recorded separately",
                ),
            )
            return a.create(
                fields={
                    "kind": "relation_observation",
                    "relation": relation,
                    "subject": subject,
                    "target": target,
                    "stance": stance,
                    "provider": source,
                },
                source=f"observation:{source}",
            )

        if not source.strip() or stance not in {"positive", "negative", "uncertain"}:
            raise ValidationError("An observation requires a source and valid stance")
        return self.knowledge.transaction(install)
