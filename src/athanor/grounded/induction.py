"""Native operational concepts, structural examples and bounded objective inquiry.

This is the structural front end of the shared process and recurrent learning machinery.
Only observations, examples and declared objectives cross its boundary.
"""

import itertools
import math
from collections.abc import Iterable
from fractions import Fraction
from typing import TYPE_CHECKING, Any, cast

import torch

from athanor import (
    Athanor,
    AthanorError,
    Endpoint,
    FieldSpec,
    Limits,
    Procedure,
    Record,
    Ref,
    Role,
    Structure,
    ValidationError,
    Value,
    record,
)
from athanor.neural import Observation
from athanor.neural.encoding import structural_features
from athanor.programs import (
    argument,
    binary,
    call,
    choose,
    field,
    has_field,
    literal,
    structural,
    variable,
)
from athanor.structural import collection, family, validate_value
from athanor.values import as_record, as_ref, as_tuple, exact_equal, references
from athanor.world.engine import exact

from .discovery import priority_vectors
from .processes import Candidate

if TYPE_CHECKING:
    from .runtime import GroundedObserver
    from .training import GroundedTrainer


def discrepancy(left: Value, right: Value) -> float:
    """Identity-independent exact structural discrepancy; zero iff exact equality."""
    if exact_equal(left, right):
        return 0.0
    if type(left) is not type(right):
        return 1.0
    if isinstance(left, Record) and isinstance(right, Record):
        keys = set(left) | set(right)
        return sum(
            discrepancy(left[k], right[k]) if k in left and k in right else 1 for k in keys
        ) / max(1, len(keys))
    if isinstance(left, tuple) and isinstance(right, tuple):
        n = max(len(left), len(right))
        return (
            abs(len(left) - len(right))
            + sum(discrepancy(a, b) for a, b in zip(left, right, strict=False))
        ) / max(1, n)
    return 1.0


def role_property(name: str, role: str = "item") -> Record:
    v = variable(role)
    return choose(has_field(v, name), field(v, name), literal(None))


class Induction:
    def __init__(self, observer: GroundedObserver) -> None:
        self.observer = observer
        self.category: Ref | None = None
        self.library: list[Ref] = []
        self.concepts: list[Ref] = []
        self.rules: list[Ref] = []
        self.candidates: list[Candidate] = []
        self.examples: list[Ref] = []
        self.observations: list[Ref] = []
        self.history: list[Ref] = []
        self.tasks: list[Ref] = []
        self.revisers: list[Ref] = []
        self.revoked: list[Ref] = []
        self.active: tuple[Ref, ...] = ()
        self.selected: Ref | None = None
        self.query: Ref | None = None
        self.status = "idle"
        self.neural_updates = 0
        self.capabilities: dict[str, dict[str, int]] = {}
        self.resource: dict[str, int] = {"max_calls": 2, "max_candidates": 384, "beam": 32}
        self._input_cache: dict[str, Ref] = {}

    @property
    def knowledge(self) -> Athanor:
        return self.observer.knowledge

    def cap(self, ref: Ref, name: str, success: bool) -> None:
        counts = self.capabilities.setdefault(str(ref), {})
        counts[name + "_attempts"] = counts.get(name + "_attempts", 0) + 1
        counts[name + "_successes"] = counts.get(name + "_successes", 0) + int(success)
        receipt = self.knowledge.create(
            fields=record(
                kind="structural_capability",
                participant=ref,
                ability=name,
                success=success,
                counts=counts,
            ),
            source="measured structural outcome",
        )
        self.history.append(receipt)
        self.observer.observer.self_map.append(receipt)
        self.observer.observer.self_map = self.observer.observer.self_map[
            -self.observer.observer.model.config.history_limit :
        ]

    def install(self) -> None:
        if self.category is not None:
            return

        def work(k: Athanor) -> tuple[Ref, list[Ref]]:
            category = k.teach_category("structural value interface", fields={"value": FieldSpec()})
            operations = []
            value = field(argument(0), "value")
            for name in (
                "reverse",
                "unique",
                "transpose",
                "reflect_x",
                "reflect_y",
                "raster_graph",
                "rasterize",
                "components",
                "nodes",
                "count",
            ):
                operations.append(
                    k.teach_operation(
                        name,
                        Procedure(
                            (Endpoint(Role.DATA, category),),
                            category,
                            {"value": structural(name, value)},
                        ),
                        source="exact generic structural algorithm",
                    )
                )
            for name in ("shift_x", "shift_y"):
                for amount in (-1, 1):
                    operations.append(
                        k.teach_operation(
                            name,
                            Procedure(
                                (Endpoint(Role.DATA, category),),
                                category,
                                {"value": structural(name, value, literal(amount))},
                            ),
                            source="bounded geometric translation",
                        )
                    )
            return category, operations

        self.category, self.library = self.knowledge.transaction(work)

    def operation(self, expression: Record, *, source: str) -> Ref:
        self.install()
        return self.knowledge.teach_operation(
            "structural constituent",
            Procedure((Endpoint(Role.DATA, self.category),), self.category, {"value": expression}),
            source=source,
        )

    def evaluate(self, operation: Ref, value: Value) -> tuple[Value, Ref]:
        self.install()
        validate_value(value)
        key = repr(value)
        datum = self._input_cache.get(key)
        if datum is None:
            datum = self.knowledge.create(
                self.category, {"value": value}, source="structural observation binding"
            )
            if len(self._input_cache) >= 512:
                del self._input_cache[next(iter(self._input_cache))]
            self._input_cache[key] = datum
        result = self.knowledge.execute(
            operation, (datum,), limits=Limits(max_steps=20000, max_depth=24)
        )
        output = self.knowledge.get(result).fields["value"]
        validate_value(output)
        return output, result

    def activate(self, value: Value, *, source: str) -> Ref:
        validate_value(value)
        self.query = self.knowledge.create(
            fields=record(
                kind="structural_request",
                value=value,
                hypotheses=self.active[:4],
                objective=self.tasks[-1] if self.tasks else None,
            ),
            source=source,
        )
        relevant = []
        for hypothesis in self.active[:4]:
            operation = self.knowledge.get(hypothesis).fields.get("operation")
            if isinstance(operation, Ref):
                relevant.append(operation)
        self.observer.observer.active_operations = tuple(
            dict.fromkeys(relevant or self.available()[:8])
        )
        self.observer.observer.observe(Observation(self.query, ()))
        return self.query

    def revise(self, ref: Ref, values: Record) -> Ref:
        old = self.knowledge.get(ref).fields
        if set(old) != set(values):
            raise ValidationError("A native revision must preserve its structural contract")
        operation = next(
            (
                r
                for r in self.revisers
                if set(as_record(as_record(self.knowledge.get(r).fields["procedure"])["fields"]))
                == set(values)
            ),
            None,
        )
        if operation is None:
            operation = self.knowledge.teach_operation(
                "revise structural evidence",
                Procedure(
                    (Endpoint(Role.DATA),) * 2,
                    None,
                    {k: field(argument(1), k) for k in values},
                    update=0,
                ),
            )
            self.revisers.append(operation)
        patch = self.knowledge.create(fields=values, source="structural revision evidence")
        return self.knowledge.execute(operation, (ref, patch))

    def publish(self, candidate: Candidate) -> None:
        old = self.knowledge.get(candidate.hypothesis).fields
        candidate.hypothesis = self.revise(
            candidate.hypothesis,
            record(
                dict(old),
                status=candidate.status,
                errors=tuple(exact(x) for x in candidate.errors),
                proposal_errors=tuple(exact(x) for x in candidate.proposal_errors),
                support=tuple(self.examples),
                results=tuple(candidate.results[-3:]),
            ),
        )

    def vectors(self, candidates: list[Candidate], *, proposal: bool = False) -> torch.Tensor:
        model = self.observer.observer.model
        options = [
            record(
                uncertainty=exact(1 / (1 + len(c.errors))),
                relevance=1,
                benefit=exact(
                    1
                    - min(
                        1,
                        max(
                            c.proposal_errors if proposal and c.proposal_errors else c.errors,
                            default=1,
                        ),
                    )
                ),
                reliability=exact(sum(x == 0 for x in c.errors) / max(1, len(c.errors))),
                cost=exact(c.calls / 8),
                risk=exact(
                    min(
                        1,
                        max(
                            c.proposal_errors if proposal and c.proposal_errors else c.errors,
                            default=1,
                        ),
                    )
                ),
            )
            for c in candidates
        ]
        base = priority_vectors(
            model, options, kind="structural_proposal" if proposal else "structural_priority"
        )
        descriptions = torch.tensor(
            [
                structural_features(
                    Structure(
                        Ref("s00000001"),
                        Role.DATA,
                        None,
                        record(kind="structural_explanation", expression=c.expression),
                        (),
                    )
                )
                for c in candidates
            ],
            device=model.device,
        )
        return cast(torch.Tensor, base + model.encoder(descriptions).tanh() * 0.03)

    def rank(self, candidates: list[Candidate], *, proposal: bool = False) -> list[Candidate]:
        if not candidates:
            return []
        with torch.no_grad():
            scores = self.observer.observer.model._scores(
                self.vectors(candidates, proposal=proposal),
                self.observer.observer.state.context.detach() * 0.05,
            )
        for c, score in zip(candidates, scores, strict=True):
            c.score = float(score)
        return sorted(candidates, key=lambda c: c.score, reverse=True)

    def learn(
        self,
        trainer: GroundedTrainer,
        candidates: list[Candidate],
        *,
        steps: int = 80,
        proposal: bool = False,
    ) -> None:
        if not self.observer.discovery.learning or not candidates:
            return
        with torch.no_grad():
            vectors = self.vectors(candidates, proposal=proposal).detach()
            targets = torch.tensor(
                [
                    1
                    - max(
                        c.proposal_errors if proposal and c.proposal_errors else c.errors, default=1
                    )
                    - c.calls * 0.01
                    for c in candidates
                ],
                device=vectors.device,
            )
            context = self.observer.observer.state.context.detach() * 0.05
        prior = trainer.updates
        self.observer.processes.train_priority(trainer, vectors, targets, context, steps=steps)
        self.neural_updates += trainer.updates - prior

    def form_concepts(
        self, values: tuple[Value, ...], *, source: str = "observed structural recurrence"
    ) -> list[Ref]:
        self.install()
        if not 1 <= len(values) <= 32:
            raise ValidationError("Concept inquiry requires 1--32 observations")
        for value in values:
            validate_value(value)
        if all(family(v) == "raster" for v in values):
            adapter = next(
                v for v in self.library if self.knowledge.get(v).fields["name"] == "raster_graph"
            )
            adapted = []
            for value in values:
                raw_observation = self.knowledge.create(
                    fields=record(kind="structural_observation", value=value), source=source
                )
                output, result = self.evaluate(adapter, value)
                adapted.append(output)
                self.history.append(
                    self.knowledge.create(
                        fields=record(
                            kind="structural_representation",
                            observation=raw_observation,
                            adapter=adapter,
                            result=result,
                        ),
                        source="exact generic raster adjacency; grouping relevance remains learned",
                    )
                )
            return self.form_concepts(
                tuple(adapted), source="derived raster relational observation"
            )
        self.observations.extend(
            self.knowledge.create(
                fields=record(kind="structural_observation", value=v), source=source
            )
            for v in values
        )
        self.activate(values[-1], source=source)
        predicates: list[Record] = []
        if all(family(v) == "graph" for v in values):
            for name in ("degree", "in_degree", "out_degree"):
                degree = structural(name, variable("root"), variable("index"))
                predicates += [
                    binary("lt", literal(0), degree),
                    binary("eq", degree, literal(0)),
                    binary("lt", literal(1), degree),
                ]
        items = (
            [v for root in values for v in collection(root)]
            if all(family(v) in {"graph", "sequence"} for v in values)
            else []
        )
        names = sorted({name for raw in items if isinstance(raw, Record) for name in raw})[:16]
        for name in names:
            observed: list[Value] = []
            for raw in items:
                if (
                    isinstance(raw, Record)
                    and name in raw
                    and not isinstance(raw[name], (Record, tuple, Ref))
                ):
                    if not any(exact_equal(raw[name], v) for v in observed):
                        observed.append(raw[name])
            for v in observed[:8]:
                prop = role_property(name)
                eq = binary("eq", prop, literal(v))
                guard = has_field(variable("item"), name)
                predicates.extend(
                    [
                        choose(guard, eq, literal(None)),
                        choose(guard, structural("not", eq), literal(None)),
                    ]
                )
            own, peer = role_property(name), role_property(name, "peer")
            guard = structural(
                "and", has_field(variable("item"), name), has_field(variable("peer"), name)
            )
            for operator in ("eq", "lt"):
                if operator == "lt" and not all(isinstance(v, Fraction) for v in observed):
                    continue
                predicates.append(
                    structural(
                        "exists",
                        variable("root"),
                        choose(guard, binary(operator, own, peer), literal(None)),
                    )
                )
        proposed: list[Candidate] = []
        if all(family(v) == "graph" for v in values):
            expression = structural("components", field(argument(0), "value"))
            if not any(
                self.knowledge.get(c).fields["predicate"] == expression for c in self.concepts
            ):
                operation = next(
                    v for v in self.library if self.knowledge.get(v).fields["name"] == "components"
                )
                groups = [self.evaluate(operation, v) for v in values]
                grouping_support = tuple(
                    w
                    for (output, _), w in zip(
                        groups, self.observations[-len(values) :], strict=True
                    )
                    if isinstance(output, tuple)
                    and len(output) > 1
                    and any(isinstance(g, tuple) and len(g) > 1 for g in output)
                )
                if grouping_support:
                    concept = self.knowledge.create(
                        fields=record(
                            kind="formed_concept",
                            operation=operation,
                            predicate=expression,
                            roles=(
                                record(name="root", type="graph"),
                                record(name="group", type="collection_of_bound_indices"),
                            ),
                            status="supported"
                            if self.distinct_support(grouping_support) >= 3
                            else "provisional",
                            support=grouping_support,
                            contradictions=(),
                            dependencies=(),
                            results=tuple(r for _, r in groups),
                            conditions=(
                                "observed connectivity partition; directed relationships "
                                "retained in the source"
                            ),
                        ),
                        source="unnamed recurring relational grouping",
                    )
                    self.concepts.append(concept)
                    proposed.append(
                        Candidate(
                            operation,
                            concept,
                            expression,
                            (),
                            1,
                            [1 - len(grouping_support) / len(values)],
                        )
                    )
                    self.cap(
                        concept, "concept_formation", self.distinct_support(grouping_support) >= 3
                    )
        for predicate in predicates[:96]:
            if any(self.knowledge.get(c).fields["predicate"] == predicate for c in self.concepts):
                continue
            outputs: list[Ref] = []
            support: list[Ref] = []
            for value, witness in zip(values, self.observations[-len(values) :], strict=True):
                # Evaluation uses an ordinary native procedure, not a predicate shortcut.
                if not outputs:
                    operation = self.operation(
                        structural("select", field(argument(0), "value"), predicate),
                        source="unnamed operational distinction",
                    )
                try:
                    output, receipt = self.evaluate(operation, value)
                    if output is not None and 0 < len(collection(output)) < len(collection(value)):
                        support.append(witness)
                    outputs.append(receipt)
                except AthanorError:
                    outputs.append(witness)
            if not support:
                continue
            concept = self.knowledge.create(
                fields=record(
                    kind="formed_concept",
                    operation=operation,
                    predicate=predicate,
                    roles=tuple(
                        record(name=n, type=t)
                        for n, t in (
                            ("root", "collection"),
                            ("item", "record"),
                            ("index", "nonnegative_integer"),
                            ("peer", "different_record_role"),
                        )
                    ),
                    status="supported" if self.distinct_support(support) >= 3 else "provisional",
                    support=tuple(support),
                    contradictions=(),
                    dependencies=(),
                    results=tuple(outputs),
                    conditions="typed structural collection; unavailable properties remain unknown",
                ),
                source=source,
            )
            self.concepts.append(concept)
            self.library.append(operation)
            c = Candidate(operation, concept, predicate, (), 1, [1 - len(support) / len(values)])
            proposed.append(c)
            self.cap(concept, "concept_formation", self.distinct_support(support) >= 3)
        for name in names:
            expression = structural("sort", field(argument(0), "value"), role_property(name))
            if any(
                as_record(as_record(self.knowledge.get(op).fields["procedure"])["fields"])["value"]
                == expression
                for op in self.library
            ):
                continue
            operation = self.operation(
                expression,
                source="observed structural key ordering",
            )
            self.library.append(operation)
        ranked = self.rank(proposed)
        self.active = tuple(c.hypothesis for c in ranked[:4])
        self.status = "concepts_formed" if ranked else "unknown"
        self.activate(values[-1], source="concept inquiry with current native evidence")
        return [c.hypothesis for c in ranked]

    def distinct_support(self, support: Iterable[Ref]) -> int:
        return len({repr(self.knowledge.get(v).fields["value"]) for v in support})

    def recognize(self, concept: Ref, value: Value) -> Value:
        if concept not in self.concepts:
            raise ValidationError("Unknown acquired concept")
        c = self.knowledge.get(concept).fields
        if c["status"] == "qualified":
            self.cap(concept, "recognition", False)
            self.activate(value, source="qualified operational concept query")
            return None
        try:
            output, receipt = self.evaluate(as_ref(c["operation"]), value)
        except AthanorError:
            output, receipt = None, None
        self.cap(concept, "recognition", output is not None)
        self.cap(concept, "relationship_application", output is not None)
        self.history.append(
            self.knowledge.create(
                fields=record(
                    kind="concept_application",
                    concept=concept,
                    value=value,
                    output=output,
                    result=receipt,
                    status="unknown" if output is None else "observed_partition",
                ),
                source="native operational membership",
            )
        )
        self.activate(value, source="concept query with measured capability feedback")
        return output

    def combine_concepts(self, first: Ref, second: Ref, observations: tuple[Value, ...]) -> Ref:
        if not 1 <= len(observations) <= 32:
            raise ValidationError("Concept composition requires 1--32 observations")
        if first not in self.concepts or second not in self.concepts:
            raise ValidationError("Concept composition needs acquired constituents")
        if any(self.knowledge.get(c).fields["status"] == "qualified" for c in (first, second)):
            raise ValidationError("Qualified concepts cannot authorize new abstractions")
        self.install()
        assert self.category is not None
        chain = (
            as_ref(self.knowledge.get(first).fields["operation"]),
            as_ref(self.knowledge.get(second).fields["operation"]),
        )
        candidate = self.observer.processes.structural_call(self.category, chain)
        results, support = [], []
        for value in observations:
            output, result = self.evaluate(candidate.operation, value)
            results.append(result)
            if output is not None:
                support.append(
                    self.knowledge.create(
                        fields=record(value=value), source="composite concept observation"
                    )
                )
        c = self.knowledge.create(
            fields=record(
                kind="formed_concept",
                operation=candidate.operation,
                predicate=candidate.expression,
                roles=tuple(
                    record(name=n, type=t)
                    for n, t in (
                        ("root", "collection"),
                        ("item", "record"),
                        ("index", "nonnegative_integer"),
                        ("peer", "different_record_role"),
                    )
                ),
                status="supported" if self.distinct_support(support) >= 3 else "provisional",
                support=tuple(support),
                contradictions=(),
                dependencies=(first, second),
                results=tuple(results),
                conditions=(
                    "sequential operational concepts; role relationships "
                    "re-evaluate in the intermediate structure"
                ),
            ),
            source="recursive native concept formation",
        )
        self.concepts.append(c)
        self.library.append(candidate.operation)
        self.active = (c, *self.active[:3])
        if observations:
            self.activate(observations[-1], source="composed operational concept inquiry")
        return c

    def _assess(self, candidate: Candidate, pairs: tuple[tuple[Value, Value], ...]) -> bool:
        valid = False
        for before, after in pairs:
            try:
                output, receipt = self.evaluate(candidate.operation, before)
                candidate.results.append(receipt)
                error = 1.0 if output is None or after is None else discrepancy(output, after)
                candidate.errors.append(error)
                proposal_error = error
                # Generic typed representation bridges give partial proposals a
                # measured relevance signal; they never turn a wrong final output
                # into supporting evidence. Every bridge executes natively.
                if family(output) == "graph" and family(after) == "raster":
                    bridge = next(
                        op
                        for op in self.library
                        if self.knowledge.get(op).fields["name"] == "rasterize"
                    )
                    try:
                        projected, _ = self.evaluate(bridge, output)
                        proposal_error = min(error, discrepancy(projected, after))
                    except AthanorError:
                        pass
                candidate.proposal_errors.append(proposal_error)
                valid |= output is not None
            except AthanorError:
                candidate.errors.append(1.0)
                candidate.proposal_errors.append(1.0)
        candidate.status = (
            "unknown"
            if any(after is None for _, after in pairs)
            else "qualified"
            if max(candidate.errors, default=1) > 0
            else "supported"
            if valid
            and len(pairs) >= 3
            and len({repr(a) for a, _ in pairs}) >= 3
            and all(a is not None and b is not None for a, b in pairs)
            else "hypothesized"
        )
        self.publish(candidate)
        return valid

    def infer(
        self,
        pairs: tuple[tuple[Value, Value], ...],
        trainer: GroundedTrainer,
        *,
        max_calls: int = 2,
        max_candidates: int = 384,
        beam: int = 32,
    ) -> Ref | None:
        if (
            type(max_calls) is not int
            or not 1 <= max_calls <= 4
            or type(max_candidates) is not int
            or not 1 <= max_candidates <= 2048
            or type(beam) is not int
            or not 1 <= beam <= 128
        ):
            raise ValidationError("Structural synthesis exceeds bounded resources")
        if not 1 <= len(pairs) <= 32:
            raise ValidationError("Transformation inference requires 1--32 examples")
        for before, after in pairs:
            validate_value(before)
            validate_value(after)
        self.install()
        assert self.category is not None
        self.resource = {"max_calls": max_calls, "max_candidates": max_candidates, "beam": beam}
        if all(family(v) in {"graph", "sequence", "raster"} for v, _ in pairs):
            self.form_concepts(
                tuple(v for v, _ in pairs),
                source="self-directed distinctions from demonstration inputs",
            )
        self.examples = [
            self.knowledge.create(
                fields=record(kind="structural_example", before=a, after=b),
                source="external structural demonstration",
            )
            for a, b in pairs
        ]
        self.candidates = []
        self.selected = None
        self.activate(pairs[0][0], source="unfamiliar structural transformation examples")
        library = self.available()
        prefixes: list[tuple[Ref, ...]] = [()]
        for _ in range(max_calls):
            level = []
            for prefix, operation in itertools.product(prefixes, library):
                if len(self.candidates) >= max_candidates:
                    break
                c = self.observer.processes.structural_call(self.category, (*prefix, operation))
                self.candidates.append(c)
                self.cap(c.operation, "rule_construction", True)
                if self._assess(c, pairs):
                    level.append(c)
            if not level or len(self.candidates) >= max_candidates:
                break
            # Measured partial discrepancies train the same scorer, then its order
            # controls which call prefixes receive the remaining construction budget.
            self.learn(trainer, level, steps=40, proposal=True)
            ranked = self.rank(level, proposal=True)
            prefixes = [c.dependencies for c in ranked[:beam]]
            if any(c.status == "supported" for c in level):
                break
        self.learn(trainer, self.candidates, steps=120)
        ranked = self.rank(self.candidates)
        self.active = tuple(c.hypothesis for c in ranked[:4])
        if not ranked or ranked[0].status != "supported":
            self.status = "unknown"
            self.activate(pairs[-1][0], source="unresolved native structural explanations")
            return None
        chosen = ranked[0]
        self.selected = chosen.operation
        alternatives = tuple(c.operation for c in self.candidates if c.status == "supported")
        rule = self.knowledge.create(
            fields=record(
                kind="acquired_structural_rule",
                operation=chosen.operation,
                hypothesis=chosen.hypothesis,
                dependencies=chosen.dependencies,
                alternatives=alternatives,
                examples=tuple(self.examples),
                status="supported",
                contradictions=(),
                outcomes=(),
                input_family=family(pairs[0][0]),
                schema=self.signature(pairs[0][0]),
                goal=None,
            ),
            source="neural-selected evidence-qualified transformation",
        )
        self.rules.append(rule)
        self.library.append(chosen.operation)
        self.status = "supported"
        self.cap(rule, "rule_induction", True)
        self.activate(pairs[-1][0], source="acquired structural rule with current evidence")
        return rule

    def available(self) -> list[Ref]:
        invalid = set(self.revoked) | {
            as_ref(self.knowledge.get(c).fields["operation"])
            for c in [*self.concepts, *self.rules]
            if self.knowledge.get(c).fields["status"] == "qualified"
        }
        return list(dict.fromkeys(v for v in self.library if v not in invalid))

    @staticmethod
    def signature(value: Value) -> Value:
        f = family(value)
        if f == "graph":
            nodes = collection(value)
            schemas = {tuple((k, type(v).__name__) for k, v in as_record(n).items()) for n in nodes}
            return record(family=f, node_schemas=tuple(sorted(schemas)))
        if f == "sequence":
            return record(
                family=f, item_types=tuple(sorted({type(v).__name__ for v in as_tuple(value)}))
            )
        return record(family=f)

    def apply(
        self, rule: Ref, value: Value, *, expected: Value = None, measured: bool = False
    ) -> Value:
        if rule not in self.rules:
            raise ValidationError("Unknown acquired structural rule")
        validate_value(value)
        self.active = (
            rule,
            *tuple(v for v in self.active if (v.namespace, v.id) != (rule.namespace, rule.id))[:3],
        )
        self.activate(value, source="structural rule query")
        spec = self.knowledge.get(rule).fields
        if spec["status"] == "qualified" or not exact_equal(spec["schema"], self.signature(value)):
            self.cap(rule, "execution", False)
            self.activate(value, source="out-of-scope structural application")
            return None
        predictions = []
        results = []
        for operation in as_tuple(spec["alternatives"]):
            try:
                prediction, receipt = self.evaluate(as_ref(operation), value)
                predictions.append(prediction)
                results.append(receipt)
            except AthanorError:
                predictions.append(None)
        survivors = tuple(
            op
            for op, predicted in zip(as_tuple(spec["alternatives"]), predictions, strict=True)
            if measured and exact_equal(predicted, expected) and predicted is not None
        )
        if measured and survivors and len(survivors) < len(predictions):
            refinement = self.knowledge.create(
                fields=record(
                    kind="structural_contradiction",
                    rule=rule,
                    input=value,
                    expected=expected,
                    alternatives=spec["alternatives"],
                    predictions=tuple(predictions),
                    retained=survivors,
                ),
                source="new measured consequence distinguishes competing rules",
            )
            new = self.revise(
                rule,
                record(
                    dict(spec),
                    alternatives=survivors,
                    contradictions=(*as_tuple(spec["contradictions"]), refinement),
                ),
            )
            self.rules[self.rules.index(rule)] = new
            rule, spec = new, self.knowledge.get(new).fields
            if spec["operation"] not in survivors:
                supported_candidates = []
                for op in survivors:
                    operation = as_ref(op)
                    hypothesis = next(
                        (c.hypothesis for c in self.candidates if c.operation == operation), None
                    )
                    if hypothesis is None:
                        self.observer.observer.builder.index.sync(self.knowledge)
                        hypothesis = next(
                            h
                            for h in reversed(
                                self.observer.observer.builder.index.kinds["structural_hypothesis"]
                            )
                            if self.knowledge.get(h).fields["operation"] == operation
                        )
                    native = self.knowledge.get(hypothesis).fields
                    supported_candidates.append(
                        Candidate(
                            operation,
                            hypothesis,
                            as_record(native["expression"]),
                            tuple(as_ref(v) for v in as_tuple(native["dependencies"])),
                            int(cast(Fraction, native["calls"])),
                            [0],
                        )
                    )
                chosen = self.rank(supported_candidates)[0]
                new = self.revise(
                    rule,
                    record(
                        dict(spec),
                        operation=chosen.operation,
                        hypothesis=chosen.hypothesis,
                        dependencies=chosen.dependencies,
                    ),
                )
                self.rules[self.rules.index(rule)] = new
                rule, spec = new, self.knowledge.get(new).fields
                if chosen.operation not in self.library:
                    self.library.append(chosen.operation)
            discarded = tuple(
                op
                for op in as_tuple(self.knowledge.get(refinement).fields["alternatives"])
                if op not in survivors
            )
            for op in discarded:
                self.revoke(as_ref(op), refinement)
            predictions = [
                predicted for predicted in predictions if exact_equal(predicted, expected)
            ]
        agreed = (
            bool(predictions)
            and predictions[0] is not None
            and all(exact_equal(v, predictions[0]) for v in predictions)
        )
        output = predictions[0] if agreed else None
        success = agreed and (not measured or exact_equal(output, expected))
        goal_satisfied = (
            None
            if spec["goal"] is None
            else self.goal_test(as_ref(spec["goal"]), output, initial=value)
        )
        if spec["goal"] is not None:
            success = success and goal_satisfied is True
            self.cap(rule, "goal_achievement", goal_satisfied is True)
        evidence = self.knowledge.create(
            fields=record(
                kind="structural_application",
                rule=rule,
                input=value,
                predictions=tuple(predictions),
                expected=expected if measured else None,
                measured=measured,
                output=output,
                results=tuple(results),
                success=success,
                uncertain=not agreed,
                goal_satisfied=goal_satisfied,
            ),
            source="native held-out structural execution",
        )
        self.history.append(evidence)
        self.cap(rule, "execution", success)
        self.cap(rule, "role_binding", agreed)
        if goal_satisfied is False and not measured:
            self.qualify(rule, evidence)
        if measured:
            self.cap(rule, "transfer", success)
            if not success:
                self.qualify(rule, evidence)
            else:
                outcomes = (*as_tuple(spec["outcomes"]), evidence)
                generalized = (
                    len({repr(self.knowledge.get(as_ref(e)).fields["input"]) for e in outcomes})
                    >= 3
                )
                new = self.revise(
                    rule,
                    record(
                        dict(spec),
                        outcomes=outcomes,
                        status="supported"
                        if generalized or spec["status"] == "supported"
                        else "provisional",
                    ),
                )
                self.rules[self.rules.index(rule)] = new
                self.cap(new, "generalization", generalized)
                rule = new
        if rule in self.rules:
            self.active = (
                rule,
                *tuple(v for v in self.active if (v.namespace, v.id) != (rule.namespace, rule.id))[
                    :3
                ],
            )
        self.activate(value, source="structural execution and current native feedback")
        return output

    def qualify(self, ref: Ref, evidence: Ref) -> None:
        if ref not in [*self.concepts, *self.rules]:
            raise ValidationError("Qualification requires acquired structural knowledge")
        invalid_operations: set[Ref] = set()
        invalid_refs: set[Ref] = {ref}
        changed = True
        while changed:
            changed = False
            for pool in (self.concepts, self.rules):
                for index, current in enumerate(pool):
                    node = self.knowledge.get(current).fields
                    deps = set(references(node.get("dependencies")))
                    if isinstance(node.get("goal"), Ref):
                        deps.update(references(self.knowledge.get(as_ref(node["goal"])).fields))
                    if node["status"] == "qualified":
                        continue
                    if current in invalid_refs or deps & (invalid_refs | invalid_operations):
                        invalid_refs.add(current)
                        invalid_operations.add(as_ref(node["operation"]))
                        revised = self.revise(
                            current,
                            record(
                                dict(node),
                                status="qualified",
                                contradictions=(*as_tuple(node["contradictions"]), evidence),
                            ),
                        )
                        pool[index] = revised
                        invalid_refs.add(revised)
                        changed = True
        self.status = "qualified"
        for operation in sorted(invalid_operations):
            if operation not in self.revoked:
                self.revoked.append(operation)
        self._qualify_candidates(invalid_operations, evidence)

    def _qualify_candidates(self, operations: set[Ref], evidence: Ref) -> None:
        for c in self.candidates:
            if c.operation in operations or set(c.dependencies) & operations:
                old = self.knowledge.get(c.hypothesis).fields
                c.hypothesis = self.revise(
                    c.hypothesis,
                    record(
                        dict(old),
                        status="qualified",
                        contradictions=(*as_tuple(old["contradictions"]), evidence),
                    ),
                )
                c.status = "qualified"
        active_ids = {(v.namespace, v.id) for v in self.active}
        self.active = tuple(
            v
            for v in [*(c.hypothesis for c in self.candidates), *self.concepts, *self.rules]
            if (v.namespace, v.id) in active_ids
        )[:4]

    def revoke(self, operation: Ref, evidence: Ref) -> None:
        self.knowledge.get(evidence)
        if operation not in self.revoked:
            self.revoked.append(operation)
        for pool in (self.concepts, self.rules):
            for prior in tuple(pool):
                ref = next(v for v in pool if v.id == prior.id and v.namespace == prior.namespace)
                spec = self.knowledge.get(ref).fields
                if spec["status"] != "qualified" and (
                    spec["operation"] == operation or operation in as_tuple(spec["dependencies"])
                ):
                    self.qualify(ref, evidence)
        self._qualify_candidates({operation}, evidence)

    def goal_expression(self, constraints: tuple[Record, ...]) -> Record:
        if not 1 <= len(constraints) <= 16:
            raise ValidationError("A structural goal needs 1--16 conditions")
        v = field(field(argument(0), "value"), "current")
        result = literal(True)
        for c in constraints:
            kind = c.get("kind")
            if kind == "equals" and set(c) == {"kind", "value"}:
                term = binary("eq", v, literal(c["value"]))
            elif kind == "count" and set(c) == {"kind", "value"}:
                if not isinstance(c["value"], Fraction) or c["value"] < 0:
                    raise ValidationError("Count objective requires a nonnegative exact number")
                term = binary("eq", structural("count", v), literal(c["value"]))
            elif kind == "ordered" and set(c) == {"kind", "field"} and isinstance(c["field"], str):
                term = binary("eq", v, structural("sort", v, role_property(c["field"])))
            elif kind in {"concept", "count_concept"} and set(c) == {"kind", "reference"}:
                concept = as_ref(c["reference"])
                if (
                    concept not in self.concepts
                    or self.knowledge.get(concept).fields["status"] == "qualified"
                ):
                    raise ValidationError("Goal concept requires available operational evidence")
                op = as_ref(self.knowledge.get(concept).fields["operation"])
                adapter = self.operation(
                    field(field(argument(0), "value"), "initial"),
                    source="explicit variable initial-state goal binding",
                )
                eligible = field(call(op, call(adapter, argument(0))), "value")
                if kind == "count_concept":
                    term = binary("eq", structural("count", v), structural("count", eligible))
                else:
                    membership = binary(
                        "lt", literal(0), structural("frequency", eligible, variable("item"))
                    )
                    term = binary(
                        "eq",
                        structural("count", structural("select", v, membership)),
                        structural("count", v),
                    )
            elif kind == "contains" and set(c) == {"kind", "value"}:
                term = binary(
                    "lt",
                    literal(0),
                    structural("frequency", v, literal(c["value"])),
                )
            else:
                raise ValidationError("Unsupported or malformed structural objective condition")
            result = structural("and", result, term)
        return result

    def goal_test(self, predicate: Ref, value: Value, *, initial: Value = None) -> bool | None:
        try:
            result, _ = self.evaluate(
                predicate, record(current=value, initial=value if initial is None else initial)
            )
            return result if type(result) is bool else None
        except AthanorError:
            return None

    def solve(
        self,
        value: Value,
        constraints: tuple[Record, ...],
        trainer: GroundedTrainer,
        *,
        max_depth: int = 3,
        max_nodes: int = 256,
        operations: tuple[Ref, ...] | None = None,
    ) -> Ref | None:
        validate_value(value)
        if (
            type(max_depth) is not int
            or not 1 <= max_depth <= 4
            or type(max_nodes) is not int
            or not 1 <= max_nodes <= 2048
        ):
            raise ValidationError("Planning requires bounded positive resources")
        self.install()
        assert self.category is not None
        available = self.available() if operations is None else list(operations)
        if not available or any(v not in self.available() for v in available):
            raise ValidationError("Planning operations must be available native constituents")
        expr = self.goal_expression(constraints)
        predicate = self.operation(expr, source="declared structural goal conditions")
        progress_expression = literal(0)
        for condition in constraints:
            progress_expression = binary(
                "add",
                progress_expression,
                choose(self.goal_expression((condition,)), literal(1), literal(0)),
            )
        progress = self.operation(
            binary("div", progress_expression, literal(len(constraints))),
            source="native goal progress",
        )
        task = self.knowledge.create(
            fields=record(
                kind="structural_task",
                initial=value,
                conditions=constraints,
                predicate=predicate,
                operations=tuple(available),
                max_depth=max_depth,
                max_nodes=max_nodes,
                status="investigating",
                selected=None,
                evaluated=0,
                outcomes=(),
            ),
            source="declared structural objective",
        )
        self.tasks.append(task)
        self.activate(value, source="goal-directed structural inquiry")
        if self.goal_test(predicate, value) is True:
            self.tasks[-1] = self.revise(
                task, record(dict(self.knowledge.get(task).fields), status="already_satisfied")
            )
            self.cap(self.tasks[-1], "goal_achievement", True)
            self.status, self.selected, self.active = "goal_supported", None, ()
            self.candidates = []
            self.activate(value, source="already satisfied native objective")
            return None
        frontier: list[tuple[tuple[Ref, ...], Value]] = [((), value)]
        visited = {repr(value)}
        evaluated: list[Candidate] = []
        solution: Candidate | None = None
        outcomes: list[Ref] = []
        for _ in range(max_depth):
            next_states: dict[Ref, tuple[tuple[Ref, ...], Value]] = {}
            level = []
            for (chain, state), op in itertools.product(frontier, available):
                if len(evaluated) >= max_nodes:
                    break
                c = self.observer.processes.structural_call(self.category, (*chain, op))
                evaluated.append(c)
                try:
                    predicted, result = self.evaluate(op, state)
                    satisfied = self.goal_test(predicate, predicted, initial=value)
                    c.results.append(result)
                    try:
                        achieved, _ = self.evaluate(
                            progress, record(current=predicted, initial=value)
                        )
                        c.errors.append(
                            1 - float(achieved) if isinstance(achieved, Fraction) else 1
                        )
                    except AthanorError:
                        c.errors.append(1)
                    c.status = "provisional" if satisfied else "qualified"
                    outcomes.append(
                        self.knowledge.create(
                            fields=record(
                                kind="planning_transition",
                                task=task,
                                operation=op,
                                state=state,
                                predicted=predicted,
                                goal_satisfied=satisfied,
                                result=result,
                            ),
                            source="native bounded planning evaluation",
                        )
                    )
                    if predicted is not None and repr(predicted) not in visited:
                        next_states[c.operation] = (c.dependencies, predicted)
                        level.append(c)
                except AthanorError:
                    c.errors.append(1)
                    c.status = "qualified"
                self.publish(c)
            if not level:
                break
            self.learn(trainer, level, steps=80)
            ranked = self.rank(level)
            # Neural selection proposes; native goal execution is the veto.
            if ranked[0].errors == [0]:
                solution = ranked[0]
                break
            frontier = [next_states[c.operation] for c in ranked[:32]]
            visited.update(repr(v) for _, v in frontier)
            if len(evaluated) >= max_nodes:
                break
        self.candidates = evaluated
        self.active = tuple(c.hypothesis for c in self.rank(evaluated)[:4])
        self.selected = None if solution is None else solution.operation
        self.status = "goal_supported" if solution else "unknown"
        fields = self.knowledge.get(task).fields
        self.tasks[-1] = self.revise(
            task,
            record(
                dict(fields),
                status=self.status,
                selected=self.selected,
                evaluated=len(evaluated),
                outcomes=tuple(outcomes),
            ),
        )
        self.cap(self.tasks[-1], "goal_achievement", solution is not None)
        if solution is None:
            self.activate(value, source="bounded objective inquiry remains unknown")
            return None
        rule = self.knowledge.create(
            fields=record(
                kind="acquired_structural_rule",
                operation=solution.operation,
                hypothesis=solution.hypothesis,
                dependencies=solution.dependencies,
                alternatives=(solution.operation,),
                examples=(),
                status="provisional",
                contradictions=(),
                outcomes=(),
                input_family=family(value),
                schema=self.signature(value),
                goal=predicate,
            ),
            source="variable-role abstraction of successful native plan",
        )
        self.rules.append(rule)
        self.library.append(solution.operation)
        self.cap(rule, "solution_abstraction", False)
        self.activate(value, source="successful plan with provisional variable abstraction")
        return rule

    def choose_test(self, values: tuple[Value, ...]) -> Value:
        if not 1 <= len(values) <= 64:
            raise ValidationError("Informative test selection requires 1--64 observations")
        supported = [c for c in self.candidates if c.status in {"supported", "hypothesized"}]
        options = []
        for value in values:
            validate_value(value)
            outputs = []
            for c in supported[:16]:
                try:
                    predicted, _ = self.evaluate(c.operation, value)
                    outputs.append(predicted)
                except AthanorError:
                    outputs.append(None)
            diversity = len({repr(v) for v in outputs}) / max(1, len(outputs))
            options.append(
                record(
                    uncertainty=1,
                    relevance=1,
                    benefit=exact(diversity),
                    reliability=1,
                    cost=exact(0.1),
                    risk=0,
                )
            )
        with torch.no_grad():
            scores = self.observer.observer.model._scores(
                priority_vectors(self.observer.observer.model, options, kind="structural_priority"),
                self.observer.observer.state.context.detach() * 0.05,
            )
        index = int(scores.argmax())
        self.history.append(
            self.knowledge.create(
                fields=record(
                    {
                        "kind": "structural_test_proposal",
                        "values": values,
                        "selected": index,
                        "scores": tuple(exact(float(v)) for v in scores),
                    },
                ),
                source="shared neural informative-test selection",
            )
        )
        self.activate(values[index], source="informative structural test proposal")
        return values[index]

    def metadata(self) -> dict[str, Any]:
        def ref(v: Ref | None) -> list[str | int] | None:
            return None if v is None else [v.id, v.revision, v.namespace]

        return {
            "category": ref(self.category),
            "library": [ref(v) for v in self.library],
            "concepts": [ref(v) for v in self.concepts],
            "rules": [ref(v) for v in self.rules],
            "candidates": [ref(c.hypothesis) for c in self.candidates],
            "candidate_scores": [c.score for c in self.candidates],
            "examples": [ref(v) for v in self.examples],
            "observations": [ref(v) for v in self.observations],
            "history": [ref(v) for v in self.history],
            "tasks": [ref(v) for v in self.tasks],
            "revisers": [ref(v) for v in self.revisers],
            "revoked": [ref(v) for v in self.revoked],
            "active": [ref(v) for v in self.active],
            "selected": ref(self.selected),
            "query": ref(self.query),
            "status": self.status,
            "neural_updates": self.neural_updates,
            "capabilities": {k: v.copy() for k, v in self.capabilities.items()},
            "resource": self.resource.copy(),
        }

    @classmethod
    def restore(cls, observer: GroundedObserver, data: dict[str, Any]) -> Induction:
        obj = cls(observer)
        if not isinstance(data, dict) or set(data) != set(obj.metadata()):
            raise ValidationError("Malformed structural inquiry metadata")

        def ref(raw: Any) -> Ref:
            if not isinstance(raw, list) or len(raw) != 3:
                raise ValidationError("Structural inquiry requires an exact pinned reference")
            r = Ref(raw[0], raw[1], raw[2])
            observer.knowledge.get(r)
            return r

        for key in ("category", "selected", "query"):
            setattr(obj, key, None if data[key] is None else ref(data[key]))
        for key in (
            "library",
            "concepts",
            "rules",
            "examples",
            "observations",
            "history",
            "tasks",
            "revisers",
            "revoked",
            "active",
        ):
            if not isinstance(data[key], list) or len(data[key]) > 65536:
                raise ValidationError("Invalid structural reference history")
            values = [ref(v) for v in data[key]]
            if len(set(values)) != len(values):
                raise ValidationError("Duplicate structural inquiry handles")
            setattr(obj, key, tuple(values) if key == "active" else values)
        if (
            obj.library
            and obj.category is None
            or obj.category is not None
            and observer.knowledge.get(obj.category).role != Role.CATEGORY
        ):
            raise ValidationError("Invalid structural interface category")
        for op in (*obj.library, *obj.revoked):
            node = observer.knowledge.get(op)
            p = as_record(node.fields.get("procedure"))
            if (
                node.role != Role.OPERATION
                or p["inputs"] != (Endpoint(Role.DATA, obj.category).to_record(),)
                or p["result"] != obj.category
                or p["update"] is not None
            ):
                raise ValidationError("Structural operation violates its pinned interface")
        if obj.category is not None and obj.knowledge.schema(obj.category) != record(
            value=FieldSpec().to_record()
        ):
            raise ValidationError("Structural category has a different value contract")
        for pool, kind in (
            (obj.concepts, "formed_concept"),
            (obj.rules, "acquired_structural_rule"),
            (obj.tasks, "structural_task"),
        ):
            for r in pool:
                n = observer.knowledge.get(r).fields
                if n.get("kind") != kind:
                    raise ValidationError("Forged structural knowledge role")
                if kind != "structural_task" and (
                    n["operation"] not in obj.library
                    or n["status"] not in {"supported", "provisional", "qualified"}
                ):
                    raise ValidationError("Invalid acquired structural knowledge")
        if not isinstance(data["candidates"], list) or len(data["candidates"]) > 2048:
            raise ValidationError("Invalid structural candidate count")
        if (
            not isinstance(data["candidate_scores"], list)
            or len(data["candidate_scores"]) != len(data["candidates"])
            or any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in data["candidate_scores"]
            )
        ):
            raise ValidationError("Invalid shared structural scores")
        for raw in data["candidates"]:
            h = ref(raw)
            n = observer.knowledge.get(h).fields
            if n.get("kind") != "structural_hypothesis" or n["status"] not in {
                "qualified",
                "supported",
                "hypothesized",
                "unknown",
                "provisional",
            }:
                raise ValidationError("Invalid native structural hypothesis")
            op = as_ref(n["operation"])
            p = as_record(observer.knowledge.get(op).fields["procedure"])
            expression = as_record(n["expression"])
            if (
                as_record(p["fields"])["value"] != field(expression, "value")
                or p["result"] != obj.category
            ):
                raise ValidationError("Structural expression disagrees with native process")
            errors = [float(cast(Fraction, v)) for v in as_tuple(n["errors"])]
            proposal_errors = [float(cast(Fraction, v)) for v in as_tuple(n["proposal_errors"])]
            if (
                len(proposal_errors) != len(errors)
                and proposal_errors
                or any(not math.isfinite(v) or not 0 <= v <= 1 for v in proposal_errors)
            ):
                raise ValidationError("Invalid structural proposal evidence")
            calls = int(cast(Fraction, n["calls"]))
            if Fraction(calls) != n["calls"]:
                raise ValidationError("Structural call count must be an exact integer")
            cursor = expression
            bound: list[Ref] = []
            while cursor.get("op") == "call":
                args = as_tuple(cursor["arguments"])
                if len(args) != 1:
                    raise ValidationError("Structural call has unbound participants")
                bound.insert(0, as_ref(cursor["operation"]))
                cursor = as_record(args[0])
            if cursor != argument(0) or tuple(bound) != n["dependencies"] or len(bound) != calls:
                raise ValidationError(
                    "Structural role binding or dependencies disagree with the process"
                )
            if n["status"] == "supported" and (len(errors) < 3 or any(errors)):
                raise ValidationError("Supported transformation lacks consistent measured examples")
            if not 1 <= calls <= 6 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in errors):
                raise ValidationError("Invalid structural discrepancy or complexity")
            obj.candidates.append(
                Candidate(
                    op,
                    h,
                    expression,
                    tuple(as_ref(v) for v in as_tuple(n["dependencies"])),
                    calls,
                    errors,
                    [as_ref(v) for v in as_tuple(n["results"])],
                    cast(str, n["status"]),
                )
            )
            obj.candidates[-1].proposal_errors = proposal_errors
        for c, score in zip(obj.candidates, data["candidate_scores"], strict=True):
            c.score = float(score)
        if obj.selected is not None and obj.selected not in {
            c.operation for c in obj.candidates
        } | {as_ref(obj.knowledge.get(r).fields["operation"]) for r in obj.rules}:
            raise ValidationError("Structural selection is not a native candidate")
        if any(
            v not in {c.hypothesis for c in obj.candidates} | set(obj.concepts) | set(obj.rules)
            for v in obj.active
        ):
            raise ValidationError("Unknown active structural hypothesis")
        if (
            data["status"]
            not in {
                "idle",
                "concepts_formed",
                "unknown",
                "supported",
                "qualified",
                "goal_supported",
            }
            or type(data["neural_updates"]) is not int
            or data["neural_updates"] < 0
        ):
            raise ValidationError("Invalid structural inquiry control state")
        obj.status, obj.neural_updates = data["status"], data["neural_updates"]
        if not isinstance(data["capabilities"], dict):
            raise ValidationError("Invalid structural capability history")
        for key, counts in data["capabilities"].items():
            if (
                not isinstance(key, str)
                or not isinstance(counts, dict)
                or any(type(v) is not int or v < 0 for v in counts.values())
            ):
                raise ValidationError("Invalid measured structural capability")
            for name in counts:
                if name.endswith("_successes") and counts[name] > counts.get(
                    name.removesuffix("_successes") + "_attempts", -1
                ):
                    raise ValidationError("Structural successes exceed attempts")
        obj.capabilities = {k: v.copy() for k, v in data["capabilities"].items()}
        resource = data["resource"]
        if (
            not isinstance(resource, dict)
            or set(resource) != {"max_calls", "max_candidates", "beam"}
            or any(type(v) is not int for v in resource.values())
            or not 1 <= resource["max_calls"] <= 4
            or not 1 <= resource["max_candidates"] <= 2048
            or not 1 <= resource["beam"] <= 128
        ):
            raise ValidationError("Invalid structural resource scope")
        obj.resource = resource.copy()
        return obj
