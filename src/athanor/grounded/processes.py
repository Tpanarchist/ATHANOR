"""Bounded native call composition and evidence-qualified interface abstraction.

No world authority, generated Python, alternate evaluator or process-specific network.
All candidate computations and adapters execute through the cornerstone.
"""

import itertools
import math
from dataclasses import dataclass
from dataclasses import field as data_field
from fractions import Fraction
from typing import TYPE_CHECKING, Any, cast

import torch
from torch.nn import functional as F

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
    ValidationError,
    record,
)
from athanor.programs import (
    argument,
    binary,
    call,
    choose,
    field,
    item,
    literal,
    record_expression,
    sequence,
)
from athanor.values import as_record, as_ref, as_tuple, exact_equal, references
from athanor.world import SensorFrame
from athanor.world.engine import exact

from .discovery import priority_vectors
from .perception import descriptor_distance, perceive

if TYPE_CHECKING:
    from .runtime import GroundedObserver
    from .training import GroundedTrainer


@dataclass(slots=True)
class Candidate:
    operation: Ref
    hypothesis: Ref
    expression: Record
    dependencies: tuple[Ref, ...]
    calls: int
    errors: list[float] = data_field(default_factory=list)
    results: list[Ref] = data_field(default_factory=list)
    status: str = "unknown"
    score: float = 0.0


def selectors(values: object, path: tuple[str | int, ...] = ()) -> list[tuple[str | int, ...]]:
    """Enumerate numeric leaves by type; names do not establish correspondence."""
    if len(path) > 5:
        return []
    if isinstance(values, Fraction):
        return [path]
    if isinstance(values, Record):
        return [p for key, value in values.items() for p in selectors(value, (*path, key))]
    if isinstance(values, tuple):
        return [p for i, value in enumerate(values) for p in selectors(value, (*path, i))]
    return []


def project(values: object, path: tuple[str | int, ...]) -> Fraction:
    for key in path:
        if isinstance(key, str) and isinstance(values, Record):
            values = values[key]
        elif type(key) is int and isinstance(values, tuple):
            values = values[key]
        else:
            raise ValidationError("Interface selector does not match the observed structure")
    if not isinstance(values, Fraction):
        raise ValidationError("Process correspondence requires an exact numeric leaf")
    return values


def stored_path(value: object) -> tuple[str | int, ...]:
    if not isinstance(value, tuple):
        raise ValidationError("A correspondence selector requires a native sequence")
    output: list[str | int] = []
    for key in value:
        if isinstance(key, str):
            output.append(key)
        elif isinstance(key, Fraction) and key.denominator == 1 and key >= 0:
            output.append(int(key))
        else:
            raise ValidationError("Invalid native selector token")
    return tuple(output)


def projection(path: tuple[str | int, ...]) -> Record:
    expression = argument(0)
    for key in path:
        expression = field(expression, key) if isinstance(key, str) else item(expression, key)
    return expression


def rebuild(
    template: object, path: tuple[str | int, ...], replacement: Record, source: Record
) -> Record:
    """Reconstruct one nested interface output through the trusted native language."""
    if not path:
        return replacement
    key, *rest = path
    if isinstance(template, Record) and isinstance(key, str) and key in template:
        return record_expression(
            {
                name: rebuild(value, tuple(rest), replacement, field(source, name))
                if name == key
                else field(source, name)
                for name, value in template.items()
            }
        )
    if isinstance(template, tuple) and type(key) is int and 0 <= key < len(template):
        return sequence(
            *(
                rebuild(value, tuple(rest), replacement, item(source, i))
                if i == key
                else item(source, i)
                for i, value in enumerate(template)
            )
        )
    raise ValidationError("Output correspondence cannot bind its native template")


def schema(values: Record) -> dict[str, FieldSpec]:
    kinds = {
        Fraction: "number",
        str: "string",
        bool: "bool",
        Ref: "ref",
        tuple: "sequence",
        Record: "record",
    }
    return {key: FieldSpec(kinds.get(type(value), "any")) for key, value in values.items()}


class Processes:
    def __init__(self, observer: GroundedObserver) -> None:
        self.observer = observer
        self.scalar: Ref | None = None
        self.library: list[Ref] = []
        self.constants: tuple[Ref, ...] = ()
        self.candidates: list[Candidate] = []
        self.archive: list[Candidate] = []
        self.examples: list[Ref] = []
        self.applications: list[Ref] = []
        self.probes: list[Ref] = []
        self.inputs: list[Fraction] = []
        self.outputs: list[Fraction] = []
        self.adopted: list[Ref] = []
        self.mappings: list[Ref] = []
        self.correspondence_hypotheses: list[Ref] = []
        self.mapping_evidence: dict[Ref, list[Ref]] = {}
        self.capabilities: dict[Ref, dict[str, int]] = {}
        self.status = "stopped"
        self.tolerance = 0.06
        self.budget = 0
        self.spent = 0
        self.selected: Ref | None = None
        self.active: tuple[Ref, ...] = ()
        self.excluded = 0
        self.neural_updates = 0
        self.action = "interact"
        self.interface = "camera"
        self.last_body_value: Fraction | None = None
        self.reviser: Ref | None = None
        self._retained: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    @property
    def knowledge(self) -> Athanor:
        return self.observer.knowledge

    def install(self) -> None:
        if self.scalar is not None:
            return
        a = self.knowledge
        # Reuse an existing typed arithmetic interface when present. Names are irrelevant.
        found: dict[str, Ref] = {}
        for node in a.structures:
            if node.role != Role.OPERATION:
                continue
            spec = node.fields["procedure"]
            if not isinstance(spec, Record) or len(as_tuple(spec["inputs"])) != 2:
                continue
            values = spec["fields"]
            expr = values.get("value") if isinstance(values, Record) else None
            if (
                isinstance(expr, Record)
                and expr.get("op") in {"add", "sub", "mul", "div"}
                and expr.get("left") == field(argument(0), "value")
                and expr.get("right") == field(argument(1), "value")
                and isinstance(spec["result"], Ref)
                and all(
                    as_record(raw)["category"] == spec["result"] for raw in as_tuple(spec["inputs"])
                )
            ):
                if self.scalar is None:
                    self.scalar = spec["result"]
                if spec["result"] == self.scalar:
                    found[str(expr["op"])] = node.ref
        if self.scalar is None:
            self.scalar = a.teach_category("Exact process quantity", {"value": FieldSpec("number")})
        for opcode in ("add", "sub", "mul", "div"):
            operation = found.get(opcode)
            if operation is None:
                operation = a.teach_operation(
                    f"Established exact {opcode}",
                    Procedure(
                        (Endpoint(Role.DATA, self.scalar),) * 2,
                        self.scalar,
                        {
                            "value": binary(
                                opcode, field(argument(0), "value"), field(argument(1), "value")
                            )
                        },
                    ),
                    source="established arithmetic teaching; no environmental answer",
                )
            self.library.append(operation)
        self.constants = (a.create(self.scalar, {"value": 1}, source="established unit quantity"),)

    def offer(self, operation: Ref) -> None:
        """Offer any taught conditional, accumulated or other compatible native procedure."""
        self.install()
        node = self.knowledge.get(operation)
        if node.role != Role.OPERATION:
            raise ValidationError("A process constituent must be an operation")
        spec = as_record(node.fields["procedure"])
        inputs = as_tuple(spec["inputs"])
        if (
            not 1 <= len(inputs) <= 3
            or spec["result"] != self.scalar
            or any(
                as_record(v)["category"] != self.scalar or as_record(v)["role"] != "data"
                for v in inputs
            )
        ):
            raise ValidationError(
                "Offered constituent does not share this typed quantity interface"
            )
        if operation not in self.library:
            self.library.append(operation)

    def _publish(self, candidate: Candidate) -> None:
        fields = record(
            kind="process_hypothesis",
            operation=candidate.operation,
            expression=candidate.expression,
            dependencies=candidate.dependencies,
            calls=candidate.calls,
            status=candidate.status,
            errors=tuple(exact(v) for v in candidate.errors),
            observations=len(candidate.errors),
            support=self.knowledge.get(candidate.hypothesis).fields.get("support", ())
            if candidate in self.archive
            else tuple(self.examples[-3:]),
            results=tuple(candidate.results[-3:]),
            tolerance=self.knowledge.get(candidate.hypothesis).fields.get(
                "tolerance", exact(self.tolerance)
            )
            if candidate in self.archive
            else exact(self.tolerance),
            preconditions="typed exact quantity; native call contracts and bounded execution",
            applicability="empirical interface scope; algebraic validity is separate",
            alternatives="other compatible call compositions may remain observationally equivalent",
            contradictions=self.knowledge.get(candidate.hypothesis).fields.get(
                "contradictions", ()
            ),
        )
        if self.reviser is None:
            self.reviser = self.knowledge.teach_operation(
                "Revise process hypothesis",
                Procedure(
                    (Endpoint(Role.DATA),) * 2,
                    None,
                    {key: field(argument(1), key) for key in fields},
                    update=0,
                ),
            )
        patch = self.knowledge.create(fields=fields, source="measured process comparison")
        candidate.hypothesis = self.knowledge.execute(self.reviser, (candidate.hypothesis, patch))

    def start(
        self,
        budget: int = 12,
        *,
        max_calls: int = 2,
        max_candidates: int = 384,
        interface: str = "camera",
    ) -> None:
        if (
            type(budget) is not int
            or not 1 <= budget <= 10000
            or type(max_calls) is not int
            or not 1 <= max_calls <= 4
            or type(max_candidates) is not int
            or not 1 <= max_candidates <= 2048
        ):
            raise ValidationError("Process construction needs explicit bounded resources")
        if interface not in {"camera", "body", "numeric"}:
            raise ValidationError("Unknown measured process interface")
        self.install()
        self.status, self.budget, self.spent = "running", budget, 0
        self.interface = interface
        self.tolerance = {"body": 0.001, "camera": 0.06, "numeric": 0.0}[interface]
        self.action = "forward" if interface == "body" else "interact"
        self.last_body_value = None
        self.examples, self.inputs, self.outputs = [], [], []
        self.probes = []
        self.archive.extend(self.candidates)
        self.candidates = self.construct(max_calls=max_calls, max_candidates=max_candidates)
        self.selected, self.active = None, ()

    def construct(self, *, max_calls: int = 2, max_candidates: int = 384) -> list[Candidate]:
        """Enumerate compatible call trees; neither evidence nor an answer enters generation."""
        if (
            type(max_calls) is not int
            or not 1 <= max_calls <= 4
            or type(max_candidates) is not int
            or not 1 <= max_candidates <= 2048
        ):
            raise ValidationError("Process construction needs bounded resources")
        self.install()
        a = self.knowledge
        terms: list[tuple[Record, int]] = [(argument(0), 0)]
        terms += [(literal(ref), 0) for ref in self.constants]
        generated: set[Record] = set()
        output: list[Candidate] = []
        attempts = 0
        for size in range(1, max_calls + 1):
            fresh: list[tuple[Record, int]] = []
            for operation in self.library:
                spec = as_record(a.get(operation).fields["procedure"])
                arity = len(as_tuple(spec["inputs"]))
                if not 1 <= arity <= 3 or spec["result"] != self.scalar:
                    continue
                if any(
                    as_record(raw)["category"] != self.scalar for raw in as_tuple(spec["inputs"])
                ):
                    continue
                for arguments in itertools.product(terms, repeat=arity):
                    attempts += 1
                    if attempts > max_candidates * 64:
                        return output
                    if sum(row[1] for row in arguments) + 1 != size:
                        continue
                    expr = call(operation, *(row[0] for row in arguments))
                    if expr in generated:
                        continue
                    if len(output) >= max_candidates:
                        return output
                    generated.add(expr)
                    fresh.append((expr, size))
                    compiled = a.teach_operation(
                        "Candidate native call composition",
                        Procedure(
                            (Endpoint(Role.DATA, self.scalar),),
                            self.scalar,
                            {"value": field(expr, "value")},
                        ),
                        source="bounded compatible structural construction; no answer supplied",
                    )
                    deps = tuple(
                        ref
                        for ref in dict.fromkeys(references(expr))
                        if a.get(ref).role == Role.OPERATION
                    )
                    hypothesis = a.create(
                        fields=record(kind="process_hypothesis", status="unknown"),
                        source="proposed untested composition",
                    )
                    candidate = Candidate(compiled, hypothesis, expr, deps, size)
                    output.append(candidate)
                    self._publish(candidate)
            terms.extend(fresh)
        return output

    def choose_probe(self, values: tuple[Fraction, ...]) -> Fraction:
        """The shared field chooses a bounded stimulus before its consequence is observed."""
        if not 1 <= len(values) <= 64 or any(not isinstance(v, Fraction) for v in values):
            raise ValidationError("Probes require a bounded exact stimulus domain")
        seen = {self.knowledge.get(r).fields["value"] for r in self.probes}
        available = [v for v in values if v not in seen]
        if not available:
            raise ValidationError("Stimulus domain exhausted; no absence claim follows")
        candidates = self.rank()[:8]
        options, predictions = [], []
        for value in available:
            predicted = []
            for c in candidates:
                try:
                    result, _ = self.evaluate(c.operation, value)
                    predicted.append(result)
                except AthanorError:
                    pass
            spread = min(Fraction(1), max(predicted) - min(predicted)) if predicted else Fraction(1)
            options.append(
                record(
                    uncertainty=1,
                    relevance=1,
                    benefit=(1 + spread) / 2,
                    reliability=1,
                    cost=abs(value) / 10,
                    risk=0,
                )
            )
            predictions.append(tuple(predicted))
        model = self.observer.observer.model
        with torch.no_grad():
            scores = model._scores(
                priority_vectors(model, options, kind="process_priority"),
                self.observer.observer.state.context.detach() * 0.05,
            )
        index = int(scores.argmax())
        self.probes.append(
            self.knowledge.create(
                fields=record(
                    kind="process_probe",
                    value=available[index],
                    score=exact(float(scores[index])),
                    predictions=predictions[index],
                ),
                source="shared neural experimental selection",
            )
        )
        return available[index]

    def evaluate(self, operation: Ref, number: Fraction) -> tuple[Fraction, Ref]:
        self.install()
        value = self.knowledge.create(self.scalar, {"value": number}, source="typed process input")
        result = self.knowledge.execute(
            operation, (value,), limits=Limits(max_steps=512, max_depth=16)
        )
        return cast(Fraction, self.knowledge.get(result).fields["value"]), result

    def _vectors(self, candidates: list[Candidate]) -> torch.Tensor:
        options = []
        for candidate in candidates:
            error = max(candidate.errors, default=self.tolerance)
            options.append(
                record(
                    uncertainty=exact(1 / (1 + len(candidate.errors))),
                    relevance=1,
                    benefit=exact(1 / (1 + error / max(self.tolerance, 1e-9))),
                    reliability=exact(
                        sum(e <= self.tolerance for e in candidate.errors)
                        / max(1, len(candidate.errors))
                    ),
                    cost=exact(candidate.calls / 8),
                    risk=exact(min(1, error)),
                )
            )
        return priority_vectors(self.observer.observer.model, options, kind="process_priority")

    def rank(self) -> list[Candidate]:
        if not self.candidates:
            return []
        model = self.observer.observer.model
        context = self.observer.observer.state.context.detach() * 0.05
        with torch.no_grad():
            scores = model._scores(self._vectors(self.candidates), context)
        for candidate, score in zip(self.candidates, scores, strict=True):
            candidate.score = float(score)
        return sorted(self.candidates, key=lambda c: c.score, reverse=True)

    def learn(self, trainer: GroundedTrainer, *, steps: int = 40) -> None:
        """Measured discrepancy teaches shared selection; retained native scores constrain drift."""
        if not self.candidates or not self.inputs:
            return
        model = self.observer.observer.model
        if type(steps) is not int or not 1 <= steps <= 1000:
            raise ValidationError("Process learning requires bounded positive updates")
        if self._retained is None:
            from athanor.neural import Observation
            from athanor.neural.curriculum import situation
            from athanor.neural.encoding import GraphBuilder
            from athanor.semantics.curriculum import case

            vectors_saved, contexts_saved, scores_saved = [], [], []
            with torch.no_grad():
                for i in range(48):
                    if i % 2 == 0:
                        frame = situation(
                            61000 + i,
                            mode=("valid", "unsupported", "missing", "incompatible", "correction")[
                                (i // 2) % 5
                            ],
                        ).frame(model.config)
                    else:
                        c = case(
                            61000 + i,
                            mode=("valid", "alias", "unknown", "missing", "ambiguous", "context")[
                                (i // 2) % 6
                            ],
                        )
                        frame = GraphBuilder(model.config).build(
                            c.knowledge,
                            Observation(c.utterance.ref, c.utterance.candidates),
                            observer=c.refs["self"],
                        )
                    for v, ctx in model([frame])[0].score_trace:
                        vectors_saved.append(v.detach())
                        contexts_saved.append(ctx.detach().expand(v.shape[0], -1))
                        scores_saved.append(model._scores(v, ctx).detach())
            self._retained = (
                torch.cat(vectors_saved),
                torch.cat(contexts_saved),
                torch.cat(scores_saved),
            )
        with torch.no_grad():
            vectors = self._vectors(self.candidates).detach()
            context = self.observer.observer.state.context.detach() * 0.05
            targets = torch.tensor(
                [
                    1 / (1 + max(c.errors, default=1) / max(self.tolerance, 1e-9)) - c.calls * 0.01
                    for c in self.candidates
                ]
            )
            trace = (
                []
                if self.observer.observer._output is None
                else list(self.observer.observer._output.score_trace)
            )
            retained = [
                (v.detach(), ctx.detach(), model._scores(v, ctx).detach()) for v, ctx in trace
            ]
            if self.observer.discovery.options:
                inquiry_values = [
                    self.knowledge.get(r).fields for r in self.observer.discovery.options
                ]
                inquiry_vectors = priority_vectors(model, inquiry_values).detach()
                retained.append(
                    (inquiry_vectors, context, model._scores(inquiry_vectors, context).detach())
                )
        # Existing scorer only; the same owned optimizer records actual updates.
        original = {name: p.requires_grad for name, p in model.named_parameters()}
        for name, p in model.named_parameters():
            p.requires_grad_(name.startswith("scorer."))
        try:
            for _ in range(steps):
                prior = {name: p.detach().clone() for name, p in model.scorer.named_parameters()}
                scores = model._scores(vectors, context)
                # Relative ranking uses observed errors, never the territory's equation.
                loss = F.kl_div(scores.log_softmax(0), (targets * 12).softmax(0), reduction="sum")
                for v, ctx, expected in retained:
                    loss = loss + 2 * F.mse_loss(model._scores(v, ctx), expected)
                v, ctx, expected = self._retained
                loss = loss + 4 * F.mse_loss(model._scores(v, ctx), expected)
                trainer.optimizer.zero_grad()
                torch.autograd.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.scorer.parameters(), 1.0)
                trainer.optimizer.step()
                with torch.no_grad():
                    for name, p in model.scorer.named_parameters():
                        p.copy_(prior[name] + (p - prior[name]).clamp(-0.0002, 0.0002))
                trainer.updates += 1
                self.neural_updates += 1
        finally:
            for name, p in model.named_parameters():
                p.requires_grad_(original[name])

    def observe(
        self, before: Fraction, after: Fraction, *, source: str, witness: Ref | None = None
    ) -> Ref:
        if self.status != "running" or self.spent >= self.budget:
            raise ValidationError("Process inquiry is not running or its resources are exhausted")
        if not isinstance(before, Fraction) or not isinstance(after, Fraction):
            raise ValidationError("Process observations require exact stored measurements")
        evidence = self.knowledge.create(
            fields=record(
                kind="process_evidence",
                before=before,
                after=after,
                source_interface=source,
                witness=witness,
                selected=self.selected,
                tolerance=exact(self.tolerance),
            ),
            source=source,
        )
        self.examples.append(evidence)
        self.inputs.append(before)
        self.outputs.append(after)
        self.spent += 1
        contradicted = []
        for candidate in self.candidates:
            try:
                predicted, result = self.evaluate(candidate.operation, before)
                error = min(1000.0, float(abs(predicted - after)))
                candidate.results.append(result)
            except AthanorError:
                error = 1000.0
            candidate.errors.append(error)
            candidate.status = (
                "qualified"
                if max(candidate.errors) > self.tolerance
                else "supported"
                if len(set(self.inputs)) >= 3
                else "hypothesized"
            )
            self._publish(candidate)
            if candidate.operation in self.adopted and candidate.status == "qualified":
                contradicted.append(candidate.operation)
        for operation in contradicted:
            self.qualify(operation, "new process observation contradicts support", witness=evidence)
        ranked = self.rank()
        self.selected = None if not ranked else ranked[0].operation
        self.active = tuple(c.hypothesis for c in ranked[:4])
        if self.spent == self.budget:
            self.status = "exhausted"
        return evidence

    def ingest(self, before: SensorFrame, after: SensorFrame, feedback: Ref) -> None:
        if self.interface == "numeric" or self.status != "running" or after.executed != self.action:
            return
        if self.interface == "body":
            measured = exact(math.dist(after.proprio[:3], (0, 0, 0)))
            if measured <= 0 or abs(after.proprio[3]) > 0.001:
                self.excluded += 1
                return
            if self.last_body_value is not None:
                self.observe(
                    self.last_body_value,
                    measured,
                    source="measured body displacement history",
                    witness=feedback,
                )
            self.last_body_value = measured
            return
        if (
            any(abs(v) > 0.001 for v in after.proprio[:4])
            or before.left == after.left
            and before.right == after.right
        ):
            self.excluded += 1
            return
        old = [
            p
            for p in perceive(self.observer.observer.sensory, before)
            if p.status == "visible" and p.relative is not None
        ]
        new = [
            p
            for p in perceive(self.observer.observer.sensory, after)
            if p.status == "visible" and p.relative is not None
        ]
        if (
            len(old) != 1
            or len(new) != 1
            or descriptor_distance(old[0].appearance, new[0].appearance) >= 0.18
        ):
            self.excluded += 1
            return
        assert old[0].relative is not None and new[0].relative is not None
        self.observe(
            exact(old[0].relative[2]),
            exact(new[0].relative[2]),
            source="raw stereo consequence",
            witness=feedback,
        )

    def adopt(self) -> Ref | None:
        # Neural ranking chooses; evidence vetoes insufficient candidates without substitution.
        ranked = self.rank()
        if not ranked:
            return None
        chosen = ranked[0]
        self.selected = chosen.operation
        self._capability(chosen.operation, "recognized", True)
        if chosen.status != "supported":
            return None
        if any(
            c.operation in chosen.dependencies and c.status != "supported"
            for c in [*self.archive, *self.candidates]
            if c.operation in self.adopted
        ):
            return None
        if chosen.operation not in self.adopted:
            self.knowledge.create(
                fields=record(
                    kind="acquired_process",
                    operation=chosen.operation,
                    hypothesis=chosen.hypothesis,
                    constituents=chosen.dependencies,
                    expression=chosen.expression,
                    input_type=self.scalar,
                    output_type=self.scalar,
                    observations=tuple(self.examples),
                    preserved="exact quantity transformation; separately tested interface scope",
                ),
                source="neural-selected evidence-qualified native process adoption",
            )
            self.adopted.append(chosen.operation)
            self.library.append(chosen.operation)
        return chosen.operation

    def _capability(self, operation: Ref, level: str, success: bool) -> None:
        counts = self.capabilities.setdefault(operation, {})
        counts[level + "_attempts"] = counts.get(level + "_attempts", 0) + 1
        counts[level + "_successes"] = counts.get(level + "_successes", 0) + int(success)
        ref = self.knowledge.create(
            fields=record(
                kind="process_capability",
                operation=operation,
                level=level,
                success=success,
                counts=record(counts),
                observer=self.observer.observer.self_ref,
            ),
            source="measured process recognition/binding/execution/transfer",
        )
        self.observer.observer.self_map.append(ref)
        self.observer.observer.self_map = self.observer.observer.self_map[
            -self.observer.observer.model.config.history_limit :
        ]

    def apply(
        self, operation: Ref, value: Fraction, *, expected: Fraction | None = None
    ) -> Ref | None:
        if operation not in self.adopted:
            raise ValidationError("Process has not been adopted")
        candidate = next(
            (c for c in [*self.archive, *self.candidates] if c.operation == operation), None
        )
        if candidate is not None and candidate.status != "supported":
            return None
        try:
            result, ref = self.evaluate(operation, value)
        except AthanorError:
            self._capability(operation, "binding", False)
            raise
        self._capability(operation, "binding", True)
        success = expected is None or abs(result - expected) <= exact(self.tolerance)
        self._capability(operation, "execution", success)
        if expected is not None:
            self._capability(operation, "transfer", success)
        if not success:
            witness = self.knowledge.create(
                fields=record(
                    kind="process_contradiction",
                    operation=operation,
                    input=value,
                    expected=expected,
                    actual=result,
                    result=ref,
                ),
                source="fresh measured application",
            )
            self.qualify(
                operation, "fresh application contradicts acquired process", witness=witness
            )
        self.applications.append(
            self.knowledge.create(
                fields=record(
                    kind="process_application",
                    operation=operation,
                    input=value,
                    result=ref,
                    output=result,
                    expected=expected,
                    success=success,
                    applicability="untested" if expected is None else "measured",
                ),
                source="actual native acquired process application",
            )
        )
        self.applications = self.applications[-64:]
        return ref

    def qualify(self, operation: Ref, reason: str, *, witness: Ref | None = None) -> None:
        affected = {operation}
        changed = True
        while changed:
            changed = False
            for candidate in [*self.archive, *self.candidates]:
                if (
                    candidate.operation in self.adopted
                    and any(d in affected for d in candidate.dependencies)
                    and candidate.operation not in affected
                ):
                    affected.add(candidate.operation)
                    changed = True
        for candidate in [*self.archive, *self.candidates]:
            if candidate.operation in affected:
                candidate.status = "qualified"
                contradiction = self.knowledge.create(
                    fields=record(
                        kind="process_contradiction",
                        operation=candidate.operation,
                        dependency=operation,
                        reason=reason,
                        witness=witness,
                    ),
                    source=reason,
                )
                prior = self.knowledge.get(candidate.hypothesis).fields
                patch = self.knowledge.create(
                    fields=record(
                        {
                            **dict(prior),
                            "contradictions": (*as_tuple(prior["contradictions"]), contradiction),
                        }
                    )
                )
                assert self.reviser is not None
                candidate.hypothesis = self.knowledge.execute(
                    self.reviser, (candidate.hypothesis, patch)
                )
                self._publish(candidate)
        for mapping in tuple(self.mappings):
            fields = self.knowledge.get(mapping).fields
            if fields["process"] in affected:
                self.knowledge.create(
                    fields=record(kind="mapping_contradiction", mapping=mapping, reason=reason),
                    source=reason,
                )
                self.mapping_evidence.setdefault(mapping, []).append(operation)

    def correspond(
        self,
        operation: Ref,
        packets: list[tuple[Record, Record]],
        *,
        source: str,
        tolerance: float = 0,
    ) -> Ref | None:
        if operation not in self.adopted or len(packets) < 3:
            return None
        if len(packets) > 64:
            raise ValidationError("Correspondence evidence exceeds its bounded import budget")
        if any(
            c.operation == operation and c.status != "supported"
            for c in [*self.archive, *self.candidates]
        ):
            return None
        if not 0 <= tolerance <= 0.1:
            raise ValidationError("Invalid correspondence tolerance")
        incoming_paths, outgoing_paths = selectors(packets[0][0]), selectors(packets[0][1])
        if len(incoming_paths) > 16 or len(outgoing_paths) > 16:
            raise ValidationError("Interface selector search exceeds its bounded resources")
        choices = []
        for incoming in incoming_paths:
            for outgoing in outgoing_paths:
                errors, inputs = [], []
                for before, after in packets:
                    try:
                        number = project(before, incoming)
                        predicted, _ = self.evaluate(operation, number)
                        inputs.append(number)
                        errors.append(float(abs(predicted - project(after, outgoing))))
                    except AthanorError, KeyError, IndexError:
                        errors.append(1000.0)
                evidence = self.knowledge.create(
                    fields=record(
                        kind="correspondence_hypothesis",
                        process=operation,
                        input_selector=incoming,
                        output_selector=outgoing,
                        errors=tuple(exact(e) for e in errors),
                        packets=tuple(record(before=b, after=a) for b, a in packets),
                        status="supported"
                        if len(set(inputs)) >= 3 and max(errors) <= tolerance
                        else "qualified",
                        source_interface=source,
                        preserved="typed quantity input/output execution",
                    ),
                    source=source,
                )
                choices.append((incoming, outgoing, evidence, errors, inputs))
                self.correspondence_hypotheses.append(evidence)
        if not choices:
            return None
        # Same shared neural selection features; validity remains an independent gate.
        model = self.observer.observer.model
        options = [
            record(
                uncertainty=exact(1 / len(packets)),
                relevance=1,
                benefit=exact(1 / (1 + max(c[3]) * 32)),
                reliability=int(max(c[3]) <= tolerance),
                cost=exact(0.1),
                risk=exact(min(1, max(c[3]))),
            )
            for c in choices
        ]
        with torch.no_grad():
            scores = model._scores(
                priority_vectors(model, options, kind="process_priority"),
                self.observer.observer.state.context.detach() * 0.05,
            )
        selected = choices[int(scores.argmax())]
        incoming, outgoing, evidence, errors, inputs = selected
        self._capability(operation, "correspondence", max(errors) <= tolerance)
        if len(set(inputs)) < 3 or max(errors) > tolerance:
            return None
        if sum(len(set(c[4])) >= 3 and max(c[3]) <= tolerance for c in choices) > 1:
            # Co-varying leaves do not establish which binding is preserved.
            self.correspondence_hypotheses.append(
                self.knowledge.create(
                    fields=record(
                        kind="correspondence_uncertainty",
                        process=operation,
                        status="unknown",
                        alternatives=tuple(
                            c[2] for c in choices if len(set(c[4])) >= 3 and max(c[3]) <= tolerance
                        ),
                        reason="multiple supported bindings require distinguishing evidence",
                    )
                )
            )
            return None
        mapping = self.knowledge.transaction(
            lambda staged: self._bind(
                staged, operation, packets, source, tolerance, incoming, outgoing, evidence, inputs
            )
        )
        self.mappings.append(mapping)
        self.mapping_evidence[mapping] = [evidence]
        return mapping

    def _bind(
        self,
        a: Athanor,
        operation: Ref,
        packets: list[tuple[Record, Record]],
        source: str,
        tolerance: float,
        incoming: tuple[str | int, ...],
        outgoing: tuple[str | int, ...],
        evidence: Ref,
        inputs: list[Fraction],
    ) -> Ref:
        conditions = record(
            {
                key: value
                for key, value in packets[0][0].items()
                if key != incoming[0] and all(b.get(key) == value for b, _ in packets)
            }
        )
        binding = projection(incoming)
        for key, value in conditions.items():
            binding = choose(
                binary("eq", field(argument(0), key), literal(value)), binding, literal(None)
            )
        input_interface = a.teach_category("Observed input packet contract", schema(packets[0][0]))
        output_interface = a.teach_category(
            "Observed output packet contract", schema(packets[0][1])
        )
        adapter = a.teach_operation(
            "Acquired typed interface input adapter",
            Procedure(
                (Endpoint(Role.DATA, input_interface),),
                self.scalar,
                {"value": binding},
            ),
            source="evidence-qualified interface binding",
        )
        application = a.teach_operation(
            "Acquired cross-interface process application",
            Procedure(
                (Endpoint(Role.DATA, input_interface),),
                self.scalar,
                {"value": field(call(operation, call(adapter, argument(0))), "value")},
            ),
            source="reuse same pinned process through a new binding interface",
        )
        output_binding = a.teach_operation(
            "Acquired typed output interface binding",
            Procedure(
                (Endpoint(Role.DATA, self.scalar), Endpoint(Role.DATA, output_interface)),
                output_interface,
                {
                    key: as_record(expr)
                    for key, expr in as_record(
                        rebuild(packets[0][1], outgoing, field(argument(0), "value"), argument(1))[
                            "fields"
                        ]
                    ).items()
                },
            ),
            source="native reconstruction of preserved output through learned selector",
        )
        mapping = a.create(
            fields=record(
                kind="process_correspondence",
                process=operation,
                adapter=adapter,
                application=application,
                output_binding=output_binding,
                input_interface=input_interface,
                output_interface=output_interface,
                input_selector=incoming,
                output_selector=outgoing,
                conditions=conditions,
                evidence=evidence,
                input_type=self.scalar,
                output_type=self.scalar,
                minimum=min(inputs),
                maximum=max(inputs),
                status="supported",
                tolerance=exact(tolerance),
                source_interface=source,
            ),
            source="learned structural correspondence; scope is observed, not universal",
        )
        return mapping

    def apply_mapping(
        self, mapping: Ref, packet: Record, *, observed: Record | None = None
    ) -> Ref | None:
        if mapping not in self.mappings:
            raise ValidationError("Unknown process correspondence")
        fields = self.knowledge.get(mapping).fields
        operation = as_ref(fields["process"])
        if any(ref in self.adopted for ref in self.mapping_evidence[mapping]):
            return None
        try:
            if any(
                not exact_equal(packet.get(k), v)
                for k, v in as_record(fields["conditions"]).items()
            ):
                self._capability(operation, "binding", False)
                return None
            value = project(packet, stored_path(fields["input_selector"]))
            if not cast(Fraction, fields["minimum"]) <= value <= cast(Fraction, fields["maximum"]):
                return None
            input_ref = self.knowledge.create(
                as_ref(fields["input_interface"]), fields=packet, source="new interface packet"
            )
            result = self.knowledge.execute(as_ref(fields["application"]), (input_ref,))
            self._capability(operation, "binding", True)
        except AthanorError, KeyError, IndexError:
            self._capability(operation, "binding", False)
            return None
        self._capability(operation, "execution", True)
        self.applications.append(
            self.knowledge.create(
                fields=record(
                    kind="process_application",
                    operation=operation,
                    mapping=mapping,
                    input=input_ref,
                    result=result,
                    output=self.knowledge.get(result).fields,
                    applicability="untested" if observed is None else "measured",
                ),
                source="actual native cross-interface application",
            )
        )
        self.applications = self.applications[-64:]
        if observed is not None:
            success = abs(
                cast(Fraction, self.knowledge.get(result).fields["value"])
                - project(observed, stored_path(fields["output_selector"]))
            ) <= cast(Fraction, fields["tolerance"])
            self._capability(operation, "transfer", success)
            evidence = self.knowledge.create(
                fields=record(
                    kind="mapping_application",
                    mapping=mapping,
                    input=input_ref,
                    result=result,
                    observed=observed,
                    success=success,
                ),
                source="fresh cross-interface consequence",
            )
            self.mapping_evidence[mapping].append(evidence)
            if not success:
                self.mapping_evidence[mapping].append(operation)
        return result

    def materialize_mapping(self, mapping: Ref, packet: Record, template: Record) -> Ref | None:
        result = self.apply_mapping(mapping, packet)
        if result is None:
            return None
        fields = self.knowledge.get(mapping).fields
        output = self.knowledge.create(
            as_ref(fields["output_interface"]),
            fields=template,
            source="output interface template; no transformation answer",
        )
        return self.knowledge.execute(as_ref(fields["output_binding"]), (result, output))

    def metadata(self) -> dict[str, Any]:
        def ref(value: Ref | None) -> list[str | int] | None:
            return None if value is None else [value.id, value.revision, value.namespace]

        def row(c: Candidate) -> dict[str, Any]:
            return dict(
                operation=ref(c.operation),
                hypothesis=ref(c.hypothesis),
                calls=c.calls,
                errors=c.errors,
                results=[ref(r) for r in c.results],
                status=c.status,
                score=c.score,
            )

        # Expressions, evidence, revisions and mappings are authoritative native data.
        return dict(
            scalar=ref(self.scalar),
            library=[ref(r) for r in self.library],
            constants=[ref(r) for r in self.constants],
            archived_candidates=[row(c) for c in self.archive],
            candidates=[
                dict(
                    operation=ref(c.operation),
                    hypothesis=ref(c.hypothesis),
                    calls=c.calls,
                    errors=c.errors,
                    results=[ref(r) for r in c.results],
                    status=c.status,
                    score=c.score,
                )
                for c in self.candidates
            ],
            examples=[ref(r) for r in self.examples],
            applications=[ref(r) for r in self.applications],
            probes=[ref(r) for r in self.probes],
            inputs=[str(v) for v in self.inputs],
            outputs=[str(v) for v in self.outputs],
            adopted=[ref(r) for r in self.adopted],
            mappings=[ref(r) for r in self.mappings],
            correspondence_hypotheses=[ref(r) for r in self.correspondence_hypotheses],
            mapping_evidence=[
                dict(mapping=ref(r), evidence=[ref(e) for e in values])
                for r, values in self.mapping_evidence.items()
            ],
            capabilities=[dict(operation=ref(r), counts=c) for r, c in self.capabilities.items()],
            status=self.status,
            tolerance=self.tolerance,
            budget=self.budget,
            spent=self.spent,
            selected=ref(self.selected),
            active=[ref(r) for r in self.active],
            excluded=self.excluded,
            neural_updates=self.neural_updates,
            action=self.action,
            interface=self.interface,
            last_body_value=None if self.last_body_value is None else str(self.last_body_value),
            reviser=ref(self.reviser),
        )

    @classmethod
    def restore(cls, observer: GroundedObserver, data: dict[str, Any]) -> Processes:
        obj = cls(observer)
        if set(data) != set(obj.metadata()):
            raise ValidationError("Malformed persistent process state")

        def ref(raw: Any) -> Any:
            if raw is None:
                return None
            if not isinstance(raw, list) or len(raw) != 3 or type(raw[1]) is not int:
                raise ValidationError("Malformed process reference")
            r = Ref(raw[0], raw[1], raw[2])
            observer.knowledge.get(r)
            return r

        def required_ref(raw: Any) -> Ref:
            resolved = ref(raw)
            if not isinstance(resolved, Ref):
                raise ValidationError(
                    "A persistent process participant requires a pinned reference"
                )
            return resolved

        obj.scalar, obj.reviser = ref(data["scalar"]), ref(data["reviser"])
        for name in (
            "library",
            "constants",
            "examples",
            "applications",
            "probes",
            "adopted",
            "mappings",
            "active",
            "correspondence_hypotheses",
        ):
            setattr(
                obj,
                name,
                tuple(required_ref(r) for r in data[name])
                if name in {"constants", "active"}
                else [required_ref(r) for r in data[name]],
            )
        obj.inputs, obj.outputs = (
            [Fraction(v) for v in data["inputs"]],
            [Fraction(v) for v in data["outputs"]],
        )
        for name in (
            "status",
            "tolerance",
            "budget",
            "spent",
            "excluded",
            "neural_updates",
            "action",
            "interface",
        ):
            setattr(obj, name, data[name])
        obj.last_body_value = (
            None if data["last_body_value"] is None else Fraction(data["last_body_value"])
        )
        obj.selected = ref(data["selected"])
        if (
            obj.status not in {"stopped", "running", "paused", "exhausted"}
            or type(obj.budget) is not int
            or type(obj.spent) is not int
            or not 0 <= obj.spent <= obj.budget <= 10000
            or not math.isfinite(obj.tolerance)
            or not 0 <= obj.tolerance <= 0.1
            or len(obj.inputs) != len(obj.outputs)
            or len(obj.examples) != len(obj.inputs)
            or len(obj.inputs) != obj.spent
            or len(data["candidates"]) > 2048
        ):
            raise ValidationError("Invalid process inquiry resources/evidence")
        archived = {str(row["operation"]) for row in data["archived_candidates"]}
        for row in [*data["candidates"], *data["archived_candidates"]]:
            operation, hypothesis = required_ref(row["operation"]), required_ref(row["hypothesis"])
            node = observer.knowledge.get(hypothesis)
            expr = as_record(node.fields["expression"])
            candidate = Candidate(
                operation,
                hypothesis,
                expr,
                tuple(as_ref(r) for r in as_tuple(node.fields["dependencies"])),
                row["calls"],
                row["errors"],
                [required_ref(r) for r in row["results"]],
                row["status"],
                row["score"],
            )
            if (
                node.fields["operation"] != operation
                or node.fields["status"] != candidate.status
                or tuple(exact(v) for v in candidate.errors) != node.fields["errors"]
                or any(not math.isfinite(v) or v < 0 for v in candidate.errors)
                or observer.knowledge.get(operation).role != Role.OPERATION
                or type(candidate.calls) is not int
                or not 1 <= candidate.calls <= 4
                or node.fields["calls"] != candidate.calls
                or not math.isfinite(candidate.score)
                or candidate.status not in {"unknown", "hypothesized", "supported", "qualified"}
                or as_record(
                    as_record(observer.knowledge.get(operation).fields["procedure"])["fields"]
                )["value"]
                != field(expr, "value")
            ):
                raise ValidationError("Process metadata disagrees with native hypothesis")
            (obj.archive if str(row["operation"]) in archived else obj.candidates).append(candidate)
        obj.mapping_evidence = {
            required_ref(row["mapping"]): [required_ref(r) for r in row["evidence"]]
            for row in data["mapping_evidence"]
        }
        obj.capabilities = {
            required_ref(row["operation"]): row["counts"] for row in data["capabilities"]
        }
        known = {c.operation for c in [*obj.candidates, *obj.archive]}
        if (
            any(r not in known for r in obj.adopted)
            or obj.selected is not None
            and obj.selected not in {c.operation for c in obj.candidates}
            or set(obj.mapping_evidence) != set(obj.mappings)
            or any(observer.knowledge.get(r).role != Role.OPERATION for r in obj.library)
            or obj.interface not in {"camera", "body", "numeric"}
            or any(type(v) is not int or v < 0 for v in (obj.excluded, obj.neural_updates))
            or len(known) != len(obj.candidates) + len(obj.archive)
            or any(
                r is None
                for values in (
                    obj.library,
                    obj.constants,
                    obj.examples,
                    obj.probes,
                    obj.adopted,
                    obj.mappings,
                    obj.active,
                    obj.correspondence_hypotheses,
                )
                for r in values
            )
            or any(r not in {c.hypothesis for c in obj.candidates} for r in obj.active)
            or any(len(c.errors) != len(obj.inputs) for c in obj.candidates)
            or obj.scalar is not None
            and observer.knowledge.get(obj.scalar).role != Role.CATEGORY
        ):
            raise ValidationError("Invalid process library, inquiry or correspondence state")
        if obj.scalar is None and (obj.library or known or obj.constants):
            raise ValidationError("A persistent process library requires its exact typed interface")
        for operation in set(obj.library) | known:
            spec = as_record(observer.knowledge.get(operation).fields["procedure"])
            endpoints = as_tuple(spec["inputs"])
            if (
                spec["result"] != obj.scalar
                or not 1 <= len(endpoints) <= 3
                or any(
                    as_record(e)["category"] != obj.scalar or as_record(e)["role"] != "data"
                    for e in endpoints
                )
            ):
                raise ValidationError("Persistent constituent interface disagrees with the library")
        if any(
            observer.knowledge.get(r).category != obj.scalar
            or observer.knowledge.get(r).role != Role.DATA
            for r in obj.constants
        ):
            raise ValidationError("Persistent constants disagree with the typed process interface")
        for mapping in obj.mappings:
            fields = observer.knowledge.get(mapping).fields
            if (
                fields.get("kind") != "process_correspondence"
                or fields["process"] not in obj.adopted
                or fields["input_type"] != obj.scalar
                or fields["output_type"] != obj.scalar
                or not isinstance(fields["minimum"], Fraction)
                or not isinstance(fields["maximum"], Fraction)
                or fields["minimum"] > fields["maximum"]
                or fields["status"] != "supported"
            ):
                raise ValidationError("Invalid persistent correspondence contract")
            stored_path(fields["input_selector"])
            stored_path(fields["output_selector"])
            as_record(fields["conditions"])
            for name in ("adapter", "application", "output_binding"):
                if observer.knowledge.get(as_ref(fields[name])).role != Role.OPERATION:
                    raise ValidationError("A correspondence binding must be a native operation")
        for counts in obj.capabilities.values():
            for name, value in counts.items():
                if (
                    type(value) is not int
                    or value < 0
                    or not name.endswith(("_attempts", "_successes"))
                    or name.endswith("_successes")
                    and value > counts.get(name.replace("_successes", "_attempts"), 0)
                ):
                    raise ValidationError("Invalid process capability evidence")
        return obj
