"""Evidence-qualified construction of executable interpretations on shared machinery.

Only observations and measured relationship outcomes cross this boundary. Proposals are
native expression data; execution, learning and planning reuse the existing observer.
"""

import itertools
from collections.abc import Iterable
from fractions import Fraction
from typing import TYPE_CHECKING, Any, cast

import torch

from athanor import (
    Athanor,
    AthanorError,
    Endpoint,
    Limits,
    Record,
    Ref,
    Role,
    ValidationError,
    Value,
    record,
)
from athanor.programs import (
    argument,
    binary,
    call,
    choose,
    field,
    has_field,
    literal,
    record_expression,
    sequence,
    structural,
    variable,
)
from athanor.structural import collection, family, validate_value
from athanor.values import as_record, as_ref, as_tuple, exact_equal
from athanor.world import SensorFrame
from athanor.world.engine import exact

from .induction import Induction, discrepancy, role_property
from .perception import pixels
from .processes import Candidate

if TYPE_CHECKING:
    from .runtime import GroundedObserver
    from .training import GroundedTrainer


def value_expression() -> Record:
    return field(argument(0), "value")


def numeric_fields(values: Iterable[Value]) -> tuple[str, ...]:
    rows = list(values)
    if not rows or any(not isinstance(v, Record) for v in rows):
        return ()
    names = set(as_record(rows[0]))
    for row in rows:
        names &= {k for k, v in as_record(row).items() if isinstance(v, Fraction)}
    return tuple(sorted(names))[:6]


class Representations:
    """A front end of native induction, not an independent symbolic solver."""

    def __init__(self, observer: GroundedObserver) -> None:
        self.observer = observer
        self.representations: list[Ref] = []
        self.observations: list[Ref] = []
        self.evidence: list[Ref] = []
        self.interpretations: list[Ref] = []
        self.concepts: list[Ref] = []
        self.fragments: list[Ref] = []
        self.tasks: list[Ref] = []
        self.history: list[Ref] = []
        self.active: tuple[Ref, ...] = ()
        self.alternatives: tuple[Ref, ...] = ()
        self.selected: Ref | None = None
        self.status = "idle"
        self.search = {
            "proposed": 0,
            "executed": 0,
            "incompatible": 0,
            "cache_hits": 0,
            "max_candidates": 96,
            "max_depth": 6,
            "beam": 8,
        }
        self.neural_updates = 0
        self._operators: dict[str, Ref] = {}
        self._observed: dict[str, Ref] = {}
        self._cache: dict[tuple[Ref, Ref], Ref] = {}

    @property
    def induction(self) -> Induction:
        return self.observer.induction

    @property
    def knowledge(self) -> Athanor:
        return self.observer.knowledge

    def current(self, ref: Ref) -> Ref:
        for pool in (self.representations, self.concepts, self.fragments):
            for value in pool:
                if value.namespace == ref.namespace and value.id == ref.id:
                    return value
        raise ValidationError("Unknown representation participant")

    def observe(
        self,
        value: Value,
        *,
        source: str = "user structural observation",
        capture: Ref | None = None,
        sufficient: bool = True,
    ) -> Ref:
        validate_value(value)
        key = repr((value, capture, sufficient))
        if key in self._observed:
            return self._observed[key]
        ref = self.knowledge.create(
            fields=record(
                kind="representation_observation",
                value=value,
                capture=capture,
                sufficient=sufficient,
            ),
            source=source,
        )
        self.observations.append(ref)
        self._observed[key] = ref
        return ref

    def camera(self, frame: SensorFrame, *, eye: int = 0, measurements: bool = False) -> Ref:
        """Learned foreground samples only. No World, objects, labels or scene graph."""
        if (
            eye not in (0, 1)
            or type(measurements) is not bool
            or frame.width > 64
            or frame.height > 64
        ):
            raise ValidationError("Camera interpretation requires a bounded eye/frame")
        self.observer.observer.sensor = frame
        encoder = self.observer.observer.sensory
        encoder.eval()
        with torch.no_grad():
            logits, appearance, _ = encoder.encode(pixels(frame))
            confidence = logits[eye].sigmoid()
        rows = tuple(
            tuple(
                record(
                    appearance=tuple(exact(float(v)) for v in appearance[eye, :, y, x]),
                    confidence=exact(float(confidence[y, x])),
                )
                if confidence[y, x] > 0.6
                else None
                for x in range(frame.width)
            )
            for y in range(frame.height)
        )
        # Symbolic raster cells are atomic; positions and feature records enter a graph
        # as measurements. Adjacency is not an authoritative object grouping.
        symbolic = tuple(tuple(True if v is not None else None for v in row) for row in rows)
        occupied = sum(v is not None for row in symbolic for v in row)
        boundary = any(v is not None for v in symbolic[0] + symbolic[-1]) or any(
            row[0] is not None or row[-1] is not None for row in symbolic
        )
        capture = self.knowledge.create(
            fields=record(
                kind="representation_camera_capture",
                width=frame.width,
                height=frame.height,
                eye=eye,
                tick=frame.tick,
                rgb=(frame.left if eye == 0 else frame.right).hex(),
                features=rows,
                sufficient=0 < occupied <= 128 and not boundary,
            ),
            source="raw RGB and existing learned foreground/appearance encoder",
        )
        value: Value = symbolic
        if measurements:
            samples = tuple(
                record(
                    x=x,
                    y=y,
                    appearance=as_record(v)["appearance"],
                    confidence=as_record(v)["confidence"],
                )
                for y, row in enumerate(rows)
                for x, v in enumerate(row)
                if v is not None
            )
            value = record(foreground=symbolic, measurements=samples)
        ref = self.observe(
            value,
            capture=capture,
            sufficient=0 < occupied <= 128 and not boundary,
            source="learned pixel measurements",
        )
        self.induction.activate(value, source="representation camera observation")
        return ref

    def datum(self, value: Value | Ref) -> Ref:
        if isinstance(value, Ref):
            if value not in self.observations and value not in self.interpretations:
                raise ValidationError("Interpretations require an owned observation reference")
            return value
        return self.observe(value)

    def raw(self, ref: Ref) -> Value:
        node = self.knowledge.get(ref).fields
        if node["kind"] == "representation_interpretation":
            return self.knowledge.get(as_ref(node["result"])).fields["value"]
        return node["value"]

    def sufficient(self, ref: Ref) -> bool:
        node = self.knowledge.get(ref).fields
        if node["kind"] == "representation_interpretation":
            return self.sufficient(as_ref(node["observation"]))
        return node["sufficient"] is True

    def operator(self, expression: Record, *, source: str) -> Ref:
        key = repr(expression)
        if key not in self._operators:
            self._operators[key] = self.induction.operation(expression, source=source)
        return self._operators[key]

    def execute(self, operation: Ref, value: Value) -> tuple[Value, Ref]:
        self.induction.install()
        validate_value(value)
        key = repr(value)
        datum = self.induction._input_cache.get(key)
        if datum is None:
            datum = self.knowledge.create(
                self.induction.category,
                {"value": value},
                source="shared immutable representation input binding",
            )
            if len(self.induction._input_cache) >= 512:
                del self.induction._input_cache[next(iter(self.induction._input_cache))]
            self.induction._input_cache[key] = datum
        output = self.knowledge.execute(
            operation, (datum,), limits=Limits(max_steps=250000, max_depth=24)
        )
        result = self.knowledge.get(output).fields["value"]
        validate_value(result)
        self.search["executed"] += 1
        return result, output

    def compatible(self, ref: Ref, value: Value) -> bool:
        node = self.knowledge.get(ref).fields
        return node["status"] != "qualified" and self.induction.signature(value) == node["input"]

    def interface_contract(self, operations: tuple[Ref, ...], sample: Value) -> Value:
        """Keep required observed keys; structural algorithms can use unfamiliar fields."""
        pending: list[Value] = [self.knowledge.get(op).fields["procedure"] for op in operations]
        followed = set(operations)
        while pending:
            value = pending.pop()
            if isinstance(value, Record):
                if value.get("op") in {"field", "has_field"} and value.get("name") != "value":
                    return self.induction.signature(sample)
                if value.get("op") == "call" and isinstance(value.get("operation"), Ref):
                    operation = as_ref(value["operation"])
                    if operation not in followed:
                        followed.add(operation)
                        pending.append(self.knowledge.get(operation).fields["procedure"])
                pending.extend(value.values())
            elif isinstance(value, tuple):
                pending.extend(value)
        kind = family(sample)
        return record(
            family=kind,
            interface_only=True,
            items=tuple(sorted({family(v) for v in collection(sample)}))
            if kind == "sequence"
            else (),
        )

    def accepts(self, contract: Value, value: Value) -> bool:
        if not isinstance(contract, Record) or contract.get("interface_only") is not True:
            return self.induction.signature(value) == contract
        kind = family(value)
        return kind == contract["family"] and (
            kind != "sequence"
            or tuple(sorted({family(v) for v in collection(value)})) == contract["items"]
        )

    def interpret(self, representation: Ref, value: Value | Ref) -> Ref | None:
        if representation not in self.representations:
            raise ValidationError("Use the current acquired representation revision")
        observation = self.datum(value)
        node = self.knowledge.get(representation).fields
        raw = self.raw(observation)
        if not self.sufficient(observation) or not self.compatible(representation, raw):
            self.induction.cap(representation, "representation_selection", False)
            self.history.append(
                self.knowledge.create(
                    fields=record(
                        kind="representation_failure",
                        observation=observation,
                        representation=representation,
                        explanations=(
                            "insufficient_observation",
                            "inadequate_representation",
                            "role_binding",
                        ),
                        status="unknown",
                    ),
                    source="unresolved interpretation; competing causes retained",
                )
            )
            self.induction.activate(raw, source="unresolved representation with current feedback")
            return None
        key = representation, observation
        reused = key in self._cache
        try:
            if reused:
                result = self._cache[key]
                self.search["cache_hits"] += 1
            else:
                output, result = self.execute(as_ref(node["operation"]), raw)
                if output is None:
                    self.induction.cap(representation, "representation_selection", False)
                    self.induction.activate(raw, source="unknown native representation outcome")
                    return None
                self._cache[key] = result
        except AthanorError:
            self.induction.cap(representation, "representation_selection", False)
            self.induction.activate(raw, source="failed native representation outcome")
            return None
        ref = self.knowledge.create(
            fields=record(
                kind="representation_interpretation",
                observation=observation,
                representation=representation,
                result=result,
                reused=reused,
                contract=node["output"],
            ),
            source="native executable interpretation receipt",
        )
        self.interpretations.append(ref)
        self.induction.cap(representation, "representation_execution", True)
        self.induction.activate(raw, source="executed representation with source observation")
        return ref

    def proposals(self, sample: Value) -> list[Ref]:
        """Construct local expressions from observed types/fields, never whole answers."""
        value = value_expression()
        expressions: list[Record] = []
        kind = family(sample)
        if kind == "record":
            for key in sorted(as_record(sample))[:8]:
                expressions.append(choose(has_field(value, key), field(value, key), literal(None)))
        if kind == "raster":
            expressions.append(structural("raster_graph", value))
        if kind == "graph":
            expressions.extend((structural("groups", value), structural("nodes", value)))
        if kind in {"graph", "sequence"}:
            rows = collection(sample)
            if rows and all(isinstance(v, Record) for v in rows):
                names = sorted(set.intersection(*(set(as_record(v)) for v in rows)))[:8]
                for name in names:
                    if all(not isinstance(as_record(v)[name], (Record, tuple, Ref)) for v in rows):
                        expressions.append(
                            structural(
                                "relate",
                                value,
                                binary("eq", role_property(name), role_property(name, "peer")),
                            )
                        )
                numeric = numeric_fields(rows)
                if numeric:
                    # Derived coordinate/measurement systems are assembled from the
                    # actual observed keys. No coordinate-name or object-role oracle.
                    for keys in list(itertools.combinations(numeric, 2))[:6] + [
                        (n,) for n in numeric
                    ]:
                        for scaled in (False, True):
                            lows = [
                                structural("minimum", structural("map", value, role_property(n)))
                                for n in keys
                            ]
                            spans = [
                                binary(
                                    "sub",
                                    structural(
                                        "maximum", structural("map", value, role_property(n))
                                    ),
                                    lo,
                                )
                                for n, lo in zip(keys, lows, strict=True)
                            ]
                            # Sequence construction is already native; the maximum span
                            # preserves uniform scale, rather than each axis independently.
                            scale = structural("maximum", sequence(*spans))
                            properties = {}
                            for index, (name, lo) in enumerate(zip(keys, lows, strict=True)):
                                delta = binary("sub", role_property(name), lo)
                                properties[f"r{index}"] = (
                                    choose(
                                        binary("eq", scale, literal(0)),
                                        literal(0),
                                        binary("div", delta, scale),
                                    )
                                    if scaled
                                    else delta
                                )
                            transformed = structural("map", value, record_expression(properties))
                            # Stable lexicographic order is composed out of native stable sorts.
                            for index in reversed(range(len(keys))):
                                transformed = structural(
                                    "sort", transformed, role_property(f"r{index}")
                                )
                            expressions.append(transformed)
            if kind == "sequence" and rows and all(isinstance(v, Fraction) for v in rows):
                expressions.append(structural("sort", value, variable("item")))
                low = structural("minimum", value)
                high = structural("maximum", value)
                for origin in (literal(0), low):
                    span = binary("sub", high, origin)
                    normalized = structural(
                        "map",
                        value,
                        choose(
                            binary("eq", span, literal(0)),
                            literal(0),
                            binary("div", binary("sub", variable("item"), origin), span),
                        ),
                    )
                    expressions.append(structural("sort", normalized, variable("item")))
            expressions.append(structural("count", value))
            if kind == "sequence" and rows and all(family(v) == "graph" for v in rows):
                expressions.append(structural("map", value, structural("count", variable("item"))))
                grouped_rows = [n for g in rows for n in collection(g)]
                numeric = numeric_fields(grouped_rows)
                for keys in list(itertools.combinations(numeric, 2))[:6]:
                    props = {}
                    for index, name in enumerate(keys):
                        measurements = structural("map", variable("item"), role_property(name))
                        props[f"r{index}"] = binary(
                            "div",
                            binary(
                                "add",
                                structural("minimum", measurements),
                                structural("maximum", measurements),
                            ),
                            literal(2),
                        )
                    expressions.append(structural("map", value, record_expression(props)))
            if kind == "sequence" and rows and all(isinstance(v, tuple) for v in rows):
                expressions.append(structural("flatten", value))
        reused = [
            as_ref(self.knowledge.get(f).fields["operation"])
            for f in self.fragments
            if self.knowledge.get(f).fields["status"] != "qualified"
            and self.accepts(self.knowledge.get(f).fields["input"], sample)
        ]
        return list(
            dict.fromkeys(
                [
                    *reused,
                    *(
                        self.operator(e, source="observed-type representation proposal")
                        for e in expressions
                    ),
                ]
            )
        )

    def candidate(self, chain: tuple[Ref, ...]) -> Candidate:
        self.induction.install()
        assert self.induction.category is not None
        return self.observer.processes.structural_call(self.induction.category, chain)

    def assess(self, candidate: Candidate, evidence: tuple[Ref, ...]) -> tuple[Value, list[Ref]]:
        receipts: list[Ref] = []
        sample: Value = None
        for witness in evidence:
            fields = self.knowledge.get(witness).fields
            a, b = as_ref(fields["left"]), as_ref(fields["right"])
            expected = fields["equivalent"]
            try:
                left, lref = self.execute(candidate.operation, self.raw(a))
                right, rref = self.execute(candidate.operation, self.raw(b))
                receipts.extend((lref, rref))
                if sample is None:
                    sample = left
                valid = (
                    self.sufficient(a)
                    and self.sufficient(b)
                    and left is not None
                    and right is not None
                )
                predicted = exact_equal(left, right) if valid else None
                error = float(predicted is None or predicted != expected)
                candidate.errors.append(error)
                candidate.proposal_errors.append(
                    discrepancy(left, right) if expected is True and valid else error
                )
            except AthanorError:
                candidate.errors.append(1)
                candidate.proposal_errors.append(1)
        return sample, receipts

    def invent(
        self,
        pairs: tuple[tuple[Value | Ref, Value | Ref, bool], ...],
        trainer: GroundedTrainer,
        *,
        max_depth: int = 6,
        max_candidates: int = 96,
        beam: int = 8,
    ) -> Ref | None:
        if (
            not 1 <= len(pairs) <= 24
            or type(max_depth) is not int
            or not 1 <= max_depth <= 6
            or type(max_candidates) is not int
            or not 1 <= max_candidates <= 256
            or type(beam) is not int
            or not 1 <= beam <= 32
            or any(type(eq) is not bool for _, _, eq in pairs)
        ):
            raise ValidationError("Representation inquiry requires bounded measured pairs")
        witnesses = tuple(
            self.knowledge.create(
                fields=record(
                    kind="representation_evidence",
                    left=self.datum(a),
                    right=self.datum(b),
                    equivalent=eq,
                ),
                source="measured correspondence outcome; representation not supplied",
            )
            for a, b, eq in pairs
        )
        self.evidence.extend(witnesses)
        raw = self.raw(as_ref(self.knowledge.get(witnesses[0]).fields["left"]))
        initial = self.induction.signature(raw)
        if any(
            self.induction.signature(self.raw(as_ref(self.knowledge.get(w).fields[k]))) != initial
            for w in witnesses
            for k in ("left", "right")
        ):
            raise ValidationError("One representation inquiry requires compatible raw interfaces")
        self.search.update(
            proposed=0,
            incompatible=0,
            max_candidates=max_candidates,
            max_depth=max_depth,
            beam=beam,
        )
        self.active, self.selected = (), None
        self.induction.activate(raw, source="representation invention from measured differences")
        identity = self.operator(value_expression(), source="uninterpreted structural baseline")
        frontier = [self.candidate((identity,))]
        proposals: list[Candidate] = []
        native: dict[Ref, Ref] = {}
        seen = set()
        for depth in range(max_depth + 1):
            level = []
            for prefix in frontier:
                if len(proposals) >= max_candidates:
                    break
                key = repr(prefix.dependencies)
                if key in seen:
                    continue
                seen.add(key)
                sample, receipts = self.assess(prefix, witnesses)
                prefix.status = (
                    "qualified"
                    if max(prefix.errors, default=1) > 0
                    else "supported"
                    if (
                        len(
                            {
                                (
                                    repr(self.raw(as_ref(self.knowledge.get(w).fields["left"]))),
                                    repr(self.raw(as_ref(self.knowledge.get(w).fields["right"]))),
                                )
                                for w in witnesses
                            }
                        )
                        >= 3
                        and any(eq is False for _, _, eq in pairs)
                        and any(eq is True for _, _, eq in pairs)
                    )
                    else "hypothesized"
                )
                ref = self.knowledge.create(
                    fields=record(
                        kind="executable_representation",
                        operation=prefix.operation,
                        hypothesis=prefix.hypothesis,
                        input=initial,
                        output=self.induction.signature(sample),
                        constituents=prefix.dependencies,
                        dependencies=(),
                        status=prefix.status,
                        support=witnesses,
                        errors=tuple(exact(v) for v in prefix.errors),
                        partial_errors=tuple(exact(v) for v in prefix.proposal_errors),
                        predictions=tuple(receipts),
                        contradictions=(),
                        outcomes=(),
                        conditions="typed immutable observation; complete measured support only",
                        limitations="bounded proposals; equality is qualified by measured context",
                        calls=prefix.calls,
                        score=0,
                    ),
                    source="constructed executable interpretation",
                )
                self.representations.append(ref)
                proposals.append(prefix)
                native[prefix.operation] = ref
                self.search["proposed"] += 1
                if sample is not None:
                    level.append((prefix, sample))
            if not level or len(proposals) >= max_candidates:
                break
            before = trainer.updates
            self.induction.learn(trainer, [c for c, _ in level], steps=40, proposal=True)
            self.neural_updates += trainer.updates - before
            ranked = self.induction.rank([c for c, _ in level], proposal=True)
            samples = {c.operation: v for c, v in level}
            if any(c.status == "supported" for c in ranked):
                break
            if depth == max_depth - 1:
                break
            next_candidates = []
            # Existing fragments get first proposal opportunity. Native type preflight
            # uses the actual intermediate structure before constructing full call trees.
            for c in ranked[:beam]:
                for op in self.proposals(samples[c.operation]):
                    if len(c.dependencies) >= 6 or op in c.dependencies:
                        self.search["incompatible"] += 1
                        continue
                    try:
                        projected, _ = self.execute(op, samples[c.operation])
                        if projected is None or exact_equal(projected, samples[c.operation]):
                            self.search["incompatible"] += 1
                            continue
                    except AthanorError:
                        self.search["incompatible"] += 1
                        continue
                    next_candidates.append(self.candidate((*c.dependencies, op)))
                    if len(next_candidates) >= max_candidates - len(proposals):
                        break
                if len(next_candidates) >= max_candidates - len(proposals):
                    break
            frontier = next_candidates
        before = trainer.updates
        self.induction.learn(trainer, proposals, steps=120)
        self.neural_updates += trainer.updates - before
        ranked = self.induction.rank(proposals)
        for c in ranked:
            ref = native[c.operation]
            updated = self.induction.revise(
                ref, record(dict(self.knowledge.get(ref).fields), score=exact(c.score))
            )
            self.representations[self.representations.index(ref)] = updated
            native[c.operation] = updated
        fitting = [native[c.operation] for c in ranked if c.status in {"supported", "hypothesized"}]
        self.active = tuple(fitting[:4])
        self.alternatives = tuple(fitting)
        self.selected = (
            native[ranked[0].operation] if ranked and ranked[0].status == "supported" else None
        )
        self.status = "supported" if self.selected else "unknown"
        if self.selected:
            node = self.knowledge.get(self.selected).fields
            op = as_ref(node["operation"])
            if op not in self.induction.library:
                self.induction.library.append(op)
            self.induction.cap(self.selected, "representation_selection", True)
            self.retain_fragments(self.selected)
        self.induction.activate(
            raw, source="qualified representation alternatives and neural selection"
        )
        return self.selected

    def retain_fragments(self, representation: Ref) -> None:
        node = self.knowledge.get(representation).fields
        chain = tuple(as_ref(v) for v in as_tuple(node["constituents"]))
        # A reusable suffix is independent of observation identities. It is provisional
        # until a distinct acquired representation actually reuses it.
        for start in range(1, max(1, len(chain) - 1)):
            suffix = chain[start:]
            if len(suffix) < 2:
                continue
            existing = next(
                (
                    f
                    for f in self.fragments
                    if self.knowledge.get(f).fields["constituents"] == suffix
                ),
                None,
            )
            if existing is None:
                c = self.candidate(suffix)
                first_evidence = self.knowledge.get(as_ref(as_tuple(node["support"])[0])).fields
                witness = as_ref(first_evidence["left"])
                prefix = self.candidate(chain[:start])
                intermediate, _ = self.execute(prefix.operation, self.raw(witness))
                derived, result = self.execute(c.operation, intermediate)
                fragment = self.knowledge.create(
                    fields=record(
                        kind="representation_fragment",
                        operation=c.operation,
                        input=self.interface_contract(suffix, intermediate),
                        output=self.induction.signature(derived),
                        result=result,
                        constituents=suffix,
                        dependencies=(representation,),
                        status="provisional",
                        support=(representation,),
                        contradictions=(),
                        savings=len(suffix) - 1,
                    ),
                    source="reopenable shared native procedure fragment",
                )
                self.fragments.append(fragment)
            else:
                fields = self.knowledge.get(existing).fields
                support = tuple(dict.fromkeys((*as_tuple(fields["support"]), representation)))
                updated = self.induction.revise(
                    existing,
                    record(
                        dict(fields),
                        support=support,
                        status="supported" if len(support) >= 2 else "provisional",
                    ),
                )
                self.fragments[self.fragments.index(existing)] = updated

        for fragment in tuple(self.fragments):
            fields = self.knowledge.get(fragment).fields
            if fields["operation"] in chain and representation not in as_tuple(fields["support"]):
                updated = self.induction.revise(
                    fragment,
                    record(
                        dict(fields),
                        support=(*as_tuple(fields["support"]), representation),
                        status="supported",
                    ),
                )
                self.fragments[self.fragments.index(fragment)] = updated

    def relationship(self, left: Value | Ref, right: Value | Ref) -> bool | None:
        a, b = self.datum(left), self.datum(right)
        predictions = []
        results: list[Ref] = []
        for ref in self.alternatives:
            if self.knowledge.get(ref).fields["status"] == "qualified":
                continue
            left_result = self.interpret(ref, a)
            r = self.interpret(ref, b)
            predictions.append(
                None
                if left_result is None or r is None
                else exact_equal(self.raw(left_result), self.raw(r))
            )
            if left_result is not None and r is not None:
                results.extend((left_result, r))
        result = (
            predictions[0]
            if predictions and all(v == predictions[0] for v in predictions)
            else None
        )
        self.history.append(
            self.knowledge.create(
                fields=record(
                    kind="representation_relationship",
                    left=a,
                    right=b,
                    alternatives=self.alternatives,
                    predictions=tuple(predictions),
                    result=result,
                    interpretations=tuple(results),
                ),
                source="competing executable predictions",
            )
        )
        for ref in self.alternatives:
            self.induction.cap(ref, "relationship_recognition", result is not None)
        self.induction.activate(
            self.raw(a), source="competing representation outcomes and feedback"
        )
        return result

    def revise(self, left: Value | Ref, right: Value | Ref, equivalent: bool) -> None:
        if type(equivalent) is not bool:
            raise ValidationError("Representation feedback requires a measured boolean")
        a, b = self.datum(left), self.datum(right)
        evidence = self.knowledge.create(
            fields=record(kind="representation_evidence", left=a, right=b, equivalent=equivalent),
            source="new measured representation test",
        )
        self.evidence.append(evidence)
        for ref in tuple(self.alternatives):
            left_result, r = self.interpret(ref, a), self.interpret(ref, b)
            if left_result is None or r is None:
                continue
            correct = exact_equal(self.raw(left_result), self.raw(r)) == equivalent
            node = self.knowledge.get(ref).fields
            support = (*as_tuple(node["support"]), evidence)
            errors = (*as_tuple(node["errors"]), Fraction(0 if correct else 1))
            outcomes = [self.knowledge.get(as_ref(w)).fields for w in support]
            distinct = {
                (repr(self.raw(as_ref(w["left"]))), repr(self.raw(as_ref(w["right"]))))
                for w in outcomes
            }
            supported = (
                len(distinct) >= 3
                and any(w["equivalent"] is True for w in outcomes)
                and any(w["equivalent"] is False for w in outcomes)
                and not any(errors)
            )
            updated = self.induction.revise(
                ref,
                record(
                    dict(node),
                    support=support,
                    errors=errors,
                    outcomes=(*as_tuple(node["outcomes"]), (left_result, r, correct)),
                    contradictions=as_tuple(node["contradictions"])
                    if correct
                    else (*as_tuple(node["contradictions"]), evidence),
                    status="supported"
                    if correct and supported
                    else node["status"]
                    if correct
                    else "qualified",
                ),
            )
            self.representations[self.representations.index(ref)] = updated
            self._cache = {
                (updated if representation == ref else representation, observation): result
                for (representation, observation), result in self._cache.items()
                if representation != ref or correct
            }
            self.active = tuple(updated if r == ref else r for r in self.active)
            self.alternatives = tuple(updated if r == ref else r for r in self.alternatives)
            if self.selected == ref:
                self.selected = updated if correct else None
            self.induction.cap(updated, "representation_revision", correct)
            if not correct:
                self.qualify_dependents(ref, evidence)
        self.active = tuple(
            r for r in self.active if self.knowledge.get(r).fields["status"] != "qualified"
        )
        self.alternatives = tuple(
            r for r in self.alternatives if self.knowledge.get(r).fields["status"] != "qualified"
        )
        candidates = [
            Candidate(
                as_ref(self.knowledge.get(r).fields["operation"]),
                r,
                value_expression(),
                (),
                int(cast(Fraction, self.knowledge.get(r).fields["calls"])),
                [
                    float(cast(Fraction, v))
                    for v in as_tuple(self.knowledge.get(r).fields["errors"])
                ],
            )
            for r in self.alternatives
        ]
        ranked = self.induction.rank(candidates)
        self.active = tuple(c.hypothesis for c in ranked[:4])
        self.selected = (
            ranked[0].hypothesis
            if ranked and self.knowledge.get(ranked[0].hypothesis).fields["status"] == "supported"
            else None
        )
        self.status = "supported" if self.selected else "unknown"
        self.induction.activate(
            self.raw(a), source="representation revision from measured feedback"
        )

    def qualify_dependents(self, ref: Ref, evidence: Ref) -> None:
        node = self.knowledge.get(ref).fields
        self.induction.revoke(as_ref(node["operation"]), evidence)
        for pool in (self.concepts, self.fragments, self.representations):
            for old in tuple(pool):
                fields = self.knowledge.get(old).fields
                if fields["status"] != "qualified" and ref in as_tuple(fields["dependencies"]):
                    updated = self.induction.revise(
                        old,
                        record(
                            dict(fields),
                            status="qualified",
                            contradictions=(*as_tuple(fields["contradictions"]), evidence),
                        ),
                    )
                    pool[pool.index(old)] = updated
                    self.qualify_dependents(old, evidence)

    def form_concept(self, representation: Ref, observations: tuple[Value | Ref, ...]) -> Ref:
        if not 1 <= len(observations) <= 24:
            raise ValidationError("Derived concepts require bounded observations")
        refs = tuple(self.datum(v) for v in observations)
        outputs = tuple(self.interpret(representation, v) for v in refs)
        valid = [r for r in outputs if r is not None]
        if not valid or any(not exact_equal(self.raw(valid[0]), self.raw(r)) for r in valid):
            raise ValidationError(
                "Concept instances must share an observed operational relationship"
            )
        prototype = as_ref(self.knowledge.get(valid[0]).fields["result"])
        represent = as_ref(self.knowledge.get(representation).fields["operation"])
        expression = binary(
            "eq", field(call(represent, argument(0)), "value"), field(literal(prototype), "value")
        )
        operation = self.operator(expression, source="unnamed concept over acquired interpretation")
        supported = (
            len(valid) == len(refs)
            and len(set(refs)) >= 3
            and (self.knowledge.get(representation).fields["status"] == "supported")
        )
        concept = self.knowledge.create(
            fields=record(
                kind="representation_concept",
                operation=operation,
                dependencies=(representation,),
                representation=representation,
                prototype=prototype,
                support=refs,
                status="supported" if supported else "provisional",
                contradictions=(),
                outcomes=(),
                roles=(
                    record(name="input", type="observation"),
                    record(name="pattern", type="derived_structure"),
                ),
            ),
            source="recurring operational distinction without a supplied concept name",
        )
        self.concepts.append(concept)
        self.induction.cap(concept, "distinction_construction", supported)
        self.induction.activate(
            self.raw(refs[-1]), source="derived operational concept and evidence"
        )
        return concept

    def recognize(
        self, concept: Ref, value: Value | Ref, *, expected: bool | None = None
    ) -> bool | None:
        if concept not in self.concepts:
            raise ValidationError("Use a current derived concept reference")
        observation = self.datum(value)
        node = self.knowledge.get(concept).fields
        representation = self.current(as_ref(node["representation"]))
        valid = node["status"] != "qualified" and self.compatible(
            representation, self.raw(observation)
        )
        result = None
        if valid and self.sufficient(observation):
            try:
                output, _ = self.execute(as_ref(node["operation"]), self.raw(observation))
                if type(output) is bool:
                    result = output
            except AthanorError:
                pass
        receipt = self.knowledge.create(
            fields=record(
                kind="representation_concept_outcome",
                concept=concept,
                observation=observation,
                predicted=result,
                expected=expected,
            ),
            source="actual derived concept execution",
        )
        self.history.append(receipt)
        self.induction.cap(concept, "relationship_recognition", result is not None)
        if expected is not None:
            self.induction.cap(concept, "representation_transfer", result == expected)
            if result is not None and result != expected:
                updated = self.induction.revise(
                    concept,
                    record(
                        dict(node),
                        status="qualified",
                        contradictions=(*as_tuple(node["contradictions"]), receipt),
                    ),
                )
                self.concepts[self.concepts.index(concept)] = updated
                self.qualify_dependents(concept, receipt)
        self.induction.activate(
            self.raw(observation), source="derived concept outcome and feedback"
        )
        return result

    def compose(self, first: Ref, second: Ref, observations: tuple[Value | Ref, ...]) -> Ref:
        if first not in self.representations or second not in self.representations:
            raise ValidationError("Composition requires current executable representations")
        if not 1 <= len(observations) <= 24:
            raise ValidationError("Composition requires bounded observations")
        a, b = self.knowledge.get(first).fields, self.knowledge.get(second).fields
        if a["status"] == "qualified" or b["status"] == "qualified" or a["output"] != b["input"]:
            raise ValidationError("Representation output/input contracts must compose")
        c = self.candidate((as_ref(a["operation"]), as_ref(b["operation"])))
        refs = tuple(self.datum(v) for v in observations)
        results = tuple(self.execute(c.operation, self.raw(v))[1] for v in refs)
        if any(self.knowledge.get(r).fields["value"] is None for r in results):
            raise ValidationError("Unknown composite result cannot support an abstraction")
        ref = self.knowledge.create(
            fields=record(
                kind="executable_representation",
                operation=c.operation,
                hypothesis=c.hypothesis,
                input=a["input"],
                output=b["output"],
                constituents=c.dependencies,
                dependencies=(first, second),
                status="provisional",
                support=(),
                errors=(),
                partial_errors=(),
                predictions=results,
                contradictions=(),
                outcomes=(),
                conditions="composed pinned representation contracts",
                limitations="successful execution does not establish unlimited generality",
                calls=2,
                score=0,
            ),
            source="higher-order reusable executable interpretation",
        )
        self.representations.append(ref)
        self.induction.library.append(c.operation)
        self.induction.cap(ref, "abstraction_execution", True)
        self.active, self.alternatives, self.selected = (ref,), (ref,), ref
        self.status = (
            "supported" if self.knowledge.get(ref).fields["status"] == "supported" else "unknown"
        )
        self.induction.activate(
            self.raw(refs[-1]), source="composed interpretation remains provisional"
        )
        return ref

    def choose_test(
        self, pairs: tuple[tuple[Value | Ref, Value | Ref], ...]
    ) -> tuple[Ref, Ref] | None:
        if not 1 <= len(pairs) <= 32:
            raise ValidationError("Active representation tests require bounded alternatives")
        from .discovery import priority_vectors

        rows, options = [], []
        for left, right in pairs:
            a, b = self.datum(left), self.datum(right)
            predictions = []
            for ref in self.alternatives:
                left_result, r = self.interpret(ref, a), self.interpret(ref, b)
                predictions.append(
                    None
                    if left_result is None or r is None
                    else exact_equal(self.raw(left_result), self.raw(r))
                )
            diversity = len(set(predictions)) - 1 if None not in predictions else 0
            rows.append((a, b, tuple(predictions), diversity))
            options.append(
                record(
                    uncertainty=1,
                    relevance=1,
                    benefit=max(0, diversity),
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
        informative = [index for index, row in enumerate(rows) if row[3] > 0]
        selected = max(informative, key=lambda index: float(scores[index])) if informative else 0
        a, b, chosen_predictions, diversity = rows[selected]
        ref = self.knowledge.create(
            fields=record(
                kind="representation_test_proposal",
                options=tuple(record(left=x, right=y, predictions=p) for x, y, p, _ in rows),
                selected=selected if diversity else None,
                predictions=chosen_predictions,
                scores=tuple(exact(float(v)) for v in scores),
                action="request_pair_outcome",
                status="proposed" if diversity else "unknown",
            ),
            source="shared recurrent informative inquiry selection",
        )
        self.history.append(ref)
        self.observer.discovery.history.append(ref)
        self.induction.cap(ref, "informative_representation_test", bool(diversity))
        self.induction.activate(self.raw(a), source="informative representation test proposal")
        return (a, b) if diversity else None

    def solve(
        self,
        value: Value | Ref,
        conditions: tuple[Record, ...],
        trainer: GroundedTrainer,
        *,
        max_nodes: int = 64,
        max_depth: int = 3,
        operations: tuple[Ref, ...] | None = None,
    ) -> Ref | None:
        if (
            type(max_nodes) is not int
            or not 1 <= max_nodes <= 2048
            or type(max_depth) is not int
            or not 1 <= max_depth <= 3
        ):
            raise ValidationError("Representation planning requires bounded shared resources")
        observation = self.datum(value)
        if not self.sufficient(observation):
            self.status = "unknown"
            return None
        raw = self.raw(observation)
        self.induction.install()
        representation_operations = {
            self.knowledge.get(r).fields["operation"] for r in self.representations
        }
        available = (
            tuple(v for v in self.induction.available() if v not in representation_operations)
            if operations is None
            else operations
        )
        if not available or any(v not in self.induction.available() for v in available):
            raise ValidationError("Use available native goal operations")
        initial = self.induction.solve(
            raw,
            conditions,
            trainer,
            max_depth=max_depth,
            max_nodes=max(1, max_nodes // 2),
            operations=available,
        )
        remaining = max_nodes - len(self.induction.candidates)
        satisfied = initial is not None or self.induction.status == "goal_supported"
        task = self.knowledge.create(
            fields=record(
                kind="representation_task",
                observation=observation,
                conditions=conditions,
                initial_task=self.induction.tasks[-1],
                alternatives=self.active,
                operations=available,
                max_nodes=max_nodes,
                max_depth=max_depth,
                attempts=(),
                selected=initial,
                status="supported" if satisfied else "investigating",
            ),
            source="bounded representation-aware objective",
        )
        self.tasks.append(task)
        if satisfied:
            self.status = "goal_supported"
            self.induction.activate(raw, source="representation objective already satisfied")
            return initial
        attempts = []
        compatible = list(
            dict.fromkeys(
                r
                for r in (*self.active, *self.representations)
                if self.compatible(r, raw)
                and self.knowledge.get(r).fields["status"] in ("supported", "provisional")
            )
        )
        # The persistent scores came from actual shared neural selection. A fresh
        # recurrent ranking determines which interpretation gets the next trial.
        ranked = self.induction.rank(
            [
                Candidate(
                    as_ref(self.knowledge.get(r).fields["operation"]),
                    r,
                    as_record(
                        as_record(
                            as_record(
                                self.knowledge.get(
                                    as_ref(self.knowledge.get(r).fields["operation"])
                                ).fields["procedure"]
                            )["fields"]
                        )["value"]
                    ),
                    (),
                    1,
                    [0],
                )
                for r in compatible
            ]
        )
        for c in ranked[:4]:
            if remaining < 1:
                break
            representation = c.hypothesis
            op = c.operation
            if op not in self.induction.library:
                self.induction.library.append(op)
            rule = self.induction.solve(
                raw,
                conditions,
                trainer,
                max_depth=min(4, max_depth + 1),
                max_nodes=remaining,
                operations=(op, *available),
            )
            remaining -= len(self.induction.candidates)
            attempts.append(self.induction.tasks[-1])
            if rule is not None:
                self.tasks[-1] = self.induction.revise(
                    task,
                    record(
                        dict(self.knowledge.get(task).fields),
                        attempts=tuple(attempts),
                        selected=rule,
                        status="supported",
                    ),
                )
                self.induction.cap(representation, "representation_revision", True)
                self.induction.cap(representation, "goal_achievement", True)
                self.selected = representation
                self.status = "goal_supported"
                self.induction.activate(
                    raw, source="representation revision achieved measured objective"
                )
                return rule
        self.tasks[-1] = self.induction.revise(
            task,
            record(
                dict(self.knowledge.get(task).fields), attempts=tuple(attempts), status="unknown"
            ),
        )
        self.status = "unknown"
        self.induction.cap(self.tasks[-1], "unresolved_recognition", True)
        self.induction.activate(raw, source="bounded representation objective remains unresolved")
        return None

    def metadata(self) -> dict[str, Any]:
        def ref(r: Ref) -> list[str | int]:
            return [r.id, r.revision, r.namespace]

        return {
            **{
                name: [ref(r) for r in getattr(self, name)]
                for name in (
                    "representations",
                    "observations",
                    "evidence",
                    "interpretations",
                    "concepts",
                    "fragments",
                    "tasks",
                    "history",
                    "active",
                    "alternatives",
                )
            },
            "selected": None if self.selected is None else ref(self.selected),
            "status": self.status,
            "search": self.search.copy(),
            "neural_updates": self.neural_updates,
            "operators": {k: ref(v) for k, v in self._operators.items()},
            "cache": [
                [ref(a), ref(b), ref(c)]
                for (a, b), c in self._cache.items()
                if a in self.representations
            ],
        }

    def validate_native_state(self) -> None:
        """Validate executable contracts and pinned evidence without running inference."""
        checked: set[Ref] = set()

        def operation(ref: Ref) -> None:
            if ref in checked:
                return
            checked.add(ref)
            node = self.knowledge.get(ref)
            procedure = as_record(node.fields.get("procedure"))
            if (
                node.role != Role.OPERATION
                or procedure["update"] is not None
                or procedure["result"] != self.induction.category
                or as_tuple(procedure["inputs"])
                != (Endpoint(Role.DATA, self.induction.category).to_record(),)
                or set(as_record(procedure["fields"])) != {"value"}
                or procedure["members"] is not None
            ):
                raise ValidationError(
                    "Representation execution requires pure common typed unary contracts"
                )
            pending: list[Value] = [procedure["fields"]]
            while pending:
                expr = pending.pop()
                if isinstance(expr, Record):
                    if expr.get("op") == "call" and expr.get("operation") != "self":
                        operation(as_ref(expr["operation"]))
                    pending.extend(expr.values())
                elif isinstance(expr, tuple):
                    pending.extend(expr)

        owned = set(self.observations)
        for observation in self.observations:
            fields = self.knowledge.get(observation).fields
            if set(fields) != {"kind", "value", "capture", "sufficient"}:
                raise ValidationError("Malformed representation observation")
            if fields["capture"] is not None:
                capture = self.knowledge.get(as_ref(fields["capture"])).fields
                if capture.get("kind") != "representation_camera_capture":
                    raise ValidationError("Camera observation requires an attributed capture")

        def output(result: Ref, op: Ref, observation: Ref) -> None:
            node = self.knowledge.get(result)
            receipt = self.knowledge.inspect(result).receipt
            if (
                node.category != self.induction.category
                or set(node.fields) != {"value"}
                or receipt.operation != op
                or len(receipt.inputs) != 1
                or not exact_equal(
                    self.knowledge.get(receipt.inputs[0]).fields.get("value"), self.raw(observation)
                )
            ):
                raise ValidationError(
                    "Interpretation must bind its actual typed native execution receipt"
                )

        for interpretation in self.interpretations:
            fields = self.knowledge.get(interpretation).fields
            if set(fields) != {
                "kind",
                "observation",
                "representation",
                "result",
                "reused",
                "contract",
            }:
                raise ValidationError("Malformed interpretation receipt")
            parent = as_ref(fields["observation"])
            if parent not in owned or type(fields["reused"]) is not bool:
                raise ValidationError("Interpretation sources must be owned and acyclic")
            representation = self.knowledge.get(as_ref(fields["representation"])).fields
            if representation.get("kind") != "executable_representation":
                raise ValidationError("Interpretation requires a pinned executable representation")
            output(as_ref(fields["result"]), as_ref(representation["operation"]), parent)
            owned.add(interpretation)
        for evidence in self.evidence:
            fields = self.knowledge.get(evidence).fields
            if (
                set(fields) != {"kind", "left", "right", "equivalent"}
                or fields["left"] not in owned
                or fields["right"] not in owned
                or type(fields["equivalent"]) is not bool
            ):
                raise ValidationError("Representation evidence requires owned observable outcomes")
        for ref in self.representations:
            fields = self.knowledge.get(ref).fields
            if set(fields) != {
                "kind",
                "operation",
                "hypothesis",
                "input",
                "output",
                "constituents",
                "dependencies",
                "status",
                "support",
                "errors",
                "partial_errors",
                "predictions",
                "contradictions",
                "outcomes",
                "conditions",
                "limitations",
                "calls",
                "score",
            }:
                raise ValidationError("Malformed executable representation")
            op = as_ref(fields["operation"])
            operation(op)
            chain = tuple(as_ref(v) for v in as_tuple(fields["constituents"]))
            if not 1 <= len(chain) <= 6 or fields["calls"] != Fraction(len(chain)):
                raise ValidationError("Invalid representation composition size")
            expression = argument(0)
            for constituent in chain:
                operation(constituent)
                expression = call(constituent, expression)
            actual = as_record(as_record(self.knowledge.get(op).fields["procedure"])["fields"])[
                "value"
            ]
            if actual != field(expression, "value"):
                raise ValidationError("Representation must execute its declared constituents")
            support = as_tuple(fields["support"])
            errors = as_tuple(fields["errors"])
            if len(support) != len(errors) or any(v not in self.evidence for v in support):
                raise ValidationError(
                    "Representation errors must bind measured correspondence evidence"
                )
            if fields["status"] == "supported":
                witnesses = [self.knowledge.get(as_ref(v)).fields for v in support]
                distinct = {
                    (repr(self.raw(as_ref(w["left"]))), repr(self.raw(as_ref(w["right"]))))
                    for w in witnesses
                }
                if (
                    len(distinct) < 3
                    or not any(w["equivalent"] is True for w in witnesses)
                    or not any(w["equivalent"] is False for w in witnesses)
                    or any(
                        not self.sufficient(as_ref(w[k]))
                        for w in witnesses
                        for k in ("left", "right")
                    )
                ):
                    raise ValidationError(
                        "Supported representations require distinct positive and negative evidence"
                    )
                initial = as_tuple(fields["predictions"])
                later = as_tuple(fields["outcomes"])
                if len(initial) % 2 or len(initial) // 2 + len(later) != len(witnesses):
                    raise ValidationError(
                        "Supported representation predictions must cover every witness"
                    )
                for index, witness in enumerate(witnesses):
                    left, right = as_ref(witness["left"]), as_ref(witness["right"])
                    if index < len(initial) // 2:
                        a, b = as_ref(initial[index * 2]), as_ref(initial[index * 2 + 1])
                    else:
                        row = as_tuple(later[index - len(initial) // 2])
                        if len(row) != 3 or row[2] is not True:
                            raise ValidationError(
                                "Supported representation requires successful feedback"
                            )
                        a = as_ref(self.knowledge.get(as_ref(row[0])).fields["result"])
                        b = as_ref(self.knowledge.get(as_ref(row[1])).fields["result"])
                    output(a, op, left)
                    output(b, op, right)
                    if (
                        exact_equal(
                            self.knowledge.get(a).fields["value"],
                            self.knowledge.get(b).fields["value"],
                        )
                        != witness["equivalent"]
                    ):
                        raise ValidationError(
                            "Representation support contradicts its native predictions"
                        )
        for ref in (*self.concepts, *self.fragments):
            fields = self.knowledge.get(ref).fields
            operation(as_ref(fields["operation"]))
            if fields["status"] not in {"supported", "provisional", "qualified"}:
                raise ValidationError("Invalid acquired abstraction evidence status")
        for (cached_representation, observation), result in self._cache.items():
            output(
                result,
                as_ref(self.knowledge.get(cached_representation).fields["operation"]),
                observation,
            )

    @classmethod
    def restore(cls, observer: GroundedObserver, data: dict[str, Any]) -> Representations:
        result = cls(observer)
        if set(data) != set(result.metadata()) or data["status"] not in (
            "idle",
            "supported",
            "unknown",
            "goal_supported",
        ):
            raise ValidationError("Invalid representation metadata")

        def ref(raw: Any) -> Ref:
            if (
                type(raw) is not list
                or len(raw) != 3
                or type(raw[0]) is not str
                or type(raw[1]) is not int
                or type(raw[2]) is not str
            ):
                raise ValidationError("Invalid representation reference")
            r = Ref(raw[0], raw[1], raw[2])
            observer.knowledge.get(r)
            return r

        kinds = {
            "representations": "executable_representation",
            "observations": "representation_observation",
            "evidence": "representation_evidence",
            "interpretations": "representation_interpretation",
            "concepts": "representation_concept",
            "fragments": "representation_fragment",
            "tasks": "representation_task",
        }
        for name in (*kinds, "history", "active", "alternatives"):
            if type(data[name]) is not list or len(data[name]) > 65536:
                raise ValidationError("Representation history exceeds import budget")
            rows = [ref(v) for v in data[name]]
            if name in kinds and any(
                observer.knowledge.get(r).fields.get("kind") != kinds[name] for r in rows
            ):
                raise ValidationError("Representation participant kind mismatch")
            setattr(result, name, tuple(rows) if name in ("active", "alternatives") else rows)
        result.selected = None if data["selected"] is None else ref(data["selected"])
        result.status = data["status"]
        if any(r not in result.representations for r in (*result.active, *result.alternatives)) or (
            result.selected is not None and result.selected not in result.representations
        ):
            raise ValidationError("Invalid active representation")
        if (
            type(data["search"]) is not dict
            or set(data["search"]) != set(result.search)
            or any(type(v) is not int or v < 0 for v in data["search"].values())
            or not 1 <= data["search"]["max_candidates"] <= 256
            or not 1 <= data["search"]["max_depth"] <= 6
            or not 1 <= data["search"]["beam"] <= 32
            or type(data["neural_updates"]) is not int
            or data["neural_updates"] < 0
        ):
            raise ValidationError("Invalid representation search state")
        result.search = data["search"].copy()
        result.neural_updates = data["neural_updates"]
        if type(data["operators"]) is not dict or len(data["operators"]) > 4096:
            raise ValidationError("Invalid representation operation cache")
        for key, raw in data["operators"].items():
            operation = ref(raw)
            fields = observer.knowledge.get(operation).fields
            p = as_record(fields["procedure"])
            expression = as_record(as_record(p["fields"])["value"])
            if (
                repr(expression) != key
                or p["update"] is not None
                or len(as_tuple(p["inputs"])) != 1
            ):
                raise ValidationError("Representation cache must bind exact pure unary expressions")
            result._operators[key] = operation
        for r in result.representations:
            fields = observer.knowledge.get(r).fields
            if fields["status"] not in ("qualified", "supported", "hypothesized", "provisional"):
                raise ValidationError("Invalid representation evidence status")
            op = observer.knowledge.get(as_ref(fields["operation"]))
            if op.role != Role.OPERATION or as_record(op.fields["procedure"])["update"] is not None:
                raise ValidationError("Representation must execute a pure native operation")
            errors = as_tuple(fields["errors"])
            if any(not isinstance(v, Fraction) or not 0 <= v <= 1 for v in errors):
                raise ValidationError("Invalid representation measured errors")
            if fields["status"] == "supported" and (len(errors) < 3 or any(errors)):
                raise ValidationError(
                    "Supported representations require measured consistent evidence"
                )
        if type(data["cache"]) is not list or len(data["cache"]) > 65536:
            raise ValidationError("Invalid interpretation result cache")
        for row in data["cache"]:
            if type(row) is not list or len(row) != 3:
                raise ValidationError("Invalid interpretation cache binding")
            a, b, c = (ref(v) for v in row)
            if a not in result.representations or b not in (
                *result.observations,
                *result.interpretations,
            ):
                raise ValidationError("Invalid interpretation cache participant")
            if (a, b) in result._cache or "value" not in observer.knowledge.get(c).fields:
                raise ValidationError("Invalid interpretation cache output")
            result._cache[a, b] = c
        for r in result.observations:
            node = observer.knowledge.get(r).fields
            validate_value(node["value"])
            if type(node["sufficient"]) is not bool:
                raise ValidationError("Invalid observation sufficiency")
            result._observed[repr((node["value"], node["capture"], node["sufficient"]))] = r
        result.validate_native_state()
        return result
