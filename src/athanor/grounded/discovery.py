"""Native empirical induction and model-ranked inquiry over restricted observations.

This module has no world authority. Induction is a bounded family of zero and
conditional constant effects, with a permanently available interface alternative.
It does not contain the world's physical coefficients or mobility properties.
"""

import math
from dataclasses import dataclass
from dataclasses import field as data_field
from fractions import Fraction
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import Tensor

from athanor import Endpoint, Procedure, Record, Ref, Role, ValidationError, record
from athanor.neural import Decision, Proposal
from athanor.neural.encoding import structural_features
from athanor.programs import argument, binary, choose, field, item, literal
from athanor.values import as_ref
from athanor.world import SensorFrame
from athanor.world.engine import ACTIONS, exact
from athanor.world.geometry import rotate

from .perception import descriptor_distance, perceive, pixels

if TYPE_CHECKING:
    from .runtime import GroundedObserver


@dataclass(slots=True)
class Estimate:
    key: str
    channel: str
    action: str
    appearance: tuple[float, ...]
    scope: str
    samples: list[tuple[float, ...]] = data_field(default_factory=list)
    evidence: list[Ref] = data_field(default_factory=list)
    conflicts: list[Ref] = data_field(default_factory=list)
    hypothesis: Ref | None = None
    relationship: Ref | None = None
    operation: Ref | None = None
    status: str = "unknown"
    excluded: int = 0
    contexts: list[tuple[float, ...]] = data_field(default_factory=list)

    @property
    def mean(self) -> tuple[float, ...]:
        if not self.samples:
            return (0.0,) * {"object": 3, "self": 4, "interface": 2}[self.channel]
        return tuple(
            sum(row[i] for row in self.samples) / len(self.samples)
            for i in range(len(self.samples[0]))
        )

    @property
    def dispersion(self) -> float:
        return max(
            (max(abs(a - b) for a, b in zip(row, self.mean, strict=True)) for row in self.samples),
            default=0.0,
        )


def experiment_features(
    values: Record, *, kind: str = "experimental_priority"
) -> dict[str, object]:
    """Shared features omit action names/IDs: priority cannot memorize a scene solution."""
    keys = ("uncertainty", "relevance", "benefit", "reliability", "cost", "risk")
    result: dict[str, object] = {"kind": kind}
    for key in keys:
        number = float(cast(Fraction, values[key]))
        result[key] = exact(number * 32)
        # General measurement distinctions; these do not choose an experiment.
        for threshold in (0.0, 0.1, 0.25, 0.5, 0.75):
            result[f"{key}_above_{threshold}"] = number > threshold
    return result


def priority_vectors(
    model: Any, options: list[Record], *, kind: str = "experimental_priority"
) -> Tensor:
    # Compact interface views follow Goal 004's intent encoding convention.
    from athanor import Structure

    features = torch.tensor(
        [
            structural_features(
                Structure(
                    Ref("s00000001"),
                    Role.DATA,
                    None,
                    record(experiment_features(values, kind=kind)),
                    (),
                )
            )
            for values in options
        ],
        device=model.device,
    )
    return cast(Tensor, model.encoder(features)).tanh()


class Discovery:
    """Fast adaptation, conditional native consolidation, and shared neural inquiry."""

    def __init__(self, observer: GroundedObserver) -> None:
        self.observer = observer
        self.status = "stopped"
        self.budget = 0
        self.spent = 0
        self.allowed: tuple[str, ...] = ACTIONS
        self.focus: str | None = None
        self.estimates: dict[str, Estimate] = {}
        self.prototypes: list[tuple[float, ...]] = []
        self.history: list[Ref] = []
        self.active: tuple[Ref, ...] = ()
        self.options: tuple[Ref, ...] = ()
        self.selected: Ref | None = None
        self.last_scores: tuple[float, ...] = ()
        self.gain = 0.0
        self.priority_trained = False
        self.neural_updates = 0
        self.last_learned_tick = -1
        self.consolidate_every = 4
        self.learning = True
        self.session = observer.observer.self_ref
        self.reviser: Ref | None = None

    def start(
        self, budget: int = 24, *, focus: str | None = None, allowed: tuple[str, ...] = ACTIONS
    ) -> None:
        if (
            type(budget) is not int
            or not 1 <= budget <= 10000
            or not allowed
            or len(set(allowed)) != len(allowed)
            or any(action not in ACTIONS for action in allowed)
            or focus is not None
            and focus not in {"object", "self", "interface"}
        ):
            raise ValidationError("Inquiry requires a bounded budget, valid focus and actions")
        self.status, self.budget, self.spent = "running", budget, 0
        self.allowed, self.focus = allowed, focus
        self.observer.observer.active_operations = tuple(self.observer.operations.values())
        self._control()

    def pause(self) -> None:
        if self.status == "running":
            self.status = "paused"
            self._control()

    def resume(self) -> None:
        if self.status != "paused" or self.spent >= self.budget:
            raise ValidationError("Only a paused inquiry with remaining resources can resume")
        self.status = "running"
        self._control()

    def stop(self) -> None:
        self.status = "stopped"
        self._control()

    def _control(self) -> None:
        self.session = self.observer.knowledge.create(
            fields=record(
                kind="autonomous_inquiry",
                status=self.status,
                budget=self.budget,
                spent=self.spent,
                focus=self.focus,
                allowed=self.allowed,
            ),
            source="explicit inquiry resource/control transition",
        )

    def _prototype(self, appearance: tuple[float, ...]) -> int:
        matches = [
            i
            for i, prior in enumerate(self.prototypes)
            if descriptor_distance(prior, appearance) < 0.18
        ]
        if matches:
            return min(matches, key=lambda i: descriptor_distance(self.prototypes[i], appearance))
        self.prototypes.append(appearance)
        return len(self.prototypes) - 1

    def estimate(
        self,
        channel: str,
        action: str,
        appearance: tuple[float, ...] = (),
        scope: str = "measured interface",
    ) -> Estimate:
        group = self._prototype(appearance) if channel == "object" else -1
        key = f"{channel}/{action}/{group}/{scope}"
        if key not in self.estimates:
            value = Estimate(
                key, channel, action, self.prototypes[group] if group >= 0 else (), scope
            )
            self.estimates[key] = value
            self._publish(value)
        return self.estimates[key]

    def _controls(self, value: Estimate) -> Estimate | None:
        if value.channel != "object" or value.action == "wait":
            return None
        return self.estimates.get(value.key.replace(f"/{value.action}/", "/wait/", 1))

    def _publish(self, value: Estimate) -> None:
        controls = self._controls(value)
        fields = record(
            kind="discovery_hypothesis",
            channel=value.channel,
            action=value.action,
            scope=value.scope,
            appearance=tuple(exact(v) for v in value.appearance),
            status=value.status,
            observations=len(value.samples),
            excluded=value.excluded,
            prediction=tuple(exact(v) for v in value.mean),
            dispersion=exact(value.dispersion),
            alternatives=(
                record(
                    family="zero_effect",
                    prediction=(0,) * len(value.mean),
                    assessment="unknown"
                    if not value.samples
                    else "contradicted"
                    if max(abs(v) for v in value.mean)
                    > (0.06 if value.channel == "object" else 0.015)
                    else "compatible",
                ),
                record(
                    family="conditional_constant_effect",
                    prediction=tuple(exact(v) for v in value.mean),
                    assessment=value.status,
                ),
                record(
                    family="unresolved_interface_or_measurement",
                    prediction=None,
                    assessment="unresolved",
                ),
            ),
            distinguishing_conditions=record(
                command_matches=True,
                matched_stereo=True,
                scope=value.scope,
                passive_comparison=value.channel == "object" and value.action != "wait",
            ),
            support=tuple(value.evidence[-3:]),
            observed_conditions=tuple(tuple(exact(v) for v in row) for row in value.contexts[-6:]),
            contradictions=tuple(value.conflicts[-3:]),
            comparison=None if controls is None else controls.hypothesis,
            causal_status="conditional empirical prediction; competing causes may coexist",
        )
        a = self.observer.knowledge
        if self.reviser is None:
            self.reviser = a.teach_operation(
                "revise discovery hypothesis",
                Procedure(
                    (Endpoint(Role.DATA), Endpoint(Role.DATA)),
                    None,
                    {key: field(argument(1), key) for key in fields},
                    update=0,
                ),
            )
        new = a.create(fields=fields, source="restricted evidence induction")
        value.hypothesis = (
            new if value.hypothesis is None else a.execute(self.reviser, (value.hypothesis, new))
        )

    def prepare(self, frame: SensorFrame) -> tuple[Ref, ...]:
        """Activate uncertainties and generic experimental opportunities, without acting."""
        if self.status not in {"running", "paused"}:
            self.active, self.options = (), ()
            return ()
        o = self.observer
        visible = [
            track for track in o.tracks if track.status == "visible" and track.relative is not None
        ]
        # User focus constrains relevance, not the expected answer.
        options = []
        active: dict[Ref, None] = {}
        for action in self.allowed:
            own = self.estimate("self", action)
            assert own.hypothesis is not None
            active[own.hypothesis] = None
            target = min(
                visible,
                key=lambda track: math.dist(track.relative or (0, 0, 0), (0, 0, 0)),
                default=None,
            )
            estimate = own
            if self.focus == "interface":
                estimate = self.estimate(
                    "interface", action, scope="visible displacement comparison"
                )
                assert estimate.hypothesis is not None
                active[estimate.hypothesis] = None
            if target is not None and self.focus not in {"self", "interface"}:
                # Reach is an observed geometric opportunity, not a world property.
                scope = "near" if (target.relative or (0, 0, 0))[2] < 1.4 else "far"
                estimate = self.estimate("object", action, target.appearance, scope)
                assert estimate.hypothesis is not None
                active[estimate.hypothesis] = None
            uncertainty = 1 / (1 + len(estimate.samples))
            if estimate.status == "qualified":
                uncertainty = 1.0
            benefit = 1.0 if estimate.status in {"unknown", "hypothesized"} else 0.05
            if (
                estimate.channel == "object"
                and estimate.status == "supported"
                and target is not None
                and target.relative is not None
                and estimate.contexts
                and min(math.dist(target.relative, old) for old in estimate.contexts) > 0.06
            ):
                # Reuse in a new measured context deserves a prediction check.
                uncertainty, benefit = max(uncertainty, 0.5), 0.5
            control = self._controls(estimate)
            if (
                estimate.channel == "object"
                and action != "wait"
                and (control is None or len(control.samples) < 2)
            ):
                benefit *= 0.5
            attempts, successes = o.observer.capabilities.get(o.operations[action], (0, 0))
            reliability = (successes + 1) / (attempts + 2)
            # Geometry constrains contact experiments. It does not supply mobility.
            if action == "interact" and (target is None or estimate.scope != "near"):
                benefit *= 0.05
            relevance = 1.0 if self.focus is None or estimate.channel == self.focus else 0.25
            applied = None
            if estimate.status == "supported":
                state = (
                    tuple(target.relative)
                    if target is not None
                    and estimate.channel == "object"
                    and target.relative is not None
                    else (0.0, 0.0)
                    if estimate.channel == "interface"
                    else (0.0,) * 4
                )
                applied = self.predict(
                    estimate, state, appearance=() if target is None else target.appearance
                )
            ref = o.knowledge.create(
                fields=record(
                    kind="experiment_option",
                    action=action,
                    operation=o.operations[action],
                    hypothesis=estimate.hypothesis,
                    acquired_prediction=applied,
                    participant=None if target is None else target.ref,
                    uncertainty=exact(uncertainty),
                    relevance=exact(relevance),
                    benefit=exact(benefit),
                    reliability=exact(reliability),
                    cost=exact(0.1),
                    risk=exact(0.05 if action == "wait" else 0.15),
                    predictions=record(
                        zero=(0,) * len(estimate.mean),
                        fitted=tuple(exact(v) for v in estimate.mean),
                        unresolved=None,
                    ),
                    rationale="distinguish conditional effects using available interface evidence",
                ),
                source="native experimental opportunity from uncertainty and measured self-map",
            )
            options.append(ref)
        # Abstention is a first-class neural candidate with no motor execution.
        options.append(
            o.knowledge.create(
                fields=record(
                    kind="experiment_option",
                    action="abstain",
                    operation=None,
                    uncertainty=0,
                    relevance=0,
                    benefit=0,
                    reliability=1,
                    cost=0,
                    risk=0,
                    rationale="insufficient expected evidence benefit or constrained action",
                ),
                source="safe neural abstention opportunity",
            )
        )
        self.options, self.active = tuple(options), tuple(active)
        return self.active

    def rank(self) -> tuple[Tensor, Tensor]:
        output = self.observer.observer._output
        if output is None or not self.options:
            raise ValidationError("Inquiry needs an active recurrent observation")
        model = self.observer.observer.model
        values = [self.observer.knowledge.get(ref).fields for ref in self.options]
        vectors = priority_vectors(model, values)
        # Reuse the active recurrent context and shared learned scorer. No new head.
        return model._scores(vectors, output.activation.context * 0.05), vectors

    def choose(self, *, commit: bool = True) -> Decision:
        o = self.observer
        base = o.observer.propose()
        if self.status != "running" or self.spent >= self.budget:
            return Decision(Proposal(None), 1.0, base.expectation)
        with torch.no_grad():
            logits, _ = self.rank()
            index = int(logits.argmax())
            if commit:
                self.last_scores = tuple(float(v) for v in logits)
            confidence = float(logits.softmax(0)[index])
        selected = self.options[index]
        if commit:
            self.selected = selected
        values = o.knowledge.get(selected).fields
        operation = values["operation"]
        return Decision(
            Proposal(None if operation is None else as_ref(operation)), confidence, base.expectation
        )

    def _add(
        self,
        value: Estimate,
        sample: tuple[float, ...] | None,
        evidence: Ref,
        *,
        context: tuple[float, ...] = (),
    ) -> None:
        if sample is None:
            value.excluded += 1
            self._publish(value)
            return
        tolerance = 0.06 if value.channel == "object" else 0.015
        contradiction = (
            len(value.samples) >= 2
            and max(abs(a - b) for a, b in zip(sample, value.mean, strict=True)) > tolerance
        )
        if contradiction:
            value.conflicts.append(evidence)
        value.samples.append(sample)
        if context:
            value.contexts.append(context)
        value.evidence.append(evidence)
        value.status = "qualified" if value.conflicts else "hypothesized"
        controls = self._controls(value)
        enough = len(value.samples) >= 3 and value.dispersion <= tolerance and not value.conflicts
        if value.channel == "object" and value.action != "wait":
            enough = (
                enough
                and controls is not None
                and len(controls.samples) >= 2
                and not controls.conflicts
                and controls.dispersion <= tolerance
                and max(abs(v) for v in controls.mean) <= tolerance
            )
        if enough:
            value.status = "supported"
        self._publish(value)
        if value.channel == "object" and value.action == "wait" and value.conflicts:
            for dependent in self.estimates.values():
                if dependent.status == "supported" and self._controls(dependent) is value:
                    dependent.status = "qualified"
                    self._publish(dependent)
        if value.status == "supported" and value.operation is None:
            self._compile(value)
        elif value.relationship is not None and value.status != "supported":
            # Current assessment gates reuse; historical immutable procedure remains visible.
            self.gain -= 0.5

    def _compile(self, value: Estimate) -> None:
        a = self.observer.knowledge
        assert value.hypothesis is not None
        checks = [
            binary("eq", field(argument(0), "action"), literal(value.action)),
            binary("eq", field(argument(0), "scope"), literal(value.scope)),
            binary("eq", field(argument(0), "valid"), literal(True)),
        ]
        if value.channel == "object":
            distance = literal(0)
            for index, number in enumerate(value.appearance):
                difference = binary(
                    "sub", item(field(argument(0), "appearance"), index), literal(exact(number))
                )
                distance = binary("add", distance, binary("mul", difference, difference))
            checks.append(binary("lt", distance, literal(exact(0.18**2))))
            for i in range(3):
                if value.contexts:
                    lower = min(row[i] for row in value.contexts) - 0.12
                    upper = max(row[i] for row in value.contexts) + 0.12
                    checks.extend(
                        (
                            binary("le", literal(exact(lower)), field(argument(0), f"value_{i}")),
                            binary("le", field(argument(0), f"value_{i}"), literal(exact(upper))),
                        )
                    )
        condition = literal(True)
        for check in reversed(checks):
            condition = choose(check, condition, literal(False))
        fields = {
            "status": choose(condition, literal("supported"), literal("unknown")),
            "hypothesis": literal(value.hypothesis),
            "scope": literal(value.scope),
        }
        for i, number in enumerate(value.mean):
            fields[f"effect_{i}"] = choose(condition, literal(exact(number)), literal(None))
            fields[f"next_{i}"] = choose(
                condition,
                binary("add", field(argument(0), f"value_{i}"), literal(exact(number))),
                literal(None),
            )
        operation = a.teach_operation(
            f"acquired {value.channel} {value.action} conditional effect",
            Procedure((Endpoint(Role.DATA),), None, fields),
            source="compiled from repeated restricted measurements; fitted coefficients",
        )
        value.operation = operation
        value.relationship = a.create(
            fields=record(
                kind="acquired_relationship",
                channel=value.channel,
                action=value.action,
                hypothesis=value.hypothesis,
                operation=operation,
                scope=value.scope,
                coefficients=tuple(exact(v) for v in value.mean),
                evidence=tuple(value.evidence[-3:]),
                alternatives="interface, measurement and untested conditions remain open",
                confidence_basis="repeated valid consequences and passive comparisons",
            ),
            source="native operational consolidation of independently acquired evidence",
        )
        self.gain += 1.0

    def ingest(
        self,
        before: SensorFrame,
        after: SensorFrame,
        action: str,
        feedback: Ref,
        *,
        autonomous: bool,
    ) -> None:
        if self.status not in {"running", "paused"}:
            return
        self.gain = 0.0
        if autonomous:
            self.spent += 1
        evidence = self.observer.knowledge.create(
            fields=record(
                kind="investigation_evidence",
                action=action,
                executed=after.executed,
                tick=after.tick,
                feedback=feedback,
                autonomous=autonomous,
                selected=self.selected,
                valid_command=after.executed == action,
            ),
            source="actual interface receipt following experimental action",
        )
        self.history.append(evidence)
        own = self.estimate("self", action)
        old_count = len(own.samples)
        self._add(own, tuple(after.proprio[:4]) if after.executed == action else None, evidence)
        self.gain += 1 / (old_count + 1) if len(own.samples) > old_count else 0
        old = perceive(self.observer.observer.sensory, before)
        new = perceive(self.observer.observer.sensory, after)
        visual_change = float((pixels(after) - pixels(before)).abs().mean())
        body_moved = math.dist(after.proprio[:3], (0, 0, 0)) > 0.01
        sensor_evidence = self.observer.knowledge.create(
            fields=record(
                kind="sensor_comparison_evidence",
                observer=self.observer.observer.self_ref,
                tick=after.tick,
                visual_change=exact(visual_change),
                odometry_moved=body_moved,
                attribution=(
                    "odometry supports motion; interface or untextured view remains possible"
                    if body_moved and visual_change < 0.005
                    else "visual and proprioceptive consequences are separately measured"
                ),
                alternatives=("observation interface", "uninformative view", "world variation"),
            ),
            source="comparison of camera changes and independent body receipts",
        )
        interface = self.estimate("interface", action, scope="visible displacement comparison")
        self._add(
            interface,
            (float(visual_change > 0.005), float(body_moved))
            if any(p.status == "visible" for p in old) and after.executed == action
            else None,
            sensor_evidence,
        )
        self.observer.observer.self_map.append(sensor_evidence)
        for percept in old:
            if percept.status != "visible" or percept.relative is None:
                continue
            scope = "near" if percept.relative[2] < 1.4 else "far"
            value = self.estimate("object", action, percept.appearance, scope)
            matches = [
                p for p in new if descriptor_distance(p.appearance, percept.appearance) < 0.18
            ]
            old_matches = [
                p for p in old if descriptor_distance(p.appearance, percept.appearance) < 0.18
            ]
            sample = None
            if (
                len(matches) == len(old_matches) == 1
                and matches[0].status == "visible"
                and matches[0].relative is not None
                and after.executed == action
                and abs(after.proprio[3]) < 0.001
            ):
                # Pitch is not measured at this interface: rotating views are excluded.
                expected = rotate(
                    cast(
                        tuple[float, float, float],
                        tuple(
                            v - d for v, d in zip(percept.relative, after.proprio[:3], strict=True)
                        ),
                    ),
                    -after.proprio[3],
                )
                sample = tuple(v - e for v, e in zip(matches[0].relative, expected, strict=True))
                if after.proprio[5] and before.left == after.left and before.right == after.right:
                    sample = None  # stale-camera/odometry conflict, not a stationary-object fact
                if action in {"look_up", "look_down"}:
                    sample = None
            observation = self.observer.knowledge.create(
                fields=record(
                    kind="effect_measurement",
                    experiment=evidence,
                    appearance=tuple(exact(v) for v in percept.appearance),
                    scope=scope,
                    measured=None if sample is None else tuple(exact(v) for v in sample),
                    attribution=(
                        "object response or sensor error"
                        if sample is not None
                        else "unresolved sensing, identity or command"
                    ),
                ),
                source="learned stereo difference compensated by restricted odometry",
            )
            count = len(value.samples)
            self._add(value, sample, observation, context=tuple(percept.relative))
            self.gain += (1 / (count + 1)) if len(value.samples) > count else 0
        # New passive comparisons can now support prior action alternatives.
        if action == "wait":
            for value in tuple(self.estimates.values()):
                control = self._controls(value)
                if (
                    value.status == "hypothesized"
                    and len(value.samples) >= 3
                    and control is not None
                    and len(control.samples) >= 2
                    and control.dispersion <= 0.06
                    and max(abs(v) for v in control.mean) <= 0.06
                    and value.dispersion <= 0.06
                    and not value.conflicts
                ):
                    value.status = "supported"
                    self._publish(value)
                    if value.operation is None:
                        self._compile(value)
        own_record = self.observer.knowledge.create(
            fields=record(
                kind="inquiry_self_observation",
                observer=self.observer.observer.self_ref,
                uncertainty=exact(
                    sum(1 / (1 + len(v.samples)) for v in self.estimates.values())
                    / max(1, len(self.estimates))
                ),
                evidence_gain=exact(self.gain),
                spent=self.spent,
                remaining=self.budget - self.spent,
                supported=sum(v.status == "supported" for v in self.estimates.values()),
                qualified=sum(v.status == "qualified" for v in self.estimates.values()),
                neural_updates=self.neural_updates,
                capability=own.hypothesis,
            ),
            source="dynamic investigation reliability and uncertainty from actual outcomes",
        )
        self.observer.observer.self_map.append(own_record)
        self.observer.observer.self_map = self.observer.observer.self_map[
            -self.observer.observer.model.config.history_limit :
        ]
        if self.spent >= self.budget:
            self.status = "exhausted"
            self._control()

    def predict(
        self,
        value: Estimate,
        state: tuple[float, ...],
        *,
        appearance: tuple[float, ...] = (),
        scope: str | None = None,
    ) -> Ref | None:
        """Execute acquired knowledge; current evidence assessment gates historical procedures."""
        if value.operation is None or value.status != "supported":
            return None
        if len(state) != len(value.mean):
            raise ValidationError("Acquired procedure input dimension mismatch")
        inputs: dict[str, object] = dict(
            action=value.action,
            scope=scope or value.scope,
            valid=True,
            appearance=tuple(exact(v) for v in appearance),
        )
        inputs.update({f"value_{i}": exact(v) for i, v in enumerate(state)})
        ref = self.observer.knowledge.create(
            fields=inputs, source="fresh acquired prediction input"
        )
        return self.observer.knowledge.execute(value.operation, (ref,))

    def metadata(self) -> dict[str, Any]:
        def ref(value: Ref | None) -> list[Any] | None:
            return None if value is None else [value.id, value.revision, value.namespace]

        return {
            "status": self.status,
            "budget": self.budget,
            "spent": self.spent,
            "allowed": list(self.allowed),
            "focus": self.focus,
            "session": ref(self.session),
            "reviser": ref(self.reviser),
            "history": [ref(v) for v in self.history],
            "active": [ref(v) for v in self.active],
            "options": [ref(v) for v in self.options],
            "selected": ref(self.selected),
            "scores": list(self.last_scores),
            "gain": self.gain,
            "prototypes": self.prototypes,
            "priority_trained": self.priority_trained,
            "neural_updates": self.neural_updates,
            "last_learned_tick": self.last_learned_tick,
            "consolidate_every": self.consolidate_every,
            "learning": self.learning,
            "estimates": [
                {
                    "key": v.key,
                    "channel": v.channel,
                    "action": v.action,
                    "appearance": v.appearance,
                    "scope": v.scope,
                    "samples": v.samples,
                    "evidence": [ref(r) for r in v.evidence],
                    "conflicts": [ref(r) for r in v.conflicts],
                    "hypothesis": ref(v.hypothesis),
                    "relationship": ref(v.relationship),
                    "operation": ref(v.operation),
                    "status": v.status,
                    "excluded": v.excluded,
                    "contexts": v.contexts,
                }
                for v in self.estimates.values()
            ],
        }

    @classmethod
    def restore(cls, observer: GroundedObserver, raw: dict[str, Any]) -> Discovery:
        result = object.__new__(cls)
        result.observer = observer

        def ref(value: list[Any] | None) -> Ref | None:
            if value is None:
                return None
            if (
                type(value) is not list
                or len(value) != 3
                or type(value[0]) is not str
                or type(value[1]) is not int
                or type(value[2]) is not str
            ):
                raise ValidationError("Malformed discovery reference")
            resolved = Ref(*value)
            observer.knowledge.get(resolved)
            return resolved

        expected = {
            "status",
            "budget",
            "spent",
            "allowed",
            "focus",
            "session",
            "reviser",
            "history",
            "active",
            "options",
            "selected",
            "scores",
            "gain",
            "prototypes",
            "priority_trained",
            "neural_updates",
            "last_learned_tick",
            "consolidate_every",
            "learning",
            "estimates",
        }
        if set(raw) != expected:
            raise ValidationError("Malformed discovery checkpoint fields")

        def required_ref(value: list[Any] | None) -> Ref:
            resolved = ref(value)
            if resolved is None:
                raise ValidationError("Missing required discovery reference")
            return resolved

        if (
            raw["status"] not in {"running", "paused", "stopped", "exhausted"}
            or any(
                type(raw[k]) is not int or raw[k] < 0 for k in ("budget", "spent", "neural_updates")
            )
            or raw["spent"] > raw["budget"]
            or raw["budget"] > 10000
            or type(raw["last_learned_tick"]) is not int
            or raw["last_learned_tick"] < -1
            or type(raw["consolidate_every"]) is not int
            or raw["consolidate_every"] < 1
            or type(raw["learning"]) is not bool
            or type(raw["priority_trained"]) is not bool
            or raw["focus"] not in {None, "object", "self", "interface"}
            or not raw["allowed"]
            or len(set(raw["allowed"])) != len(raw["allowed"])
            or any(v not in ACTIONS for v in raw["allowed"])
            or raw["status"] in {"running", "paused"}
            and raw["spent"] >= raw["budget"]
            or raw["status"] == "exhausted"
            and raw["spent"] != raw["budget"]
        ):
            raise ValidationError("Invalid discovery controls")
        for key in (
            "status",
            "budget",
            "spent",
            "focus",
            "gain",
            "priority_trained",
            "neural_updates",
            "last_learned_tick",
            "consolidate_every",
            "learning",
        ):
            setattr(result, key, raw[key])
        result.allowed = tuple(raw["allowed"])
        result.prototypes = [tuple(v) for v in raw["prototypes"]]
        if any(len(v) != 3 or any(not 0 <= x <= 1 for x in v) for v in result.prototypes):
            raise ValidationError("Invalid discovery appearance prototypes")
        result.session = cast(Ref, ref(raw["session"]))
        if result.session is None:
            raise ValidationError("Discovery checkpoint lacks its native session")
        result.reviser, result.selected = ref(raw["reviser"]), ref(raw["selected"])
        result.history = [required_ref(v) for v in raw["history"]]
        result.active = tuple(required_ref(v) for v in raw["active"])
        result.options = tuple(required_ref(v) for v in raw["options"])
        result.last_scores = tuple(raw["scores"])
        result.estimates = {}
        for row in raw["estimates"]:
            if (
                row["channel"] not in {"self", "object", "interface"}
                or row["action"] not in ACTIONS
                or row["status"] not in {"unknown", "hypothesized", "supported", "qualified"}
                or type(row["excluded"]) is not int
                or row["excluded"] < 0
                or len(row["samples"]) != len(row["evidence"])
                or row["channel"] == "object"
                and len(row["contexts"]) != len(row["samples"])
                or any(
                    len(v) != 3 or any(type(x) not in (int, float) for x in v)
                    for v in row["contexts"]
                )
                or any(
                    len(v) != {"object": 3, "self": 4, "interface": 2}[row["channel"]]
                    or any(type(x) not in (int, float) for x in v)
                    for v in row["samples"]
                )
            ):
                raise ValidationError("Invalid discovery hypothesis measurements")
            value = Estimate(
                row["key"],
                row["channel"],
                row["action"],
                tuple(row["appearance"]),
                row["scope"],
                [tuple(v) for v in row["samples"]],
                [required_ref(v) for v in row["evidence"]],
                [required_ref(v) for v in row["conflicts"]],
                ref(row["hypothesis"]),
                ref(row["relationship"]),
                ref(row["operation"]),
                row["status"],
                row["excluded"],
                [tuple(v) for v in row["contexts"]],
            )
            if (
                value.key in result.estimates
                or value.hypothesis is None
                or observer.knowledge.get(value.hypothesis).fields.get("kind")
                != "discovery_hypothesis"
            ):
                raise ValidationError("Invalid native hypothesis binding")
            fields = observer.knowledge.get(value.hypothesis).fields
            if (
                fields["status"] != value.status
                or fields["channel"] != value.channel
                or fields["action"] != value.action
                or fields["scope"] != value.scope
                or fields["observations"] != len(value.samples)
                or fields["excluded"] != value.excluded
                or fields["prediction"] != tuple(exact(v) for v in value.mean)
                or any(ref not in value.evidence for ref in value.conflicts)
                or (value.operation is None) != (value.relationship is None)
                or value.status == "supported"
                and len(value.samples) < 3
            ):
                raise ValidationError("Discovery metadata disagrees with native evidence")
            if (
                value.operation is not None
                and observer.knowledge.get(value.operation).role != Role.OPERATION
            ):
                raise ValidationError("Acquired prediction is not an executable operation")
            if value.relationship is not None and (
                observer.knowledge.get(value.relationship).fields.get("operation")
                != value.operation
            ):
                raise ValidationError("Acquired relationship binds a different procedure")
            result.estimates[value.key] = value
        # The inner checkpoint owns the exact active operation scope. Keep it,
        # including after a stopped inquiry, rather than reconstructing a new graph.
        return result
