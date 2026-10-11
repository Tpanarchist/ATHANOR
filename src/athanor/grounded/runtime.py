"""Restricted inference and embodied execution using the existing structural observer."""

import hashlib
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import TYPE_CHECKING, cast

import torch

from athanor import (
    Athanor,
    AthanorError,
    Endpoint,
    Evidence,
    FieldSpec,
    Procedure,
    Ref,
    Role,
    ValidationError,
    record,
)
from athanor.neural import Decision, Observation, Observer
from athanor.neural.contracts import Activation
from athanor.neural.encoding import Frame, structural_features
from athanor.neural.model import NeuralObserverModel
from athanor.programs import argument, binary, choose, field, literal
from athanor.semantics import Example, Lexicon, Teacher
from athanor.values import as_ref, as_tuple
from athanor.world import Action, SensorFrame, World
from athanor.world.engine import ACTIONS, exact

from .discovery import Discovery
from .perception import Percept, SensoryEncoder, descriptor_distance, perceive, pixels

if TYPE_CHECKING:
    from .training import GroundedTrainer


def install_map(namespace: str = "observer-map") -> tuple[Athanor, dict[str, Ref], Ref]:
    a = Athanor(namespace=namespace)
    command = a.teach_category("embodied command", {"action": FieldSpec("string")})
    teacher = Teacher(a)
    operations = {}
    for name in ACTIONS:
        concept = teacher.define(
            name,
            "a bounded motor/interface command; installation does not prove physical success",
            Procedure((), command, {"action": literal(name)}),
            expressions=(name.replace("_", " "),),
            examples=(Example((), record(action=name)),),
        )
        operations[name] = as_ref(a.get(concept).fields["operation"])
    quantity = a.teach_category("Quantity", {"value": FieldSpec("number")})
    a.create(fields=record(kind="language_anchor", quantity=quantity))
    target = a.create(fields=record(kind="visual_goal", definition="observed appearance prototype"))
    return a, operations, target


def request_fields(
    goal: str, target: Percept | None, *, contact: bool = False, width: int = 24
) -> dict[str, object]:
    relative = None if target is None else target.relative
    return {
        "kind": "embodied_observation",
        "goal": goal,
        "visible": target is not None,
        "depth_known": relative is not None,
        "horizontal": exact(0 if target is None else (target.center[0] - width / 2) / (width / 2)),
        "distance": exact(0 if relative is None else relative[2]),
        "contact": contact,
    }


def intent_activation(
    model: NeuralObserverModel, frame: Frame, knowledge: Athanor | None = None
) -> torch.Tensor:
    # This is the same shared structural encoder. Intent is a first-class compact
    # interface view; image/receipt/track metadata must not numerically dilute it.
    if knowledge is None:
        features = (
            frame.features[frame.query_index]
            if frame.intent_features is None
            else frame.intent_features
        )
    else:
        node = knowledge.get(frame.refs[frame.query_index])
        keys = ("kind", "goal", "visible", "depth_known", "horizontal", "distance", "contact")
        compact = (
            node
            if node.fields.get("kind") == "structural_request"
            else replace(node, fields=record({key: node.fields[key] for key in keys}))
        )
        features = torch.tensor(structural_features(compact, observation=True))
    return cast(torch.Tensor, model.encoder(features)).tanh()


def contextual_frame(frame: Frame, knowledge: Athanor) -> Frame:
    node = knowledge.get(frame.refs[frame.query_index])
    if node.fields.get("kind") == "structural_request":
        return frame
    keys = ("kind", "goal", "visible", "depth_known", "horizontal", "distance", "contact")
    compact = replace(node, fields=record({key: node.fields[key] for key in keys}))
    return replace(
        frame, intent_features=torch.tensor(structural_features(compact, observation=True))
    )


class PixelObserver(Observer):
    """Same parameters/messages/GRU/decoders, with explicit sensory query activation."""

    def __init__(
        self,
        knowledge: Athanor,
        model: NeuralObserverModel,
        sensory: SensoryEncoder,
        *,
        self_ref: Ref | None = None,
    ) -> None:
        super().__init__(knowledge, model, self_ref=self_ref)
        self.sensory = sensory
        self.sensor: SensorFrame | None = None
        self.input_state = self.state.detached()
        self.active_operations: tuple[Ref, ...] | None = None

    def think(self, *, steps: int | None = None) -> Decision:
        if self.sensor is None:
            return super().think(steps=steps)
        if self.observation is None:
            raise ValidationError("A sensory step requires a structural observation")
        frame = self.builder.build(
            self.knowledge,
            self.observation,
            self_refs=tuple(self.self_map),
            observer=self.self_ref,
            operations=self.active_operations,
        )
        frame = contextual_frame(frame, self.knowledge)
        self.model.eval()
        self.sensory.eval()
        self.input_state = self.state.detached()
        with torch.no_grad():
            activation = self.sensory.activation(
                pixels(self.sensor),
                torch.tensor(self.sensor.proprio),
                intent_activation(self.model, frame, self.knowledge),
            )
            output = self.model([frame], [self.state], sensory=[activation], steps=steps)[0]
        self._frame, self._output = frame, output
        self.state = output.activation.detached()
        self.activation_history.append(tuple(float(v) for v in self.state.context[:8]))
        self.activation_history = self.activation_history[-self.model.config.history_limit :]
        return self.propose()


@dataclass(slots=True)
class Track:
    ref: Ref
    appearance: tuple[float, float, float]
    relative: tuple[float, float, float] | None
    confidence: float
    last_seen: int
    status: str


class GroundedObserver:
    """No World field or privileged labels: all inference consumes SensorFrame only."""

    def __init__(
        self,
        model: NeuralObserverModel | None = None,
        sensory: SensoryEncoder | None = None,
        *,
        namespace: str = "observer-map",
    ) -> None:
        model = NeuralObserverModel() if model is None else model
        sensory = SensoryEncoder(model.config.width) if sensory is None else sensory
        a, self.operations, self.target_anchor = install_map(namespace)
        self.observer = PixelObserver(a, model, sensory)
        self.tracks: list[Track] = []
        from .processes import Processes

        self.processes = Processes(self)
        from .induction import Induction

        self.induction = Induction(self)
        self.goal = "observe"
        self.target: Ref | None = None
        self.prediction: tuple[float, ...] | None = None
        self.last_percepts: tuple[Percept, ...] = ()
        self.last_request: Ref | None = None
        self.last_feedback: Ref | None = None
        self.history: list[Ref] = []
        self.experience: dict[str, tuple[int, float, float]] = {}
        self.motion_evidence: list[Ref] = []
        self.consolidations: dict[str, Ref] = {}
        self.update_track = a.teach_operation(
            "revise measured map",
            Procedure(
                (Endpoint(Role.DATA), Endpoint(Role.DATA)),
                None,
                {
                    name: field(argument(1), name)
                    for name in (
                        "kind",
                        "appearance",
                        "relative",
                        "confidence",
                        "last_seen",
                        "status",
                        "x",
                        "y",
                        "z",
                    )
                },
                update=0,
            ),
        )
        # A taught relation is an exact computation over estimated map values, not truth.
        boolean = a.teach_category("Estimated relation result", {"value": FieldSpec("bool")})
        near = binary(
            "lt",
            binary(
                "add",
                binary("mul", field(argument(0), "x"), field(argument(0), "x")),
                binary("mul", field(argument(0), "z"), field(argument(0), "z")),
            ),
            literal(Fraction(9, 4)),
        )
        self.near = Teacher(a).define(
            "within reach",
            "estimated distance from stereo map below 1.5; unknown depth cannot justify it",
            Procedure(
                (Endpoint(Role.DATA),),
                boolean,
                {
                    "value": choose(
                        binary("eq", field(argument(0), "status"), literal("visible")),
                        near,
                        literal(None),
                    )
                },
            ),
            expressions=("near", "within reach"),
            roles=("perceived participant",),
        )
        self.discovery = Discovery(self)

    @property
    def knowledge(self) -> Athanor:
        return self.observer.knowledge

    def teach_visible(self, expression: str, index: int = 0) -> Ref:
        if not 0 <= index < len(self.last_percepts):
            raise ValidationError("Teaching requires an actual visible perceptual example")
        percept = self.last_percepts[index]
        if percept.status != "visible":
            raise ValidationError("Unresolved correspondence cannot establish a prototype")

        def install(a: Athanor) -> Ref:
            concept = a.create(
                fields=record(
                    kind="visual_concept",
                    definition=(
                        "appearance grounded by a camera example; "
                        "kind membership remains a hypothesis"
                    ),
                    appearance=tuple(exact(v) for v in percept.appearance),
                    exemplar=self.last_request,
                    installed=True,
                    recognition_demonstrated=False,
                ),
                source="teacher:pointer to measured camera region",
            )
            Teacher(a).alias(expression, concept, evidence=Evidence("teacher:camera exemplar"))
            return concept

        return self.knowledge.transaction(install)

    def instruct(self, text: str) -> str:
        words = text.casefold().strip().split(maxsplit=1)
        if not words or words[0] not in ("approach", "observe", "search", *ACTIONS):
            raise ValidationError(
                "Instruction grammar: approach <taught name>, observe, search or bounded motor verb"
            )
        self.goal, self.target = words[0], None
        if self.goal == "approach":
            if len(words) != 2:
                raise ValidationError("Approach requires a taught expression")
            bindings = [
                b
                for b in Lexicon(self.knowledge).bindings(words[1])
                if b.assessment.value == "supported"
                and self.knowledge.get(b.referent).fields.get("kind") == "visual_concept"
            ]
            if len(bindings) != 1:
                self.goal = "unknown"
                return "unknown or ambiguous concept"
            self.target = bindings[0].referent
        return "accepted bounded goal; action remains a learned proposal"

    def _target_percept(self, percepts: tuple[Percept, ...]) -> Percept | None:
        if self.target is None:
            return percepts[0] if len(percepts) == 1 else None
        descriptor = tuple(
            float(cast(Fraction, x))
            for x in as_tuple(self.knowledge.get(self.target).fields["appearance"])
        )
        matches = [p for p in percepts if descriptor_distance(p.appearance, descriptor) < 0.18]
        return matches[0] if len(matches) == 1 else None

    def _map(self, frame: SensorFrame, percepts: tuple[Percept, ...]) -> tuple[Ref, ...]:
        previous = self.observer.sensor
        delta = frame.proprio[:4] if previous is None or frame.tick != previous.tick else (0.0,) * 4
        matched: set[int] = set()
        active = []
        for percept in percepts:
            matches = [
                (i, t)
                for i, t in enumerate(self.tracks)
                if descriptor_distance(t.appearance, percept.appearance) < 0.18
            ]
            status = percept.status
            if len(matches) > 1 or len(matches) == 1 and matches[0][0] in matched:
                # Identity is not inferable from indistinguishable appearance alone.
                status = "ambiguous_identity"
                continue
            if matches:
                index, track = matches[0]
                matched.add(index)
            else:
                track = Track(
                    self.knowledge.create(
                        fields={"kind": "identity_hypothesis"},
                        source="camera-derived track initialization",
                    ),
                    percept.appearance,
                    None,
                    0,
                    frame.tick,
                    status,
                )
                self.tracks.append(track)
                matched.add(len(self.tracks) - 1)
            if track.relative is not None and percept.relative is not None:
                from athanor.world.geometry import rotate

                expected = rotate(
                    cast(
                        tuple[float, float, float],
                        tuple(old - d for old, d in zip(track.relative, delta[:3], strict=True)),
                    ),
                    -delta[3],
                )
                residual = tuple(
                    new - old for new, old in zip(percept.relative, expected, strict=True)
                )
                self.motion_evidence.append(
                    self.knowledge.create(
                        fields=record(
                            kind="perceptual_motion_evidence",
                            participant=track.ref,
                            residual=tuple(exact(v) for v in residual),
                            body_moved=bool(frame.proprio[5]),
                            interpretation="possible independent motion or measurement error",
                            uncertain=True,
                        ),
                        source="learned stereo measurements compared with odometry",
                    )
                )
                self.motion_evidence = self.motion_evidence[-4:]
            self._publish(
                track, percept.appearance, percept.relative, percept.confidence, frame.tick, status
            )
            active.append(track.ref)
        for i, track in enumerate(self.tracks):
            if i not in matched:
                relative = track.relative
                if relative is not None:
                    from athanor.world.geometry import rotate

                    relative = rotate(
                        (
                            relative[0] - delta[0],
                            relative[1] - delta[1],
                            relative[2] - delta[2],
                        ),
                        -delta[3],
                    )
                self._publish(
                    track,
                    track.appearance,
                    relative,
                    track.confidence * 0.9,
                    track.last_seen,
                    "occluded_hypothesis",
                )
        return tuple(active)

    def _publish(
        self,
        track: Track,
        appearance: tuple[float, float, float],
        relative: tuple[float, float, float] | None,
        confidence: float,
        tick: int,
        status: str,
    ) -> None:
        position = relative or (0.0, 0.0, 0.0)
        values = record(
            kind="perceived_track",
            appearance=tuple(exact(v) for v in appearance),
            relative=None if relative is None else tuple(exact(v) for v in relative),
            confidence=exact(confidence),
            last_seen=tick,
            status=status,
            x=exact(position[0]),
            y=exact(position[1]),
            z=exact(position[2]),
        )
        temporary = self.knowledge.create(
            fields=values, source="learned pixels / calibrated stereo / conservative association"
        )
        track.ref = self.knowledge.execute(self.update_track, (track.ref, temporary))
        track.appearance, track.relative, track.confidence, track.last_seen, track.status = (
            appearance,
            relative,
            confidence,
            tick,
            status,
        )

    def observe(self, frame: SensorFrame) -> Decision:
        if self.induction.query is not None:
            self.observer.active_operations = tuple(self.operations.values())
        self.last_percepts = perceive(self.observer.sensory, frame)
        active = self._map(frame, self.last_percepts)
        inquiries = self.discovery.prepare(frame)
        target = self._target_percept(self.last_percepts)
        values = request_fields(
            self.goal, target, contact=bool(frame.proprio[4]), width=frame.width
        )
        values.update(
            tracks=active,
            target_concept=self.target,
            relationships=tuple(self.motion_evidence),
            hypotheses=tuple(t.ref for t in self.tracks if t.status != "visible")
            + inquiries
            + self.processes.active
            + self.induction.active,
            inquiry_options=self.discovery.options,
        )
        self.last_request = self.knowledge.create(
            fields=values, source="restricted stereo sensory interface"
        )
        self.knowledge.create(
            fields=record(
                kind="sensor_capture_receipt",
                observation=self.last_request,
                tick=frame.tick,
                cameras=tuple(
                    hashlib.sha256(image).hexdigest() for image in (frame.left, frame.right)
                ),
            ),
            source="camera interface provenance; outside active neural query",
        )
        self.observer.sensor = frame
        decision = self.observer.observe(Observation(self.last_request, ()))
        return decision

    def propose(self, *, commit: bool = True) -> tuple[Action, Decision]:
        decision = (
            self.discovery.choose(commit=commit)
            if self.discovery.status != "stopped"
            else self.observer.propose()
        )
        name = (
            "wait"
            if self.goal == "unknown"
            else next(
                (
                    name
                    for name, operation in self.operations.items()
                    if operation == decision.proposal.operation
                ),
                "wait",
            )
        )
        output = self.observer._output
        sensor = self.observer.sensor
        if output is None or sensor is None:
            raise ValidationError("Prediction requires an observed state")
        with torch.no_grad():
            prediction = tuple(
                float(v)
                for v in self.observer.sensory.predict(
                    output.activation.context, ACTIONS.index(name), torch.tensor(sensor.proprio)
                )
            )
        if commit:
            self.prediction = prediction
        return Action(name), decision

    def command(self, decision: Decision) -> Ref | None:
        # Kernel validation precedes any effect. Unknown operation cannot become a motor action.
        if decision.proposal.operation not in self.operations.values() or self.goal == "unknown":
            return None
        # Structural admissibility is distinct from physical success. World feedback
        # supplies the self-record after the actual effect, without an extra fictitious
        # body outcome or unmatched recurrent step between perception and action.
        assert decision.proposal.operation is not None
        try:
            return self.knowledge.execute(decision.proposal.operation, decision.proposal.arguments)
        except AthanorError as error:
            rejected = self.knowledge.create(
                fields=record(
                    kind="embodied_proposal_rejection",
                    observer=self.observer.self_ref,
                    proposed=decision.proposal.operation,
                    arguments=decision.proposal.arguments,
                    error=type(error).__name__,
                    reason=str(error),
                    physical_effect=False,
                ),
                source="kernel validated rejection before world execution",
            )
            self.history.append(rejected)
            self.observer.self_map.append(rejected)
            self.observer.self_map = self.observer.self_map[
                -self.observer.model.config.history_limit :
            ]
            return None

    def feedback(
        self, before: SensorFrame, after: SensorFrame, intended: str, predicted: tuple[float, ...]
    ) -> Ref:
        actual = physical_outcome(before, after)
        error = tuple(a - p for a, p in zip(actual, predicted, strict=True))
        movement = intended in {"forward", "backward", "left", "right"}
        ambiguous = movement and abs(actual[0]) + abs(actual[1]) < 0.05
        alternatives = (
            (
                "world obstruction",
                "observer actuator limitation",
                "interface observation limitation",
            )
            if ambiguous
            else ("world or self motion observed",)
        )
        evidence = (
            "contact supports obstruction; other causes may coexist"
            if after.proprio[4]
            else "odometry supports motion but images may be stale"
            if after.proprio[5] and actual[3] < 0.005
            else "insufficient evidence to uniquely attribute cause"
            if ambiguous
            else "restricted sensor consequence"
        )
        success = (
            bool(after.proprio[5])
            if movement
            else bool(after.proprio[4])
            if intended == "interact"
            else after.executed == intended
        )
        preference = None
        if self.goal == "approach":
            old_target = self._target_percept(perceive(self.observer.sensory, before))
            new_target = self._target_percept(perceive(self.observer.sensory, after))
            if old_target and new_target and old_target.relative and new_target.relative:
                preference = old_target.relative[2] - new_target.relative[2]
        ref = self.knowledge.create(
            fields=record(
                kind="embodied_self_observation",
                observer=self.observer.self_ref,
                intended=intended,
                executed=after.executed,
                expected=tuple(exact(v) for v in predicted),
                observed=tuple(exact(v) for v in actual),
                predictive_discrepancy=tuple(exact(v) for v in error),
                capability_success=success,
                preference_discrepancy=None if preference is None else exact(preference),
                alternatives=alternatives,
                attribution=evidence,
                uncertain=ambiguous,
                tick=after.tick,
            ),
            source="measured interface consequences; no world authority",
        )
        self.last_feedback = ref
        if intended in self.operations:
            operation = self.operations[intended]
            attempts, successes = self.observer.capabilities.get(operation, (0, 0))
            self.observer.capabilities[operation] = attempts + 1, successes + int(success)
        self.history.append(ref)
        self.observer.self_map.append(ref)
        self.observer.self_map = self.observer.self_map[-self.observer.model.config.history_limit :]
        count, changes, physical_successes = self.experience.get(intended, (0, 0.0, 0.0))
        self.experience[intended] = (
            count + 1,
            changes + actual[3],
            physical_successes + float(success),
        )
        n, c, s = self.experience[intended]
        self.consolidations[intended] = self.knowledge.create(
            fields=record(
                kind="acquired_consequence_summary",
                action=intended,
                observations=n,
                mean_visual_change=exact(c / n),
                success_fraction=exact(s / n),
                status=(
                    "supported empirical association within observed interface; "
                    "not universal causality"
                ),
            ),
            source="consolidated measured experience",
        )
        return ref


def physical_outcome(before: SensorFrame, after: SensorFrame) -> tuple[float, ...]:
    change = float((pixels(after) - pixels(before)).abs().mean())
    return (
        after.proprio[0] / 0.25,
        after.proprio[2] / 0.25,
        after.proprio[4],
        change,
        float(after.proprio[5]),
        min(1.0, abs(after.proprio[3]) / 0.3),
    )


class Runtime:
    """Coordinator alone owns territory. Model calls receive only restricted frames."""

    def __init__(
        self,
        world: World | None = None,
        observer: GroundedObserver | None = None,
        *,
        trainer: GroundedTrainer | None = None,
    ) -> None:
        self.world = World() if world is None else world
        self.observer = GroundedObserver() if observer is None else observer
        self.paused = True
        self.running = True
        self.last_transition: (
            tuple[SensorFrame, SensorFrame, str, Frame | None, Activation | None] | None
        ) = None
        self.observer.observe(self.world.sense())
        if trainer is not None and trainer.observer is not self.observer:
            raise ValidationError("Continual optimizer belongs to a different grounded observer")
        self.discovery_trainer: GroundedTrainer | None = trainer

    def start_discovery(
        self, budget: int = 24, *, focus: str | None = None, allowed: tuple[str, ...] = ACTIONS
    ) -> None:
        self.observer.discovery.start(budget, focus=focus, allowed=allowed)
        self.running, self.paused = True, False
        self.observer.observe(self.world.sense())

    def autonomous_step(self) -> dict[str, object]:
        discovery = self.observer.discovery
        if self.paused or not self.running or discovery.status != "running":
            return {"acted": False, "reason": discovery.status, "paused": self.paused}
        return self.step()

    def start_process_discovery(
        self, budget: int = 12, *, max_calls: int = 2, interface: str = "body"
    ) -> None:
        self.observer.processes.start(budget, max_calls=max_calls, interface=interface)
        if interface == "numeric":
            self.observer.discovery.pause()
            self.paused = True
            self.observer.observe(self.world.sense())
            return
        self.start_discovery(
            budget,
            focus="self" if interface == "body" else "object",
            allowed=("wait", "forward" if interface == "body" else "interact"),
        )

    def step(self, manual: Action | None = None) -> dict[str, object]:
        discovery = self.observer.discovery
        if manual is None and discovery.status not in {"stopped", "running"}:
            return {"acted": False, "reason": discovery.status}
        before = self.world.sense()
        training_frame = self.observer.observer._frame
        training_state = self.observer.observer.input_state.detached()
        action, decision = self.observer.propose()
        if manual is None and discovery.status == "running" and decision.proposal.operation is None:
            discovery.pause()
            self.paused = True
            return {"acted": False, "reason": "neural abstention"}
        if manual is None:
            command = self.observer.command(decision)
            if command is not None:
                action = Action(str(self.observer.knowledge.get(command).fields["action"]))
            else:
                action = Action()
        else:
            action = manual
            # An explicit external action is not presented as a learned policy decision.
            with torch.no_grad():
                output = self.observer.observer._output
                assert output is not None
                self.observer.prediction = tuple(
                    float(v)
                    for v in self.observer.observer.sensory.predict(
                        output.activation.context,
                        ACTIONS.index(action.name),
                        torch.tensor(before.proprio),
                    )
                )
        prediction = self.observer.prediction or (0.0,) * 6
        event = self.world.step(action)
        after = self.world.sense()
        self.last_transition = (before, after, action.name, training_frame, training_state)
        feedback = self.observer.feedback(before, after, action.name, prediction)
        discovery.ingest(before, after, action.name, feedback, autonomous=manual is None)
        processes = self.observer.processes
        acquired_process = None
        previous_samples = len(processes.inputs)
        processes.ingest(before, after, feedback)
        if len(processes.inputs) > previous_samples and discovery.learning:
            from .training import GroundedTrainer

            if self.discovery_trainer is None:
                self.discovery_trainer = GroundedTrainer(self.observer, learning_rate=0.0001)
            processes.learn(self.discovery_trainer, steps=10)
            acquired_process = processes.adopt()
        if (
            manual is None
            and discovery.learning
            and discovery.status != "stopped"
            and discovery.spent % discovery.consolidate_every == 0
            and discovery.last_learned_tick != after.tick
        ):
            from .training import GroundedTrainer

            if self.discovery_trainer is None:
                self.discovery_trainer = GroundedTrainer(self.observer, learning_rate=0.0001)
            self.discovery_trainer.consolidate_transition(
                before,
                after,
                action.name,
                frame=training_frame,
                prior=training_state,
                information_gain=discovery.gain,
            )
            discovery.neural_updates += 1
            discovery.last_learned_tick = after.tick
        self.observer.observe(after)
        return {
            "acted": True,
            "intended": action.name,
            "manual": manual is not None,
            "executed": after.executed,
            "world_event": str(event),
            "feedback": str(feedback),
            "prediction": prediction,
            "observed": physical_outcome(before, after),
            "inquiry_status": discovery.status,
            "experiment": None if discovery.selected is None else str(discovery.selected),
            "evidence_gain": discovery.gain,
            "acquired_process": None if acquired_process is None else str(acquired_process),
            "decision_source": "shared neural experiment scorer"
            if discovery.status != "stopped"
            else "shared neural motor pointer",
        }
