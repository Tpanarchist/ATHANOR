"""Authority is confined to curriculum preparation; actual RGB drives model learning."""

import math
import random
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor
from torch.nn import functional as F

from athanor import Ref, ValidationError
from athanor.neural.contracts import Activation
from athanor.neural.curriculum import Lesson
from athanor.neural.encoding import Frame
from athanor.neural.model import NeuralObserverModel
from athanor.neural.training import Trainer
from athanor.world import Action, SensorFrame, World
from athanor.world.engine import ACTIONS

from .perception import SensoryEncoder, pixels
from .runtime import GroundedObserver, intent_activation, physical_outcome

COLORS = ((225, 45, 35), (40, 195, 75), (40, 90, 230), (230, 190, 35), (175, 45, 200))


@dataclass(frozen=True, slots=True)
class VisualLesson:
    frame: SensorFrame
    foreground: Tensor
    appearance: Tensor


def visual_lessons(count: int = 240, *, seed: int = 404) -> list[VisualLesson]:
    rng = random.Random(seed)
    output = []
    for i in range(count):
        world = World(namespace=f"visual-teacher-{seed}-{i}")
        for _ in range(rng.randrange(0, 4)):
            color = tuple(rng.randrange(35, 240) for _ in range(3))
            world.add_object(
                (rng.uniform(-2.5, 2.5), rng.uniform(-0.7, 0.7), rng.uniform(1, 5)),
                shape=rng.choice(("sphere", "box")),
                size=(rng.uniform(0.25, 0.55),) * 3,
                color=cast(tuple[int, int, int], color),
            )
        world.yaw = rng.uniform(-0.5, 0.5)
        world.pitch = rng.uniform(-0.1, 0.1)
        left, right, labels = world.render(labels=True)
        frame = world.sense()
        assert frame.left == left and frame.right == right
        mask = (torch.tensor(labels).reshape(2, 24, 24) > 0).float()
        image = pixels(frame)
        maximum = image.amax(1, keepdim=True).clamp_min(0.01)
        # Shared continuous appearance encoding; not one classifier per concept.
        appearance = image / maximum
        output.append(VisualLesson(frame, mask, appearance))
    return output


def train_vision(
    encoder: SensoryEncoder, material: list[VisualLesson], *, epochs: int = 80
) -> dict[str, float]:
    if not material or epochs < 1:
        raise ValidationError("Visual training requires material and positive epochs")
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=0.003)
    for module in (encoder.image, encoder.foreground, encoder.appearance):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    images = torch.cat([pixels(lesson.frame) for lesson in material])
    masks = torch.cat([lesson.foreground for lesson in material])
    appearance = torch.cat([lesson.appearance for lesson in material])
    rng = random.Random(405)
    loss = torch.tensor(0.0)
    encoder.train()
    for _ in range(epochs):
        order = list(range(len(images)))
        rng.shuffle(order)
        for start in range(0, len(order), 48):
            indices = order[start : start + 48]
            logits, predictions, _ = encoder.encode(images[indices])
            foreground = masks[indices]
            segmentation = F.binary_cross_entropy_with_logits(
                logits, foreground, pos_weight=torch.tensor(4.0)
            )
            visual = ((predictions - appearance[indices]).square() * foreground[:, None]).sum() / (
                3 * foreground.sum()
            ).clamp_min(1)
            loss = segmentation + visual
            optimizer.zero_grad()
            torch.autograd.backward(loss)
            optimizer.step()
    encoder.eval()
    return {"loss": float(loss.detach()), **evaluate_vision(encoder, material)}


def evaluate_vision(encoder: SensoryEncoder, material: list[VisualLesson]) -> dict[str, float]:
    with torch.no_grad():
        logits, predictions, _ = encoder.encode(
            torch.cat([pixels(lesson.frame) for lesson in material])
        )
        labels = torch.cat([lesson.foreground for lesson in material]).bool()
        predicted = logits.sigmoid() > 0.6
        intersection = (predicted & labels).sum().item()
        union = (predicted | labels).sum().item()
        target = torch.cat([lesson.appearance for lesson in material])
        error = ((predictions - target).abs() * labels[:, None]).sum() / (
            3 * labels.sum()
        ).clamp_min(1)
    return {"foreground_iou": intersection / max(union, 1), "appearance_mae": float(error)}


@dataclass(frozen=True, slots=True)
class EmbodiedLesson:
    frame: Frame
    sensor: SensorFrame
    operation: Ref
    action: int
    outcome: tuple[float, ...]
    prior: Activation | None = None
    previous_frame: Frame | None = None
    previous_sensor: SensorFrame | None = None
    history: tuple[tuple[Frame, SensorFrame], ...] = ()


def embodied_lessons(
    observer: GroundedObserver, count: int = 400, *, seed: int = 406
) -> list[EmbodiedLesson]:
    """Native teaching episodes vary library size, names, clutter and prior history."""
    rng = random.Random(seed)
    material: list[EmbodiedLesson] = []
    model, sensory = observer.observer.model, observer.observer.sensory
    previous_frame: Frame | None = None
    previous_sensor: SensorFrame | None = None
    prototypes: list[Ref] = []
    palette = list(COLORS)
    for i in range(count):
        if i % 20 == 0:
            observer = GroundedObserver(
                model=model, sensory=sensory, namespace=f"curriculum-{seed}-{i}"
            )
            palette = rng.sample(list(COLORS), rng.randrange(1, 6))
            prototypes = []
            for j, color in enumerate(palette):
                example = World(namespace=f"prototype-{seed}-{i}-{j}")
                example.add_object((0, 0, 3), color=color)
                observer.observe(example.sense())
                # Names have no authoritative category or identity shortcut.
                name = rng.choice(("marker", "beacon", "token", "lamp", "kind")) + " " + chr(97 + j)
                prototypes.append(observer.teach_visible(name))
            from athanor.neural.contracts import empty_activation

            observer.observer.state = empty_activation(model.config.width)
            observer.observer.self_map.clear()
            observer.tracks.clear()
            observer.motion_evidence.clear()
            previous_frame, previous_sensor = None, None
        world = World(namespace=f"action-teacher-{seed}-{i}")
        x, z = rng.uniform(-1.8, 1.8), rng.uniform(0.75, 4.5)
        color = rng.choice(palette)
        velocity = (rng.choice((-0.4, 0.4)), 0.0, 0.0) if i % 4 == 0 else (0.0, 0.0, 0.0)
        ref = world.add_object(
            (x, 0, z), size=(0.3, 0.3, 0.3), color=color, movable=True, velocity=velocity
        )
        for _ in range(rng.randrange(3)):
            other_color = rng.choice([c for c in COLORS if c != color])
            other_x = rng.uniform(-2.2, 2.2)
            other_z = rng.uniform(1.8, 4.8)
            if abs(other_x - x) > 0.7 or abs(other_z - z) > 1:
                world.add_object(
                    (other_x, rng.uniform(-0.3, 0.3), other_z),
                    color=other_color,
                    shape=rng.choice(("sphere", "box")),
                    size=(0.25, 0.25, 0.25),
                )
        if i % 7 == 0:
            world.add_object((0, 0, 0.8), shape="box", size=(0.5, 0.7, 0.1), color=(60, 65, 75))
        if i % 11 == 0:
            world.motor = False
        sensor = world.sense()
        goal = rng.choice(("approach", "approach", "approach", "observe", "search", *ACTIONS[:8]))
        observer.goal = goal
        observer.target = prototypes[palette.index(color)] if goal == "approach" else None
        prior = observer.observer.state.detached()
        observer.observe(sensor)
        perceived = observer._target_percept(observer.last_percepts)
        if goal == "approach":
            _, _, labels = world.render(labels=True)
            visible = any(1 in eye for eye in labels)
            bearing = (
                0.0
                if perceived is None or perceived.relative is None
                else perceived.relative[0] / max(perceived.relative[2], 0.1)
            )
            if not visible or perceived is None or perceived.relative is None:
                target = "left"
            elif bearing < -0.18:
                target = "turn_left"
            elif bearing > 0.18:
                target = "turn_right"
            elif z > 1.1:
                target = "forward"
            else:
                target = "interact"
        else:
            target = "wait" if goal == "observe" else "left" if goal == "search" else goal
        frame = observer.observer._frame
        assert frame is not None and observer.last_request is not None
        action = rng.randrange(len(ACTIONS))
        world.step(Action(ACTIONS[action]))
        after = world.sense()
        material.append(
            EmbodiedLesson(
                frame,
                sensor,
                observer.operations[target],
                action,
                physical_outcome(sensor, after),
                prior,
                previous_frame,
                previous_sensor,
            )
        )
        observer.feedback(sensor, after, ACTIONS[action], (0.0,) * 6)
        previous_frame, previous_sensor = frame, sensor
        assert ref.namespace != observer.last_request.namespace
    return material


def continuous_lessons(
    observer: GroundedObserver, count: int = 160, *, seed: int = 408
) -> list[EmbodiedLesson]:
    """General movement episodes, with changing intent and actual recurrent history."""
    rng = random.Random(seed)
    output: list[EmbodiedLesson] = []
    model, sensory = observer.observer.model, observer.observer.sensory
    for episode in range((count + 7) // 8):
        observer = GroundedObserver(
            model=model, sensory=sensory, namespace=f"continuous-map-{seed}-{episode}"
        )
        world = World(namespace=f"continuous-territory-{seed}-{episode}")
        color = rng.choice(COLORS)
        world.add_object(
            (rng.uniform(-1.5, 1.5), 0, rng.uniform(1.5, 4.5)), color=color, movable=True
        )
        world.add_object(
            (rng.choice((-2.0, 2.0)), 0, 3.8),
            color=rng.choice([c for c in COLORS if c != color]),
            velocity=(0.3, 0, 0),
        )
        observer.observe(world.sense())
        candidates = [
            (sum((p.appearance[j] - color[j] / max(color)) ** 2 for j in range(3)), i)
            for i, p in enumerate(observer.last_percepts)
            if p.status == "visible"
        ]
        if not candidates:
            continue
        observer.teach_visible("new participant", min(candidates)[1])
        observer.instruct("approach new participant")
        history: list[tuple[Frame, SensorFrame]] = []
        for step in range(8):
            if len(output) >= count:
                break
            if step == 4 and episode % 3 == 0:
                observer.instruct(rng.choice(("observe", "search", *ACTIONS[:8])))
            sensor = world.sense()
            observer.observe(sensor)
            frame = observer.observer._frame
            assert frame is not None
            target = observer._target_percept(observer.last_percepts)
            if observer.goal == "approach":
                if target is None or target.relative is None:
                    desired = "left"
                else:
                    bearing = target.relative[0] / max(target.relative[2], 0.1)
                    desired = (
                        "turn_left"
                        if bearing < -0.18
                        else "turn_right"
                        if bearing > 0.18
                        else "forward"
                        if target.relative[2] > 1.1
                        else "interact"
                    )
            else:
                desired = (
                    "wait"
                    if observer.goal == "observe"
                    else "left"
                    if observer.goal == "search"
                    else observer.goal
                )
            # Mix teacher movement and exploration; neither is an inference fallback.
            executed = desired if step % 3 else rng.choice(ACTIONS)
            if step == 6 and episode % 5 == 0:
                world.motor = False
            world.step(Action(executed))
            after = world.sense()
            output.append(
                EmbodiedLesson(
                    frame,
                    sensor,
                    observer.operations[desired],
                    ACTIONS.index(executed),
                    physical_outcome(sensor, after),
                    history=tuple(history[-4:]),
                )
            )
            observer.feedback(sensor, after, executed, (0.0,) * 6)
            history.append((frame, sensor))
    return output


def lesson_prior(
    model: NeuralObserverModel, encoder: SensoryEncoder, row: EmbodiedLesson
) -> Activation | None:
    history = row.history or (
        ()
        if row.previous_frame is None or row.previous_sensor is None
        else ((row.previous_frame, row.previous_sensor),)
    )
    prior = None
    for frame, sensor in history:
        prior = model(
            [frame],
            [prior],
            sensory=[
                encoder.activation(
                    pixels(sensor), torch.tensor(sensor.proprio), intent_activation(model, frame)
                )
            ],
        )[0].activation.detached()
    return prior


class GroundedTrainer:
    def __init__(self, observer: GroundedObserver, *, learning_rate: float = 0.002) -> None:
        if not math.isfinite(learning_rate) or not 0 < learning_rate < 1:
            raise ValidationError("Invalid grounded learning rate")
        self.observer = observer
        self.optimizer = torch.optim.AdamW(
            [*observer.observer.model.parameters(), *observer.observer.sensory.parameters()],
            lr=learning_rate,
            weight_decay=0.0001,
        )
        self.rng = random.Random(407)
        self.updates = 0
        self.native = Trainer(observer.observer.model)
        self.native.optimizer = self.optimizer
        self.native.rng = self.rng

    def train(
        self, lessons: list[EmbodiedLesson], *, epochs: int = 50, replay: list[Lesson] | None = None
    ) -> dict[str, float]:
        if not lessons or type(epochs) is not int or epochs < 1:
            raise ValidationError("Embodied training requires lessons and positive epochs")
        model, encoder = self.observer.observer.model, self.observer.observer.sensory
        # Visual grounding is already trained. Preserve it while fitting fusion,
        # the shared recurrent process and physical readouts; explicit visual lessons
        # can subsequently adapt these layers through train_vision again.
        for module in (encoder.image, encoder.foreground, encoder.appearance):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        model.train()
        encoder.train()
        totals = {"selection": 0.0, "prediction": 0.0, "loss": 0.0}
        batches = 0
        replay_groups = (
            []
            if not replay
            else [
                group
                for group in (
                    [row for row in replay if row.frame.meanings],
                    [row for row in replay if not row.frame.meanings],
                )
                if group
            ]
        )
        for _ in range(epochs):
            indices = list(range(len(lessons)))
            self.rng.shuffle(indices)
            for start in range(0, len(indices), 16):
                model.train()
                batch = [lessons[index] for index in indices[start : start + 16]]
                embeddings = [
                    encoder.activation(
                        pixels(row.sensor),
                        torch.tensor(row.sensor.proprio),
                        intent_activation(model, row.frame),
                    )
                    for row in batch
                ]
                with torch.no_grad():
                    priors = [lesson_prior(model, encoder, row) for row in batch]
                outputs = model([row.frame for row in batch], priors, sensory=embeddings)
                selection, prediction = [], []
                for row, output in zip(batch, outputs, strict=True):
                    target = row.frame.operations.index(row.operation)
                    selection.append(
                        F.cross_entropy(output.operation_logits[None], torch.tensor([target]))
                    )
                    physical = encoder.predict(
                        output.activation.context, row.action, torch.tensor(row.sensor.proprio)
                    )
                    prediction.append(F.mse_loss(physical, torch.tensor(row.outcome)))
                s, p = torch.stack(selection).mean(), torch.stack(prediction).mean()
                loss = s + p
                self.optimizer.zero_grad()
                torch.autograd.backward(loss)
                torch.nn.utils.clip_grad_norm_([*model.parameters(), *encoder.parameters()], 5.0)
                self.optimizer.step()
                self.updates += 1
                if replay_groups:
                    # Reuse the original lesson machinery, including carried state,
                    # ordered roles, applicability and all three feedback channels.
                    # One model, optimizer and owned RNG serve both kinds of update.
                    retained = [self.rng.choice(group) for group in replay_groups for _ in range(8)]
                    self.native.train(retained, epochs=1)
                    self.updates += 1
                batches += 1
                for key, value in (("selection", s), ("prediction", p), ("loss", loss)):
                    totals[key] += float(value.detach())
        model.eval()
        encoder.eval()
        return {key: value / batches for key, value in totals.items()}

    def learn_transition(
        self,
        before: SensorFrame,
        after: SensorFrame,
        action: str,
        *,
        preference: float = 0,
        frame: Frame | None = None,
        prior: Activation | None = None,
    ) -> float:
        """Measured prediction learning plus bounded consequential policy gradient."""
        if (
            action not in ACTIONS
            or not math.isfinite(preference)
            or after.tick != before.tick + 1
            or (before.width, before.height, before.focal, before.baseline)
            != (after.width, after.height, after.focal, after.baseline)
        ):
            raise ValidationError(
                "Learning requires one valid measured transition and finite preference"
            )
        observer = self.observer.observer
        frame = observer._frame if frame is None else frame
        if frame is None:
            raise ValidationError("Experience learning requires an active observation")
        embedding = observer.sensory.activation(
            pixels(before), torch.tensor(before.proprio), intent_activation(observer.model, frame)
        )
        output = observer.model([frame], [prior], sensory=[embedding])[0]
        action_index = ACTIONS.index(action)
        predicted = observer.sensory.predict(
            output.activation.context, action_index, torch.tensor(before.proprio)
        )
        discrepancy = F.mse_loss(predicted, torch.tensor(physical_outcome(before, after)))
        operation = self.observer.operations[action]
        log_probability = output.operation_logits.log_softmax(0)[frame.operations.index(operation)]
        # Preference adjusts behavior, never establishes perceptual truth.
        loss = discrepancy - max(-1.0, min(1.0, preference)) * log_probability * 0.05
        self.optimizer.zero_grad()
        torch.autograd.backward(loss)
        torch.nn.utils.clip_grad_norm_(
            [*observer.model.parameters(), *observer.sensory.parameters()], 5.0
        )
        self.optimizer.step()
        self.updates += 1
        return float(loss.detach())


def evaluate_actions(
    observer: GroundedObserver, material: list[EmbodiedLesson]
) -> dict[str, float]:
    correct, errors = 0, []
    with torch.no_grad():
        for row in material:
            embedding = observer.observer.sensory.activation(
                pixels(row.sensor),
                torch.tensor(row.sensor.proprio),
                intent_activation(observer.observer.model, row.frame),
            )
            prior = lesson_prior(observer.observer.model, observer.observer.sensory, row)
            output = observer.observer.model([row.frame], [prior], sensory=[embedding])[0]
            correct += int(
                output.operation_index < len(row.frame.operations)
                and row.frame.operations[output.operation_index] == row.operation
            )
            prediction = observer.observer.sensory.predict(
                output.activation.context, row.action, torch.tensor(row.sensor.proprio)
            )
            errors.append(float(F.mse_loss(prediction, torch.tensor(row.outcome))))
    return {
        "operation_accuracy": correct / len(material),
        "consequence_mse": sum(errors) / len(errors),
    }
