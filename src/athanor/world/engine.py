"""Native authoritative territory with a deliberately narrow pixel-only sensor port."""

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import cast

from athanor import Athanor, Endpoint, Procedure, Record, Ref, Role, ValidationError, record
from athanor.programs import argument, field
from athanor.values import as_record, as_ref, as_tuple

from .geometry import (
    Vec3,
    add,
    box_hit,
    distance,
    overlap,
    rigid_overlap,
    rotate,
    scale,
    sphere_hit,
)

BACKGROUND = (18, 22, 30)
ACTIONS = (
    "forward",
    "backward",
    "left",
    "right",
    "turn_left",
    "turn_right",
    "look_up",
    "look_down",
    "interact",
    "wait",
)
OBJECT_FIELDS = ("kind", "position", "size", "color", "shape", "velocity", "movable")


def exact(value: float) -> Fraction:
    return Fraction(f"{value:.9f}")


def vector(value: object) -> Vec3:
    values = as_tuple(value)  # type: ignore[arg-type]
    if len(values) != 3:
        raise ValidationError("A 3D vector requires three coordinates")
    return (
        float(cast(Fraction, values[0])),
        float(cast(Fraction, values[1])),
        float(cast(Fraction, values[2])),
    )


@dataclass(frozen=True, slots=True)
class Action:
    name: str = "wait"

    def __post_init__(self) -> None:
        if self.name not in ACTIONS:
            raise ValidationError("Unknown embodied action")


@dataclass(frozen=True, slots=True)
class SensorFrame:
    """No references, IDs, depth maps, labels, absolute positions or fault flag."""

    tick: int
    width: int
    height: int
    left: bytes
    right: bytes
    focal: float
    baseline: float
    proprio: tuple[float, float, float, float, float, float]
    executed: str

    def __post_init__(self) -> None:
        if (
            type(self.width) is not int
            or type(self.height) is not int
            or not 8 <= self.width <= 128
            or not 8 <= self.height <= 128
        ):
            raise ValidationError("Camera dimensions outside supported bounds")
        if type(self.left) is not bytes or type(self.right) is not bytes:
            raise ValidationError("Camera inputs require immutable raw RGB bytes")
        if len(self.left) != self.width * self.height * 3 or len(self.right) != len(self.left):
            raise ValidationError("Malformed RGB frame")
        if (
            type(self.tick) is not int
            or self.tick < 0
            or type(self.focal) not in (int, float)
            or type(self.baseline) not in (int, float)
            or not math.isfinite(self.focal)
            or self.focal <= 0
            or not math.isfinite(self.baseline)
            or self.baseline <= 0
        ):
            raise ValidationError("Invalid camera time/calibration")
        if (
            type(self.proprio) is not tuple
            or len(self.proprio) != 6
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in self.proprio)
        ):
            raise ValidationError("Invalid limited proprioception")
        Action(self.executed)


class World:
    def __init__(
        self,
        *,
        namespace: str = "territory",
        resolution: int = 24,
        interaction_distance: float = 0.2,
    ) -> None:
        if type(resolution) is not int or not 8 <= resolution <= 128:
            raise ValidationError("Invalid camera resolution")
        if (
            type(interaction_distance) not in (int, float)
            or not math.isfinite(interaction_distance)
            or not 0 <= interaction_distance <= 0.5
        ):
            raise ValidationError("Invalid bounded territory interaction response")
        self.interaction_distance = float(interaction_distance)
        self.interaction_process: Ref | None = None
        self.motion_process: Ref | None = None
        self.motion_value: Ref | None = None
        self.knowledge = Athanor(namespace=namespace)
        self.resolution = resolution
        self.bounds: Vec3 = (6, 3, 6)
        self.objects: list[Ref] = []
        self.tick = 0
        self.yaw = 0.0
        self.pitch = 0.0
        self.motor = True
        self.camera_hold = False
        self._held: tuple[bytes, bytes] | None = None
        self.proprio = (0.0,) * 6
        self.executed = "wait"
        self.events: list[Ref] = []
        self.updater = self.knowledge.teach_operation(
            "rigid state transition",
            Procedure(
                (Endpoint(Role.DATA), Endpoint(Role.DATA)),
                None,
                {name: field(argument(1), name) for name in OBJECT_FIELDS},
                update=0,
            ),
        )
        self.body = self.add_object(
            (0, 0, 0), shape="sphere", size=(0.2, 0.2, 0.2), color=(230, 230, 230)
        )
        self.objects.remove(self.body)

    @property
    def position(self) -> Vec3:
        return vector(self.knowledge.get(self.body).fields["position"])

    def set_interaction_process(self, operation: Ref) -> None:
        """Authority-only native surface-gap law; never exported through SensorFrame."""
        spec = self.knowledge.get(operation).fields.get("procedure")
        if (
            self.knowledge.get(operation).role != Role.OPERATION
            or not isinstance(spec, Record)
            or len(as_tuple(spec["inputs"])) != 1
            or not isinstance(spec["result"], Ref)
            or as_record(as_tuple(spec["inputs"])[0])["category"] != spec["result"]
            or spec["update"] is not None
            or "value" not in as_record(spec["fields"])
        ):
            raise ValidationError(
                "Interaction process requires one typed quantity and quantity result"
            )
        self.interaction_process = operation

    def set_motion_process(self, operation: Ref, *, initial: Fraction = Fraction(1, 4)) -> None:
        """Private actuator state evolves by a bounded native quantity process."""
        prior = self.interaction_process
        self.set_interaction_process(operation)
        self.interaction_process = prior
        spec = as_record(self.knowledge.get(operation).fields["procedure"])
        if not isinstance(initial, Fraction) or not 0 < initial <= Fraction(1, 2):
            raise ValidationError("Invalid initial bounded actuator magnitude")
        self.motion_value = self.knowledge.create(as_ref(spec["result"]), {"value": initial})
        self.motion_process = operation

    def add_object(
        self,
        position: Vec3,
        *,
        shape: str = "sphere",
        size: Vec3 = (0.4, 0.4, 0.4),
        color: tuple[int, int, int] = (230, 50, 40),
        velocity: Vec3 = (0, 0, 0),
        movable: bool = False,
    ) -> Ref:
        if (
            len(position) != 3
            or len(size) != 3
            or len(velocity) != 3
            or len(color) != 3
            or type(movable) is not bool
        ):
            raise ValidationError("Rigid objects require explicit 3D vectors, RGB and mobility")
        if shape not in {"sphere", "box"} or any(not math.isfinite(x) or x <= 0 for x in size):
            raise ValidationError("Invalid rigid geometry")
        if shape == "sphere" and len(set(size)) != 1:
            raise ValidationError("Spheres require one radius repeated in all axes")
        if any(type(x) is not int or not 0 <= x <= 255 for x in color):
            raise ValidationError("Invalid RGB material")
        if any(
            not math.isfinite(p) or abs(p) + s > b
            for p, s, b in zip(position, size, self.bounds, strict=True)
        ):
            raise ValidationError("Object lies outside bounded territory")
        if any(not math.isfinite(v) or abs(v) > 2 for v in velocity):
            raise ValidationError("Invalid bounded velocity")
        ref = self.knowledge.create(
            fields=record(
                kind="rigid_object",
                position=tuple(exact(v) for v in position),
                size=tuple(exact(v) for v in size),
                color=color,
                shape=shape,
                velocity=tuple(exact(v) for v in velocity),
                movable=movable,
            ),
            source="territory initialization",
        )
        self.objects.append(ref)
        return ref

    def relocate(self, ref: Ref, position: Vec3) -> Ref:
        """Authority/editor only. Never supplied to observer inference."""
        if ref != self.body and ref not in self.objects:
            raise ValidationError("Territory transition needs a current owned participant")
        if len(position) != 3:
            raise ValidationError("Territory positions require exactly three coordinates")
        node = self.knowledge.get(ref)
        size = vector(node.fields["size"])
        if any(
            not math.isfinite(p) or abs(p) + s > b
            for p, s, b in zip(position, size, self.bounds, strict=True)
        ):
            raise ValidationError("Position violates bounds")
        fields: dict[str, object] = dict(node.fields)
        fields["position"] = tuple(exact(v) for v in position)
        next_state = self.knowledge.create(fields=fields, source="territory transition input")
        new = self.knowledge.execute(self.updater, (ref, next_state))
        if ref == self.body:
            self.body = new
        else:
            self.objects[self.objects.index(ref)] = new
        return new

    def _blocked(self, position: Vec3) -> bool:
        if any(abs(p) + 0.2 > b for p, b in zip(position, self.bounds, strict=True)):
            return True
        return any(
            overlap(
                position,
                0.2,
                vector(node.fields["position"]),
                str(node.fields["shape"]),
                vector(node.fields["size"]),
            )
            for node in (self.knowledge.get(ref) for ref in self.objects)
        )

    def step(self, action: Action | None = None) -> Ref:
        action = Action() if action is None else action
        if not isinstance(action, Action):
            raise ValidationError("World execution requires a validated Action")
        old = self.position
        local = {
            "forward": (0.0, 0.0, 0.25),
            "backward": (0.0, 0.0, -0.25),
            "left": (-0.25, 0.0, 0.0),
            "right": (0.25, 0.0, 0.0),
        }.get(action.name, (0.0, 0.0, 0.0))
        delta = rotate(local, self.yaw)
        contact = False
        # Swept integration in 25 mm substeps, with all authoritative adoption atomic.
        snapshot = (
            self.body,
            list(self.objects),
            self.tick,
            self.yaw,
            self.pitch,
            self.motion_value,
        )
        try:

            def transition(a: Athanor) -> Ref:
                original = self.knowledge
                self.knowledge = a
                try:
                    nonlocal contact, delta
                    if self.motion_process is not None and any(delta) and self.motor:
                        assert self.motion_value is not None
                        self.motion_value = a.execute(self.motion_process, (self.motion_value,))
                        magnitude = float(cast(Fraction, a.get(self.motion_value).fields["value"]))
                        if not math.isfinite(magnitude) or not 0 < magnitude <= 0.5:
                            raise ValidationError("Actuator process exceeds bounded motion")
                        delta = scale(delta, magnitude / distance(delta, (0, 0, 0)))
                    for ref in tuple(self.objects):
                        node = a.get(ref)
                        velocity = vector(node.fields["velocity"])
                        proposed = add(vector(node.fields["position"]), scale(velocity, 0.1))
                        clear = not any(
                            rigid_overlap(
                                proposed,
                                str(node.fields["shape"]),
                                vector(node.fields["size"]),
                                vector(a.get(other).fields["position"]),
                                str(a.get(other).fields["shape"]),
                                vector(a.get(other).fields["size"]),
                            )
                            for other in self.objects
                            if other != ref
                        )
                        if (
                            clear
                            and all(
                                abs(p) + s <= b
                                for p, s, b in zip(
                                    proposed, vector(node.fields["size"]), self.bounds, strict=True
                                )
                            )
                            and not overlap(
                                self.position,
                                0.2,
                                proposed,
                                str(node.fields["shape"]),
                                vector(node.fields["size"]),
                            )
                        ):
                            if any(velocity):
                                self.relocate(ref, proposed)
                    if self.motor:
                        for _ in range(10):
                            proposed = add(self.position, scale(delta, 0.1))
                            if self._blocked(proposed):
                                contact = True
                                break
                            if any(delta):
                                self.relocate(self.body, proposed)
                    if action.name == "turn_left":
                        self.yaw -= math.pi / 12
                    if action.name == "turn_right":
                        self.yaw += math.pi / 12
                    if action.name == "look_up":
                        self.pitch = min(0.6, self.pitch + 0.15)
                    if action.name == "look_down":
                        self.pitch = max(-0.6, self.pitch - 0.15)
                    affected: list[Ref] = []
                    if action.name == "interact" and self.motor:
                        for ref in tuple(self.objects):
                            node = a.get(ref)
                            if (
                                node.fields["movable"]
                                and distance(self.position, vector(node.fields["position"])) < 1.2
                            ):
                                displacement = self.interaction_distance
                                if self.interaction_process is not None:
                                    spec = as_record(
                                        a.get(self.interaction_process).fields["procedure"]
                                    )
                                    gap = (
                                        rotate(
                                            add(
                                                vector(node.fields["position"]),
                                                scale(self.position, -1),
                                            ),
                                            -self.yaw,
                                        )[2]
                                        - vector(node.fields["size"])[2]
                                    )
                                    input_ref = a.create(
                                        as_ref(spec["result"]), {"value": exact(gap)}
                                    )
                                    result = a.execute(self.interaction_process, (input_ref,))
                                    next_gap = float(cast(Fraction, a.get(result).fields["value"]))
                                    if (
                                        not math.isfinite(next_gap)
                                        or not 0 < next_gap <= 6
                                        or abs(next_gap - gap) > 1
                                    ):
                                        raise ValidationError(
                                            "Native law exceeds bounded physical consequences"
                                        )
                                    displacement = next_gap - gap
                                pushed = add(
                                    vector(node.fields["position"]),
                                    rotate((0, 0, displacement), self.yaw),
                                )
                                clear = not any(
                                    rigid_overlap(
                                        pushed,
                                        str(node.fields["shape"]),
                                        vector(node.fields["size"]),
                                        vector(a.get(other).fields["position"]),
                                        str(a.get(other).fields["shape"]),
                                        vector(a.get(other).fields["size"]),
                                    )
                                    for other in self.objects
                                    if other != ref
                                )
                                if clear and all(
                                    abs(p) + s <= b
                                    for p, s, b in zip(
                                        pushed,
                                        vector(node.fields["size"]),
                                        self.bounds,
                                        strict=True,
                                    )
                                ):
                                    affected.append(self.relocate(ref, pushed))
                    self.tick += 1
                    contact = contact or bool(affected)
                    return a.create(
                        fields=record(
                            kind="causal_event",
                            tick=self.tick,
                            intended=action.name,
                            executed=action.name if self.motor or not any(delta) else "wait",
                            body=self.body,
                            before=tuple(exact(v) for v in old),
                            after=tuple(exact(v) for v in self.position),
                            contact=contact,
                            affected=tuple(affected),
                        ),
                        source="authoritative fixed-step physics",
                    )
                finally:
                    self.knowledge = original

            event = self.knowledge.transaction(transition)
        except Exception:
            self.body, self.objects, self.tick, self.yaw, self.pitch, self.motion_value = snapshot
            raise
        self.events.append(event)
        actual = rotate(add(self.position, scale(old, -1)), -snapshot[3])
        self.proprio = (*actual, self.yaw - snapshot[3], float(contact), float(bool(any(actual))))
        self.executed = str(self.knowledge.get(event).fields["executed"])
        return event

    def render(self, *, labels: bool = False) -> tuple[bytes, bytes, tuple[tuple[int, ...], ...]]:
        """Labels are a separate training-authority API, never part of SensorFrame."""
        images: list[bytes] = []
        masks: list[tuple[int, ...]] = []
        n = self.resolution
        focal = n / (2 * math.tan(math.radians(70) / 2))
        nodes = [self.knowledge.get(ref) for ref in self.objects]
        for eye in (-0.12, 0.12):
            origin = add(self.position, rotate((eye, 0.05, 0.15), self.yaw, self.pitch))
            pixels = bytearray()
            mask: list[int] = []
            for row in range(n):
                for col in range(n):
                    direction = ((col + 0.5 - n / 2) / focal, (n / 2 - row - 0.5) / focal, 1.0)
                    direction = rotate(
                        scale(direction, 1 / math.sqrt(sum(v * v for v in direction))),
                        self.yaw,
                        self.pitch,
                    )
                    hits = []
                    for i, node in enumerate(nodes):
                        pos, size = vector(node.fields["position"]), vector(node.fields["size"])
                        hit = (
                            sphere_hit(origin, direction, pos, size[0])
                            if node.fields["shape"] == "sphere"
                            else box_hit(origin, direction, pos, size)
                        )
                        if hit is not None:
                            hits.append((hit, i))
                    if hits:
                        _, index = min(hits)
                        color = tuple(
                            int(cast(Fraction, v)) for v in as_tuple(nodes[index].fields["color"])
                        )
                        pixels.extend(color)
                        mask.append(index + 1)
                    else:
                        pixels.extend(BACKGROUND)
                        mask.append(0)
            images.append(bytes(pixels))
            if labels:
                masks.append(tuple(mask))
        return images[0], images[1], tuple(masks)

    def sense(self) -> SensorFrame:
        left, right, _ = self.render()
        if self.camera_hold and self._held is not None:
            left, right = self._held
        elif not self.camera_hold:
            self._held = (left, right)
        n = self.resolution
        return SensorFrame(
            self.tick,
            n,
            n,
            left,
            right,
            n / (2 * math.tan(math.radians(70) / 2)),
            0.24,
            self.proprio,
            self.executed,
        )

    def dumps(self) -> str:
        import json

        return json.dumps(
            {
                "version": 2
                if self.interaction_process is None and self.motion_process is None
                else 3,
                "interaction_distance": self.interaction_distance,
                **(
                    {}
                    if self.interaction_process is None and self.motion_process is None
                    else {
                        "interaction_process": None
                        if self.interaction_process is None
                        else [
                            self.interaction_process.id,
                            self.interaction_process.revision,
                            self.interaction_process.namespace,
                        ],
                        "motion_process": None
                        if self.motion_process is None
                        else [
                            self.motion_process.id,
                            self.motion_process.revision,
                            self.motion_process.namespace,
                        ],
                        "motion_value": None
                        if self.motion_value is None
                        else [
                            self.motion_value.id,
                            self.motion_value.revision,
                            self.motion_value.namespace,
                        ],
                    }
                ),
                "knowledge": self.knowledge.dumps(),
                "resolution": self.resolution,
                "bounds": self.bounds,
                "objects": [[r.id, r.revision, r.namespace] for r in self.objects],
                "body": [self.body.id, self.body.revision, self.body.namespace],
                "updater": [self.updater.id, self.updater.revision, self.updater.namespace],
                "tick": self.tick,
                "yaw": self.yaw,
                "pitch": self.pitch,
                "motor": self.motor,
                "camera_hold": self.camera_hold,
                "held": None if self._held is None else [v.hex() for v in self._held],
                "proprio": self.proprio,
                "executed": self.executed,
                "events": [[r.id, r.revision, r.namespace] for r in self.events],
            },
            allow_nan=False,
        )

    @classmethod
    def loads(cls, text: str) -> World:
        import json

        try:
            data = json.loads(text)
            expected = {
                "version",
                "knowledge",
                "resolution",
                "bounds",
                "objects",
                "body",
                "updater",
                "tick",
                "yaw",
                "pitch",
                "motor",
                "camera_hold",
                "held",
                "proprio",
                "executed",
                "events",
            }
            if isinstance(data, dict) and data.get("version") in (2, 3):
                expected.add("interaction_distance")
                if data["version"] == 3:
                    expected.add("interaction_process")
                    expected.update({"motion_process", "motion_value"})
            if (
                type(data) is not dict
                or set(data) != expected
                or type(data["version"]) is not int
                or data["version"] not in (1, 2, 3)
            ):
                raise ValidationError("Unsupported world snapshot")
            obj = cls(
                resolution=data["resolution"],
                interaction_distance=data.get("interaction_distance", 0.2),
            )
            obj.knowledge = Athanor.loads(data["knowledge"])

            def ref(raw: list[object]) -> Ref:
                if (
                    type(raw) is not list
                    or len(raw) != 3
                    or type(raw[0]) is not str
                    or type(raw[1]) is not int
                    or type(raw[2]) is not str
                ):
                    raise ValidationError("Invalid exact territory reference")
                r = Ref(raw[0], raw[1], raw[2])
                obj.knowledge.get(r)
                return r

            obj.objects = [ref(r) for r in data["objects"]]
            obj.body, obj.updater = ref(data["body"]), ref(data["updater"])
            obj.events = [ref(r) for r in data["events"]]
            if data["version"] == 3:
                if data["interaction_process"] is not None:
                    obj.set_interaction_process(ref(data["interaction_process"]))
                if (data["motion_process"] is None) != (data["motion_value"] is None):
                    raise ValidationError("Actuator law/state disagree")
                if data["motion_process"] is not None:
                    prior = obj.interaction_process
                    obj.set_interaction_process(ref(data["motion_process"]))
                    obj.motion_process, obj.interaction_process = obj.interaction_process, prior
                    assert obj.motion_process is not None
                    obj.motion_value = ref(data["motion_value"])
                    quantity = obj.knowledge.get(obj.motion_value)
                    spec = as_record(obj.knowledge.get(obj.motion_process).fields["procedure"])
                    if quantity.category != spec["result"] or not 0 < cast(
                        Fraction, quantity.fields["value"]
                    ) <= Fraction(1, 2):
                        raise ValidationError("Invalid persisted actuator quantity")
            obj.bounds = tuple(data["bounds"])
            obj.tick, obj.yaw, obj.pitch = data["tick"], data["yaw"], data["pitch"]
            obj.motor, obj.camera_hold = data["motor"], data["camera_hold"]
            obj._held = (
                None
                if data["held"] is None
                else (bytes.fromhex(data["held"][0]), bytes.fromhex(data["held"][1]))
            )
            obj.proprio, obj.executed = tuple(data["proprio"]), data["executed"]
            if (
                obj.bounds != (6, 3, 6)
                or type(obj.tick) is not int
                or obj.tick < 0
                or type(obj.motor) is not bool
                or type(obj.camera_hold) is not bool
                or any(not math.isfinite(v) for v in (obj.yaw, obj.pitch))
                or not -0.6 <= obj.pitch <= 0.6
                or len(set(obj.objects)) != len(obj.objects)
                or obj.body in obj.objects
            ):
                raise ValidationError("Invalid territory configuration")
            for r in (obj.body, *obj.objects):
                node = obj.knowledge.get(r)
                size = vector(node.fields["size"])
                position = vector(node.fields["position"])
                color = as_tuple(node.fields["color"])
                velocity = vector(node.fields["velocity"])
                if (
                    node.role != Role.DATA
                    or node.fields["kind"] != "rigid_object"
                    or type(node.fields["movable"]) is not bool
                    or len(color) != 3
                    or any(
                        not isinstance(c, Fraction) or c.denominator != 1 or not 0 <= c <= 255
                        for c in color
                    )
                    or any(not math.isfinite(v) or abs(v) > 2 for v in velocity)
                    or node.fields["shape"] == "sphere"
                    and len(set(size)) != 1
                    or obj.knowledge.get(r.id).ref != r
                    or node.fields["shape"] not in {"sphere", "box"}
                    or any(not math.isfinite(s) or s <= 0 for s in size)
                    or any(
                        not math.isfinite(p) or abs(p) + s > b
                        for p, s, b in zip(position, size, obj.bounds, strict=True)
                    )
                ):
                    raise ValidationError("Invalid persistent rigid geometry")
            body = obj.knowledge.get(obj.body)
            if (
                body.fields["shape"] != "sphere"
                or vector(body.fields["size"]) != (0.2, 0.2, 0.2)
                or obj.knowledge.get(obj.updater).role != Role.OPERATION
            ):
                raise ValidationError("Invalid observer body or world transition")
            if any(obj.knowledge.get(r).fields.get("kind") != "causal_event" for r in obj.events):
                raise ValidationError("World event history must contain causal receipts")
            if obj._held is not None and any(
                len(image) != obj.resolution**2 * 3 for image in obj._held
            ):
                raise ValidationError("Malformed persistent camera state")
            # Validation must not replace a saved capture cache with a new view.
            # A later camera hold may legitimately replay that earlier capture.
            held = obj._held
            obj.sense()
            obj._held = held
            return obj
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            raise ValidationError("Invalid world snapshot") from exc
