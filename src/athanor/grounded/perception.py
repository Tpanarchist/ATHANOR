"""Trainable pixel adapter, conservative stereo measurements and identity hypotheses."""

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn

from athanor.world import SensorFrame


def pixels(frame: SensorFrame) -> Tensor:
    return (
        torch.tensor([list(frame.left), list(frame.right)], dtype=torch.float32)
        .reshape(2, frame.height, frame.width, 3)
        .permute(0, 3, 1, 2)
        / 255
    )


class SensoryEncoder(nn.Module):
    def __init__(self, width: int = 64) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(414)
            self.image = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(),
            )
            self.foreground = nn.Conv2d(16, 1, 1)
            self.appearance = nn.Conv2d(16, 3, 1)
            self.fusion = nn.Linear(16 * 2 + 6 + width, width)
            self.physical = nn.Sequential(
                nn.Linear(width + 10 + 6, width), nn.Tanh(), nn.Linear(width, 6)
            )

    def encode(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features = cast(Tensor, self.image(images))
        return (
            cast(Tensor, self.foreground(features)).squeeze(1),
            cast(Tensor, self.appearance(features)).sigmoid(),
            features,
        )

    def activation(self, images: Tensor, proprio: Tensor, context: Tensor | None = None) -> Tensor:
        _, _, features = self.encode(images)
        summary = features.mean(dim=(-1, -2)).flatten()
        # Pixel activations and a compact view of intent enter the same learned
        # fusion. The context uses the existing shared structural encoder.
        if context is None:
            context = torch.zeros(self.fusion.out_features)
        return cast(Tensor, self.fusion(torch.cat((summary, proprio, context)))).tanh()

    def predict(self, context: Tensor, action: int, proprio: Tensor) -> Tensor:
        one_hot = torch.zeros(10, device=context.device)
        one_hot[action] = 1
        raw = cast(Tensor, self.physical(torch.cat((context, one_hot, proprio))))
        return torch.cat((raw[:2].tanh(), raw[2:].sigmoid()))


@dataclass(frozen=True, slots=True)
class Region:
    center: tuple[float, float]
    appearance: tuple[float, float, float]
    confidence: float
    area: int
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class Percept:
    appearance: tuple[float, float, float]
    relative: tuple[float, float, float] | None
    confidence: float
    status: str
    center: tuple[float, float]


def regions(mask: Tensor, appearance: Tensor) -> tuple[Region, ...]:
    """Exact connected components over learned foreground, not scene classification."""
    h, w = mask.shape
    cells = set((y, x) for y, x in (mask > 0.6).nonzero().tolist())
    output = []
    while cells:
        # Touching foreground silhouettes need not denote one participant. Use
        # the learned continuous appearance, including on new taught kinds, to
        # prevent one object/occluder from swallowing another's visible evidence.
        seed = min(cells, key=lambda cell: (-float(mask[cell]), cell))
        seed_descriptor = tuple(appearance[:, seed[0], seed[1]].tolist())
        cells.remove(seed)
        stack, component = [seed], []
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for other in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (
                    other in cells
                    and descriptor_distance(
                        seed_descriptor, tuple(appearance[:, other[0], other[1]].tolist())
                    )
                    < 0.35
                ):
                    cells.remove(other)
                    stack.append(other)
        if len(component) < 3:
            continue
        ys, xs = zip(*component, strict=True)
        descriptor = appearance[:, list(ys), list(xs)].mean(1).tolist()
        output.append(
            Region(
                (sum(xs) / len(xs) + 0.5, sum(ys) / len(ys) + 0.5),
                (descriptor[0], descriptor[1], descriptor[2]),
                float(mask[list(ys), list(xs)].mean()),
                len(xs),
                any(x in (0, w - 1) or y in (0, h - 1) for y, x in component),
            )
        )
    return tuple(output)


def descriptor_distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return float(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) ** 0.5)


def perceive(encoder: SensoryEncoder, frame: SensorFrame) -> tuple[Percept, ...]:
    encoder.eval()
    with torch.no_grad():
        logits, colors, _ = encoder.encode(pixels(frame))
    eyes = [regions(logits[i].sigmoid(), colors[i]) for i in range(2)]
    output = []
    for left in eyes[0]:
        matches = [
            r
            for r in eyes[1]
            if descriptor_distance(left.appearance, r.appearance) < 0.2
            and abs(left.center[1] - r.center[1]) < 2
        ]
        if len(matches) != 1:
            output.append(
                Percept(
                    left.appearance,
                    None,
                    left.confidence * 0.5,
                    "ambiguous" if matches else "monocular",
                    left.center,
                )
            )
            continue
        right = matches[0]
        if left.truncated or right.truncated:
            output.append(Percept(left.appearance, None, 0.4, "partial_observation", left.center))
            continue
        # Reciprocal correspondence is required; identical appearances stay uncertain.
        if (
            sum(descriptor_distance(right.appearance, other.appearance) < 0.2 for other in eyes[0])
            != 1
        ):
            output.append(Percept(left.appearance, None, 0.4, "ambiguous", left.center))
            continue
        disparity = left.center[0] - right.center[0]
        if disparity <= 0.35:
            output.append(Percept(left.appearance, None, 0.4, "depth_unknown", left.center))
            continue
        z = frame.focal * frame.baseline / disparity
        x = ((left.center[0] + right.center[0]) / 2 - frame.width / 2) * z / frame.focal
        y = (frame.height / 2 - (left.center[1] + right.center[1]) / 2) * z / frame.focal
        if not 0 < z < 12:
            output.append(Percept(left.appearance, None, 0.4, "depth_unknown", left.center))
            continue
        output.append(
            Percept(
                left.appearance,
                (x, y, z),
                min(left.confidence, right.confidence),
                "visible",
                left.center,
            )
        )
    return tuple(output)
