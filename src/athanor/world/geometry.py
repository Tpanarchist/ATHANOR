"""Small analytic 3D geometry, independently usable without PyTorch."""

import math

Vec3 = tuple[float, float, float]


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale(a: Vec3, value: float) -> Vec3:
    return (a[0] * value, a[1] * value, a[2] * value)


def distance(a: Vec3, b: Vec3) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def rotate(a: Vec3, yaw: float, pitch: float = 0) -> Vec3:
    x, y, z = a
    y, z = y * math.cos(pitch) + z * math.sin(pitch), z * math.cos(pitch) - y * math.sin(pitch)
    return (x * math.cos(yaw) + z * math.sin(yaw), y, z * math.cos(yaw) - x * math.sin(yaw))


def sphere_hit(origin: Vec3, direction: Vec3, center: Vec3, radius: float) -> float | None:
    delta = add(origin, scale(center, -1))
    b = sum(x * y for x, y in zip(delta, direction, strict=True))
    c = sum(x * x for x in delta) - radius * radius
    d = b * b - c
    if d < 0:
        return None
    near, far = -b - math.sqrt(d), -b + math.sqrt(d)
    return near if near > 0 else far if far > 0 else None


def box_hit(origin: Vec3, direction: Vec3, center: Vec3, size: Vec3) -> float | None:
    near, far = -math.inf, math.inf
    for o, d, c, s in zip(origin, direction, center, size, strict=True):
        if abs(d) < 1e-12:
            if not c - s <= o <= c + s:
                return None
            continue
        a, b = (c - s - o) / d, (c + s - o) / d
        near, far = max(near, min(a, b)), min(far, max(a, b))
    if near > far or far <= 0:
        return None
    return near if near > 0 else far


def overlap(position: Vec3, radius: float, center: Vec3, shape: str, size: Vec3) -> bool:
    if shape == "sphere":
        return distance(position, center) < radius + size[0] - 1e-9
    nearest = tuple(
        max(c - s, min(c + s, p)) for p, c, s in zip(position, center, size, strict=True)
    )
    return distance(position, (nearest[0], nearest[1], nearest[2])) < radius - 1e-9


def rigid_overlap(
    position: Vec3, shape: str, size: Vec3, other: Vec3, other_shape: str, other_size: Vec3
) -> bool:
    """Exact static sphere/sphere, sphere/AABB and AABB/AABB overlap."""
    if shape == "sphere":
        return overlap(position, size[0], other, other_shape, other_size)
    if other_shape == "sphere":
        return overlap(other, other_size[0], position, shape, size)
    return all(
        abs(a - b) < sa + sb - 1e-9
        for a, b, sa, sb in zip(position, other, size, other_size, strict=True)
    )
