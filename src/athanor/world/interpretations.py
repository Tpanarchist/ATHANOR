"""Independent correspondence observations, without observer or solution authority.

Private fixture parameters generate raw collections and RGB. The observer receives only
these observations and outcomes of correspondence queries, never grouping assignments.
"""

import random
from collections.abc import Sequence

from athanor import Value, record
from athanor.values import as_tuple

from .engine import BACKGROUND, SensorFrame


class Correspondences:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.key = f"k{self.rng.randrange(100000)}"
        self.nuisance = f"u{self.rng.randrange(100000)}"
        self.tick = 0

    def collection(self, sizes: Sequence[int]) -> Value:
        rows: list[Value] = []
        for size in sizes:
            symbol = f"v{self.rng.randrange(1000000)}"
            rows.extend(
                record({self.key: symbol, self.nuisance: f"v{self.rng.randrange(1000000)}"})
                for _ in range(size)
            )
        self.rng.shuffle(rows)
        return tuple(rows)

    def raster(
        self, sizes: Sequence[int], *, offset: tuple[int, int] = (0, 0), appearance: str = "x"
    ) -> Value:
        width, height = 28, 20
        rows: list[list[Value]] = [[None] * width for _ in range(height)]
        for index, size in enumerate(sizes):
            x, y = 3 + index * 7 + offset[0], 4 + offset[1]
            for yy in range(y, y + size):
                for xx in range(x, x + 3):
                    if 0 <= xx < width and 0 <= yy < height:
                        rows[yy][xx] = appearance
        return tuple(tuple(row) for row in rows)

    def camera(
        self,
        sizes: Sequence[int],
        *,
        offset: tuple[int, int] = (0, 0),
        color: tuple[int, int, int] = (200, 70, 50),
    ) -> SensorFrame:
        raster = self.raster(sizes, offset=offset)
        rgb = bytes(
            channel
            for row in as_tuple(raster)
            for value in as_tuple(row)
            for channel in (BACKGROUND if value is None else color)
        )
        self.tick += 1
        return SensorFrame(
            self.tick, 28, 20, rgb, rgb, 20.0, 0.2, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), "wait"
        )
