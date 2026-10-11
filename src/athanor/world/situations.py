"""Independent procedural relational and symbolic raster environments.

The private program is an authority fixture. Public observations and demonstrations
are the entire learner interface; it never receives a program, role label or grouping.
"""

import random

from athanor import Athanor, Endpoint, FieldSpec, Procedure, Role, Value, record
from athanor.programs import argument, binary, call, field, literal, structural, variable


class ProblemStream:
    def __init__(self, seed: int, representation: str) -> None:
        if representation not in {"relational", "raster"}:
            raise ValueError("Unknown generated observation interface")
        self.representation = representation
        self.rng = random.Random(seed)
        self._authority = Athanor(namespace="private-generated-problem")
        category = self._authority.teach_category("private value", fields={"value": FieldSpec()})
        self._category = category
        value = field(argument(0), "value")
        self.property = "k" + str(self.rng.randrange(100000, 999999))
        if representation == "relational":
            degree = structural(
                self.rng.choice(("degree", "in_degree", "out_degree")),
                variable("root"),
                variable("index"),
            )
            choices = (
                binary("lt", literal(0), degree),
                binary("eq", degree, literal(0)),
                binary("lt", literal(1), degree),
            )
            predicate = self.rng.choice(choices)
            result = structural("reverse", structural("select", value, predicate))
        else:
            primitives: list[tuple[str, int | None]] = [
                (n, None) for n in ("transpose", "reflect_x", "reflect_y")
            ]
            primitives += [("shift_x", -1), ("shift_y", 1)]
            first = self.rng.choice(primitives)
            second = self.rng.choice([p for p in primitives if p != first])
            result = value
            for name, amount in (first, second):
                result = (
                    structural(name, result)
                    if amount is None
                    else structural(name, result, literal(amount))
                )
        self._program = self._authority.teach_operation(
            "private generated transformation",
            Procedure((Endpoint(Role.DATA, category),), category, {"value": result}),
        )
        self._raster_adapter = self._authority.teach_operation(
            "private interface adapter",
            Procedure(
                (Endpoint(Role.DATA, category),),
                category,
                {"value": structural("raster_graph", value)},
            ),
        )
        self._raster_program = self._authority.teach_operation(
            "private represented consequence",
            Procedure(
                (Endpoint(Role.DATA, category),),
                category,
                {
                    "value": structural(
                        "rasterize",
                        field(
                            call(self._program, call(self._raster_adapter, argument(0))), "value"
                        ),
                    )
                },
            ),
        )

    def observe(self) -> Value:
        rng = self.rng
        if self.representation == "relational":
            count = rng.randrange(7, 12)
            nodes = tuple(
                record(
                    {
                        self.property: "s" + str(rng.randrange(10**9)),
                        "tag": rng.choice(("u", "v", "w")),
                    }
                )
                for _ in range(count)
            )
            participants = rng.sample(range(count), count - 3)
            edges = tuple(
                (participants[i], participants[i + 1]) for i in range(len(participants) - 1)
            )
            return record(nodes=nodes, edges=edges)
        width, height = rng.randrange(5, 9), rng.randrange(4, 8)
        return tuple(
            tuple(rng.choice((None, None, None, "a", "b", "c")) for _ in range(width))
            for _ in range(height)
        )

    def demonstrate(self, observation: Value) -> Value:
        datum = self._authority.create(self._category, {"value": observation})
        result = self._authority.execute(self._program, (datum,))
        return self._authority.get(result).fields["value"]

    def demonstrate_raster(self, observation: Value) -> Value:
        datum = self._authority.create(self._category, {"value": observation})
        result = self._authority.execute(self._raster_program, (datum,))
        return self._authority.get(result).fields["value"]
