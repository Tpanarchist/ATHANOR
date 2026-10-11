"""Exact bounded structural algorithms for the cornerstone expression algebra.

No hypotheses or goals live here: these are generic functions on canonical values.
"""

from collections.abc import Callable, Mapping
from fractions import Fraction

from .errors import ExecutionError, ValidationError
from .values import Record, Value, as_index, as_record, as_tuple, exact_equal, record

ARITIES = {
    "count": 1,
    "reverse": 1,
    "unique": 1,
    "nodes": 1,
    "components": 1,
    "transpose": 1,
    "reflect_x": 1,
    "reflect_y": 1,
    "raster_graph": 1,
    "rasterize": 1,
    "degree": 2,
    "in_degree": 2,
    "out_degree": 2,
    "frequency": 2,
    "at": 2,
    "concat": 2,
    "shift_x": 2,
    "shift_y": 2,
    "not": 1,
    "and": 2,
    "or": 2,
    "select": 2,
    "sort": 2,
    "map": 2,
    "exists": 2,
}
SCOPED = frozenset({"select", "sort", "map", "exists"})


def validate_value(value: Value) -> None:
    pending = [(value, 0)]
    visited = 0
    while pending:
        v, depth = pending.pop()
        visited += 1
        if visited > 8192 or depth > 16:
            raise ValidationError("Structural observation exceeds bounded size/depth")
        if isinstance(v, Record):
            pending.extend((x, depth + 1) for x in v.values())
        elif isinstance(v, tuple):
            if len(v) > 4096:
                raise ValidationError("Structural collection exceeds observation budget")
            pending.extend((x, depth + 1) for x in v)
        elif isinstance(v, str) and len(v) > 4096:
            raise ValidationError("Structural symbol exceeds observation budget")
    if isinstance(value, Record) and {"nodes", "edges"} <= set(value):
        graph(value)


def graph(value: Value) -> tuple[tuple[Value, ...], tuple[tuple[int, int], ...]]:
    v = as_record(value, "graph")
    nodes = as_tuple(v.get("nodes"))
    if len(nodes) > 128 or any(not isinstance(n, Record) for n in nodes):
        raise ExecutionError("Graph requires at most 128 node records")
    edges = []
    for pair in as_tuple(v.get("edges")):
        p = as_tuple(pair)
        if len(p) != 2:
            raise ExecutionError("A directed edge requires two ordered indices")
        a, b = as_index(p[0]), as_index(p[1])
        if a >= len(nodes) or b >= len(nodes):
            raise ExecutionError("Graph edge participant is unavailable")
        edges.append((a, b))
    if len(edges) > 1024 or len(set(edges)) != len(edges):
        raise ExecutionError("Graph edges must be bounded and unique")
    return nodes, tuple(edges)


def collection(value: Value) -> tuple[Value, ...]:
    return graph(value)[0] if isinstance(value, Record) else as_tuple(value)


def grid(value: Value) -> tuple[tuple[Value, ...], ...]:
    rows = tuple(as_tuple(v) for v in as_tuple(value))
    if not rows or not rows[0] or len(rows) > 64 or len(rows[0]) > 64:
        raise ExecutionError("Raster requires nonempty bounded rows")
    if any(len(row) != len(rows[0]) for row in rows):
        raise ExecutionError("Raster rows must be rectangular")
    if any(isinstance(v, (Record, tuple)) for row in rows for v in row):
        raise ExecutionError("Raster cells require atomic canonical values")
    return rows


def family(value: Value) -> str:
    if isinstance(value, Record) and {"nodes", "edges"} <= set(value):
        graph(value)
        return "graph"
    if isinstance(value, tuple):
        if value and all(isinstance(v, tuple) for v in value):
            try:
                grid(value)
                return "raster"
            except ExecutionError:
                pass
        return "sequence"
    return "record" if isinstance(value, Record) else "atom"


def reindex(value: Value, indices: list[int]) -> Value:
    items = collection(value)
    if len(set(indices)) != len(indices):
        raise ExecutionError("A graph ordering cannot duplicate participant roles")
    if not isinstance(value, Record):
        return tuple(items[i] for i in indices)
    _, edges = graph(value)
    reverse = {old: new for new, old in enumerate(indices)}
    return record(
        dict(value),
        nodes=tuple(items[i] for i in indices),
        edges=tuple((reverse[a], reverse[b]) for a, b in edges if a in reverse and b in reverse),
    )


def evaluate(
    operator: str,
    expressions: tuple[Value, ...],
    child: Callable[[Value, Mapping[str, Value]], Value],
    scope: Mapping[str, Value],
    tick: Callable[[], None],
) -> Value:
    source = child(expressions[0], scope)
    if operator in SCOPED:
        items = collection(source)
        output: list[Value] = []
        indices: list[int] = []
        unknown = False
        for index, value in enumerate(items):
            tick()
            local = dict(scope)
            if operator == "exists":
                if source is scope.get("root") and scope.get("index") == Fraction(index):
                    continue
                local.update(peer=value, peer_index=Fraction(index))
            else:
                local.update(item=value, index=Fraction(index), root=source)
            result = child(expressions[1], local)
            if operator in {"select", "exists"}:
                if result is None:
                    unknown = True
                elif type(result) is not bool:
                    raise ExecutionError("Structural predicate requires boolean or unknown")
                elif result:
                    if operator == "exists":
                        return True
                    indices.append(index)
            else:
                output.append(result)
        if operator == "exists":
            return None if unknown else False
        if operator == "select":
            return None if unknown else reindex(source, indices)
        if operator == "map":
            return tuple(output)
        if any(v is None for v in output):
            return None
        if not output:
            return reindex(source, [])
        if any(type(v) is not type(output[0]) for v in output) or not isinstance(
            output[0], (Fraction, str, bool)
        ):
            raise ExecutionError("Ordering keys require comparable values of the same exact type")
        return reindex(source, sorted(range(len(items)), key=lambda i: output[i]))  # type: ignore[arg-type,return-value]
    other = child(expressions[1], scope) if len(expressions) == 2 else None
    if operator in {"not", "and", "or"}:
        if (
            source is not None
            and type(source) is not bool
            or other is not None
            and type(other) is not bool
        ):
            raise ExecutionError("Logical operands require boolean or unknown")
        if operator == "not":
            return None if source is None else not source
        if operator == "and":
            return (
                False
                if source is False or other is False
                else None
                if source is None or other is None
                else True
            )
        return (
            True
            if source is True or other is True
            else None
            if source is None or other is None
            else False
        )
    if operator == "at":
        items = collection(source)
        i = as_index(other)
        if i >= len(items):
            raise ExecutionError("Participant index out of range")
        return items[i]
    if operator == "count":
        return Fraction(len(collection(source)))
    if operator == "nodes":
        return graph(source)[0]
    if operator == "frequency":
        return Fraction(sum(exact_equal(v, other) for v in collection(source)))
    if operator == "concat":
        values = as_tuple(source) + as_tuple(other)
        if len(values) > 4096:
            raise ExecutionError("Concatenation exceeds bounded size")
        return values
    if operator == "reverse":
        return reindex(source, list(reversed(range(len(collection(source))))))
    if operator == "unique":
        values = collection(source)
        indices = []
        for i, v in enumerate(values):
            tick()
            if not any(exact_equal(v, values[j]) for j in indices):
                indices.append(i)
        return reindex(source, indices)
    if operator in {"degree", "in_degree", "out_degree", "components"}:
        nodes, edges = graph(source)
        if operator in {"degree", "in_degree", "out_degree"}:
            i = as_index(other)
            if i >= len(nodes):
                raise ExecutionError("Relational role outside graph")
            return Fraction(
                sum(
                    (
                        a == i
                        if operator == "out_degree"
                        else b == i
                        if operator == "in_degree"
                        else a == i or b == i
                    )
                    for a, b in edges
                )
            )
        pending = set(range(len(nodes)))
        groups = []
        while pending:
            start = min(pending)
            found, queue = {start}, [start]
            while queue:
                current = queue.pop()
                for a, b in edges:
                    tick()
                    neighbor = b if a == current else a if b == current else None
                    if neighbor is not None and neighbor not in found:
                        found.add(neighbor)
                        queue.append(neighbor)
            pending -= found
            groups.append(tuple(Fraction(i) for i in sorted(found)))
        return tuple(groups)
    if operator == "rasterize":
        nodes, _ = graph(source)
        v = as_record(source)
        w, h = as_index(v.get("width")), as_index(v.get("height"))
        if not 1 <= w <= 64 or not 1 <= h <= 64:
            raise ExecutionError("Raster dimensions unavailable")
        output_grid: list[list[Value]] = [[None] * w for _ in range(h)]
        occupied = set()
        for raw in nodes:
            tick()
            n = as_record(raw)
            x, y = as_index(n.get("x")), as_index(n.get("y"))
            if x >= w or y >= h or (x, y) in occupied:
                raise ExecutionError("Invalid or duplicate spatial role")
            occupied.add((x, y))
            output_grid[y][x] = n.get("value")
        return tuple(tuple(row) for row in output_grid)
    rows = grid(source)
    h, w = len(rows), len(rows[0])
    for row in rows:
        for _ in row:
            tick()
    if operator == "transpose":
        return tuple(tuple(rows[y][x] for y in range(h)) for x in range(w))
    if operator == "reflect_x":
        return tuple(tuple(reversed(row)) for row in rows)
    if operator == "reflect_y":
        return tuple(reversed(rows))
    if operator in {"shift_x", "shift_y"}:
        if not isinstance(other, Fraction) or other.denominator != 1 or abs(other) > 64:
            raise ExecutionError("Spatial translation requires a bounded exact integer")
        dx, dy = (int(other), 0) if operator == "shift_x" else (0, int(other))
        return tuple(
            tuple(
                rows[y - dy][x - dx] if 0 <= x - dx < w and 0 <= y - dy < h else None
                for x in range(w)
            )
            for y in range(h)
        )
    if operator == "raster_graph":
        nodes_out: list[Value] = []
        positions: dict[tuple[int, int], int] = {}
        for y, row in enumerate(rows):
            for x, value in enumerate(row):
                tick()
                if value is not None:
                    positions[x, y] = len(nodes_out)
                    nodes_out.append(record(x=x, y=y, value=value))
        if len(nodes_out) > 128:
            raise ExecutionError("Raster observation has too many occupied participants")
        edges_out = []
        for (x, y), i in positions.items():
            for pos in ((x + 1, y), (x, y + 1)):
                if pos in positions:
                    edges_out.append((i, positions[pos]))
        return record(nodes=tuple(nodes_out), edges=tuple(edges_out), width=w, height=h)
    raise ExecutionError("Unavailable structural algorithm")
