"""Identity-free structural features and sparse typed edges for every participant."""

import hashlib
import math
from dataclasses import dataclass, replace
from fractions import Fraction

import torch
from torch import Tensor

from athanor import Athanor, Record, Ref, Role, Structure, ValidationError, Value, record
from athanor.values import references

from .contracts import ModelConfig, Observation
from .index import KnowledgeIndex

TOKEN_BINS = 256
FEATURE_SIZE = len(Role) + 8 + TOKEN_BINS + 1
EDGE_BINS = 32
EDGE_SIZE = EDGE_BINS + 5
_DISPLAY = frozenset({"name", "label", "source", "note", "error_message", "namespace"})


def activation_view(node: Structure) -> Structure:
    """Bound active audit history, while exact native structures stay untouched."""
    if node.fields.get("kind") in {
        "discovery_hypothesis",
        "investigation_evidence",
        "effect_measurement",
        "acquired_relationship",
    }:
        audit = {
            "support",
            "contradictions",
            "comparison",
            "feedback",
            "selected",
            "experiment",
            "evidence",
        }
        return replace(
            node, fields=record({k: v for k, v in node.fields.items() if k not in audit})
        )
    return node


def _add_token(vector: list[float], token: str, start: int, bins: int, scale: float = 1) -> None:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    index = int.from_bytes(digest[:4]) % bins
    vector[start + index] += scale if digest[4] & 1 else -scale


def _number(value: Fraction) -> float:
    if not value:
        return 0.0
    # Avoid conversion overflow for large exact operands; this is an input descriptor.
    log = math.log2(abs(value.numerator)) - math.log2(value.denominator)
    magnitude = math.log1p(2 ** min(log, 50)) if log < 50 else log * math.log(2)
    return math.copysign(math.tanh(magnitude / 8), value.numerator)


def structural_features(node: Structure, *, observation: bool = False) -> list[float]:
    values = [0.0] * FEATURE_SIZE
    values[list(Role).index(node.role)] = 1
    stats = len(Role)
    values[stats] = min(len(node.fields), 32) / 32
    values[stats + 1] = min(len(node.members), 32) / 32
    values[stats + 2] = float(node.category is not None)
    tokens = stats + 8
    count = 0

    def visit(value: Value, path: str) -> None:
        nonlocal count
        count += 1
        _add_token(values, f"{path}:{type(value).__name__}", tokens, TOKEN_BINS)
        if isinstance(value, Ref):
            values[stats + 3] += 1 / 32
        elif isinstance(value, Fraction):
            number = _number(value)
            _add_token(values, f"{path}:number", tokens, TOKEN_BINS, number)
            _add_token(values, f"{path}:zero", tokens, TOKEN_BINS, float(value == 0))
            values[stats + 4] += number / 16
            values[stats + 5] += float(value.denominator != 1) / 16
        elif type(value) is bool:
            _add_token(values, f"{path}:boolean:{value}", tokens, TOKEN_BINS)
        elif isinstance(value, str):
            _add_token(values, f"{path}:text:{value}", tokens, TOKEN_BINS)
        elif isinstance(value, Record):
            for key, item in value.items():
                if key in _DISPLAY and isinstance(item, str) and node.role != Role.RELATION:
                    continue
                visit(item, f"{path}.{key}")
        elif isinstance(value, tuple):
            values[stats + 6] += min(len(value), 16) / 16
            for item in value:
                visit(item, f"{path}[]")

    visit(node.fields, "fields")
    values[stats + 7] = min(count, 128) / 128
    normalizer = math.sqrt(max(count, 1))
    for index in range(tokens, tokens + TOKEN_BINS):
        values[index] /= normalizer
    values[-1] = float(observation)
    return values


def _edges(value: Value, path: str, position: int = 0) -> list[tuple[Ref, str, int]]:
    if isinstance(value, Ref):
        return [(value, path, position)]
    result: list[tuple[Ref, str, int]] = []
    if isinstance(value, Record):
        for key, item in value.items():
            result.extend(_edges(item, f"{path}.{key}", position))
    elif isinstance(value, tuple):
        for index, item in enumerate(value):
            result.extend(_edges(item, f"{path}[]", index))
    return result


def _edge_features(path: str, position: int, inverse: bool) -> list[float]:
    result = [0.0] * EDGE_SIZE
    _add_token(result, path, 0, EDGE_BINS)
    result[EDGE_BINS:] = [
        position / (position + 1),
        1 / (position + 1),
        float(position == 0),
        float(inverse),
        float(not inverse),
    ]
    return result


@dataclass(frozen=True, slots=True)
class Frame:
    refs: tuple[Ref, ...]
    features: Tensor
    edge_index: Tensor
    edge_features: Tensor
    query_index: int
    operation_indices: tuple[int, ...]
    candidate_indices: tuple[int, ...]
    arities: tuple[int, ...]
    meaning_indices: tuple[int, ...] = ()
    intent_features: Tensor | None = None

    @property
    def operations(self) -> tuple[Ref, ...]:
        return tuple(self.refs[index] for index in self.operation_indices)

    @property
    def candidates(self) -> tuple[Ref, ...]:
        return tuple(self.refs[index] for index in self.candidate_indices)

    @property
    def meanings(self) -> tuple[Ref, ...]:
        return tuple(self.refs[index] for index in self.meaning_indices)


class GraphBuilder:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.index = KnowledgeIndex()

    def build(
        self,
        knowledge: Athanor,
        observation: Observation,
        *,
        self_refs: tuple[Ref, ...] = (),
        observer: Ref | None = None,
        operations: tuple[Ref, ...] | None = None,
    ) -> Frame:
        self.index.sync(knowledge)
        operations = tuple(self.index.operations) if operations is None else operations
        if len(set(operations)) != len(operations) or any(
            knowledge.get(ref).role != Role.OPERATION for ref in operations
        ):
            raise ValidationError("Active operations must be distinct installed procedures")
        seeds = [observation.ref, *observation.candidates, *operations, *self_refs]
        if observer is not None:
            seeds.append(observer)
        selected: dict[Ref, Structure] = {}

        def closure(pending: list[Ref]) -> None:
            while pending:
                ref = pending.pop()
                if ref in selected:
                    continue
                node = activation_view(knowledge.get(ref))
                selected[ref] = node
                if len(selected) > self.config.max_nodes:
                    raise ValidationError("Active neural graph exceeds its explicit node budget")
                pending.extend(references(node.fields))
                pending.extend(node.members)
                if node.category is not None:
                    pending.append(node.category)

        closure(list(reversed(seeds)))
        # Include incident claims as structures; do not resolve/filter applicability.
        # Following their references can reveal other intentions/relationships.
        changed = True
        while changed:
            additions = [
                ref for ref in self.index.related_claims(tuple(selected)) if ref not in selected
            ]
            changed = bool(additions)
            closure(list(reversed(additions)))
        refs = tuple(selected)
        positions = {ref: index for index, ref in enumerate(refs)}
        edge_pairs: list[tuple[int, int]] = []
        descriptors: list[list[float]] = []
        for ref, node in selected.items():
            links = _edges(node.fields, "fields")
            links.extend((member, "members", index) for index, member in enumerate(node.members))
            if node.category is not None:
                links.append((node.category, "category", 0))
            for target, path, position in links:
                source_index, target_index = positions[ref], positions[target]
                edge_pairs.extend(((source_index, target_index), (target_index, source_index)))
                descriptors.extend(
                    (_edge_features(path, position, False), _edge_features(path, position, True))
                )
        arities = []
        for ref in operations:
            procedure = selected[ref].fields["procedure"]
            if not isinstance(procedure, Record) or not isinstance(procedure["inputs"], tuple):
                raise ValidationError("Installed procedure interface is malformed")
            arities.append(len(procedure["inputs"]))
        return Frame(
            refs,
            torch.tensor(
                [
                    structural_features(node, observation=ref == observation.ref)
                    for ref, node in selected.items()
                ],
                dtype=torch.float32,
            ),
            torch.tensor(edge_pairs, dtype=torch.long).reshape(-1, 2).T.contiguous(),
            torch.tensor(descriptors, dtype=torch.float32).reshape(-1, EDGE_SIZE),
            positions[observation.ref],
            tuple(positions[ref] for ref in operations),
            tuple(positions[ref] for ref in observation.candidates),
            tuple(arities),
            tuple(
                index
                for index, node in enumerate(selected.values())
                if node.role == Role.DATA and node.fields.get("kind") == "concept"
            ),
        )
