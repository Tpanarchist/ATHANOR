"""Incremental native activation indexes. Exact pinned references remain the handles."""

from collections import defaultdict

from athanor import Athanor, Ref, Role
from athanor.values import references


class KnowledgeIndex:
    def __init__(self) -> None:
        self.knowledge: Athanor | None = None
        self.offset = 0
        self.operations: list[Ref] = []
        self.roles: dict[Role, list[Ref]] = defaultdict(list)
        self.kinds: dict[str, list[Ref]] = defaultdict(list)
        self.incident: dict[Ref, list[Ref]] = defaultdict(list)
        self.categories: dict[Ref, list[Ref]] = defaultdict(list)
        self.indexed = 0

    def sync(self, knowledge: Athanor) -> None:
        if self.knowledge is not knowledge or self.offset > knowledge.structure_count:
            KnowledgeIndex.__init__(self)
            self.knowledge = knowledge
        additions = knowledge.structures_since(self.offset)
        for node in additions:
            self.roles[node.role].append(node.ref)
            if node.role == Role.OPERATION:
                self.operations.append(node.ref)
            kind = node.fields.get("kind")
            if isinstance(kind, str):
                self.kinds[kind].append(node.ref)
            if node.category is not None:
                self.categories[node.category].append(node.ref)
            if node.role == Role.CLAIM:
                for endpoint in dict.fromkeys(references(node.fields)):
                    self.incident[endpoint].append(node.ref)
        self.offset += len(additions)
        self.indexed += len(additions)

    def related_claims(self, refs: tuple[Ref, ...]) -> tuple[Ref, ...]:
        # Publication order preserves the original builder's traversal semantics.
        return tuple(
            sorted(
                {claim for ref in refs for claim in self.incident.get(ref, ())},
                key=lambda ref: (ref.id, ref.revision),
            )
        )
