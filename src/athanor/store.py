"""Internal version store and contract validation shared by mutation and restore."""

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

from .contracts import (
    Receipt,
    Role,
    Structure,
    validate_evidence,
    validate_field_spec,
    validate_fields,
)
from .errors import ValidationError
from .programs import validate_expression
from .values import (
    Record,
    Ref,
    as_index,
    as_record,
    as_ref,
    as_string,
    as_tuple,
    exact_equal,
    record,
    references,
)


@dataclass(slots=True)
class Store:
    namespace: str = "local"
    nodes: dict[Ref, Structure] = field(default_factory=dict)
    latest: dict[str, Ref] = field(default_factory=dict)
    receipts: list[Receipt] = field(default_factory=list)
    producers: dict[Ref, int] = field(default_factory=dict)

    def stage(self) -> Store:
        return Store(
            self.namespace,
            self.nodes.copy(),
            self.latest.copy(),
            self.receipts.copy(),
            self.producers.copy(),
        )

    def get(self, ref: Ref) -> Structure:
        if not isinstance(ref, Ref) or ref.namespace != self.namespace or ref not in self.nodes:
            raise ValidationError(f"Unresolved local reference: {ref}")
        return self.nodes[ref]

    def require_role(self, ref: Ref, role: Role) -> Structure:
        result = self.get(ref)
        if result.role != role:
            raise ValidationError(f"{ref} is not a {role.value}")
        return result

    def ancestors(self, category: Ref) -> tuple[Ref, ...]:
        """Deterministic full traversal, not a first-parent shortcut."""
        ordered: list[Ref] = []
        visited: set[Ref] = set()
        pending = [category]
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            node = self.require_role(current, Role.CATEGORY)
            ordered.append(current)
            parents = tuple(as_ref(raw) for raw in as_tuple(node.fields["parents"]))
            pending.extend(reversed(parents))
        return tuple(ordered)

    def schema(self, categories: Iterable[Ref], own: Record | None = None) -> Record:
        merged = {} if own is None else dict(own)
        for category in categories:
            for ancestor in self.ancestors(category):
                fields = as_record(self.get(ancestor).fields["schema"])
                for key, spec in fields.items():
                    if key in merged and not exact_equal(merged[key], spec):
                        raise ValidationError(f"Incompatible inherited field contract: {key}")
                    merged[key] = spec
        return record(merged)

    def inheritance_path(self, start: Ref, target: Ref) -> tuple[Ref, ...]:
        pending: deque[tuple[Ref, ...]] = deque([(start,)])
        visited: set[Ref] = set()
        while pending:
            path = pending.popleft()
            current = path[-1]
            if current == target:
                return path
            if current in visited:
                continue
            visited.add(current)
            node = self.require_role(current, Role.CATEGORY)
            for parent in as_tuple(node.fields["parents"]):
                pending.append((*path, as_ref(parent)))
        return ()

    def endpoint(self, raw: Record) -> None:
        if set(raw) != {"role", "category"}:
            raise ValidationError("Malformed endpoint interface")
        role = raw["role"]
        if role is not None and role not in {item.value for item in Role}:
            raise ValidationError("Unknown endpoint role")
        category = raw["category"]
        if category is not None:
            self.require_role(as_ref(category), Role.CATEGORY)
            if role not in {None, Role.DATA.value, Role.CATEGORY.value}:
                raise ValidationError("Category constraints apply to data or category endpoints")

    def matches(self, ref: Ref, endpoint: Record) -> bool:
        node = self.get(ref)
        role, category = endpoint["role"], endpoint["category"]
        if role is not None and node.role.value != role:
            return False
        if category is None:
            return True
        actual = node.ref if node.role == Role.CATEGORY else node.category
        return actual is not None and as_ref(category) in self.ancestors(actual)

    def validate(self, node: Structure) -> None:
        """Validate a new node against already published/staged prior revisions."""
        for ref in references(node.fields):
            self.get(ref)
        for ref in node.members:
            self.get(ref)
        if node.category is not None:
            self.require_role(node.category, Role.CATEGORY)
        fields = node.fields
        if node.role != Role.DATA and (node.category is not None or node.members):
            raise ValidationError(
                "Definitions and claims cannot declare a data category or members"
            )
        if node.role == Role.DATA:
            if node.category is not None:
                validate_fields(fields, self.schema((node.category,)))
        elif node.role == Role.CATEGORY:
            if set(fields) != {"name", "schema", "parents"}:
                raise ValidationError("Malformed category definition")
            as_string(fields["name"], "category name")
            schema = as_record(fields["schema"], "category schema")
            for spec in schema.values():
                validate_field_spec(as_record(spec))
            parents = tuple(as_ref(raw) for raw in as_tuple(fields["parents"]))
            if len(set(parents)) != len(parents):
                raise ValidationError("Duplicate category parents")
            self.schema(parents, schema)
        elif node.role == Role.RELATION:
            if set(fields) != {"name", "source", "target", "inherit_source"}:
                raise ValidationError("Malformed relationship definition")
            as_string(fields["name"], "relationship name")
            source, target = as_record(fields["source"]), as_record(fields["target"])
            self.endpoint(source)
            self.endpoint(target)
            if type(fields["inherit_source"]) is not bool:
                raise ValidationError("Inheritance flag must be boolean")
            if fields["inherit_source"] and source["role"] != Role.CATEGORY.value:
                raise ValidationError("Inherited relationship sources must be categories")
        elif node.role == Role.CLAIM:
            if set(fields) != {"relation", "source", "target", "evidence"}:
                raise ValidationError("Malformed claim")
            relation = self.require_role(as_ref(fields["relation"]), Role.RELATION)
            claim_source, claim_target = as_ref(fields["source"]), as_ref(fields["target"])
            if not self.matches(claim_source, as_record(relation.fields["source"])):
                raise ValidationError("Relationship source violates its interface")
            if not self.matches(claim_target, as_record(relation.fields["target"])):
                raise ValidationError("Relationship target violates its interface")
            validate_evidence(as_record(fields["evidence"]))
        elif node.role == Role.OPERATION:
            if set(fields) != {"name", "procedure"}:
                raise ValidationError("Malformed operation definition")
            as_string(fields["name"], "operation name")
            self.procedure(as_record(fields["procedure"]))
        else:
            raise ValidationError("Unknown structure role")

    def procedure(self, spec: Record) -> None:
        if set(spec) != {"inputs", "result", "fields", "members", "update"}:
            raise ValidationError("Malformed procedure")
        inputs = as_tuple(spec["inputs"], "procedure inputs")
        for raw in inputs:
            self.endpoint(as_record(raw))
        category = spec["result"]
        if category is not None:
            self.require_role(as_ref(category), Role.CATEGORY)
        update = spec["update"]
        if update is not None and as_index(update) >= len(inputs):
            raise ValidationError("Update target is outside input interface")

        def check_operation(ref: Ref) -> None:
            self.require_role(ref, Role.OPERATION)

        for expression in as_record(spec["fields"], "output fields").values():
            validate_expression(as_record(expression), len(inputs), check_operation)
        if spec["members"] is not None:
            validate_expression(as_record(spec["members"]), len(inputs), check_operation)

    def publish(
        self,
        role: Role,
        category: Ref | None,
        fields: Record,
        members: tuple[Ref, ...],
        action: str,
        *,
        update: Ref | None = None,
        inputs: tuple[Ref, ...] = (),
        operation: Ref | None = None,
        causes: tuple[int, ...] = (),
        details: Record | None = None,
    ) -> Ref:
        if update is None:
            if len(self.latest) >= 99_999_999:
                raise ValidationError("Local identity namespace exhausted")
            ref = Ref(f"s{len(self.latest) + 1:08d}", namespace=self.namespace)
        else:
            old = self.get(update)
            if self.latest[update.id] != update:
                raise ValidationError("Cannot revise a stale input; use its current revision")
            if old.role != Role.DATA or role != Role.DATA or old.category != category:
                raise ValidationError("Only data state may change; identity/category are preserved")
            ref = Ref(update.id, update.revision + 1, self.namespace)
        node = Structure(ref, role, category, fields, members)
        self.validate(node)
        for dependency in inputs:
            self.get(dependency)
        if operation is not None:
            self.require_role(operation, Role.OPERATION)
        if any(type(cause) is not int or not 1 <= cause <= len(self.receipts) for cause in causes):
            raise ValidationError("Invalid receipt dependency")
        receipt = Receipt(
            len(self.receipts) + 1,
            action,
            ref,
            inputs,
            operation,
            causes,
            Record() if details is None else details,
        )
        self.nodes[ref] = node
        self.latest[ref.id] = ref
        self.producers[ref] = receipt.sequence
        self.receipts.append(receipt)
        return ref
