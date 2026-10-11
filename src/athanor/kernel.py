"""The small public foundation: structures, teaching, binding, and exact execution."""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from threading import RLock
from uuid import uuid4

from .contracts import (
    Assessment,
    Difference,
    Endpoint,
    Evidence,
    FieldSpec,
    Inspection,
    Limits,
    Receipt,
    ResolvedClaim,
    Role,
    Structure,
    compare,
    fields_record,
)
from .errors import ExecutionError, ResourceLimitError, ValidationError
from .programs import Evaluator, Procedure
from .store import Store
from .values import Record, Ref, as_index, as_record, as_ref, as_tuple, freeze, record, references


class Athanor:
    """One independent local knowledge store, serialized mutations, no external model."""

    def __init__(self, *, namespace: str | None = None) -> None:
        chosen = uuid4().hex if namespace is None else namespace
        if type(chosen) is not str or not chosen:
            raise ValidationError("A store namespace must be a nonempty string")
        freeze(chosen)
        self._store = Store(namespace=chosen)
        self._lock = RLock()
        self._transaction_active = False

    @property
    def namespace(self) -> str:
        return self._store.namespace

    def _commit(self, action: Callable[[Store], Ref]) -> Ref:
        with self._lock:
            if self._transaction_active:
                raise ValidationError("Use the staged kernel while a transaction is active")
            staged = self._store.stage()
            result = action(staged)
            self._store = staged
            return result

    def transaction[T](self, work: Callable[[Athanor], T]) -> T:
        """Commit all staged kernel calls together; external side effects are not transactional."""
        with self._lock:
            if self._transaction_active:
                raise ValidationError("Use the staged kernel for nested transactions")
            staged = Athanor(namespace=self.namespace)
            staged._store = self._store.stage()
            self._transaction_active = True
            try:
                result = work(staged)
                self._store = staged._store
                return result
            finally:
                self._transaction_active = False

    def get(self, ref: Ref | str) -> Structure:
        """Refs resolve exact snapshots; an explicit ID string requests latest state."""
        with self._lock:
            if isinstance(ref, str):
                if ref not in self._store.latest:
                    raise ValidationError(f"Unknown local identity: {ref}")
                ref = self._store.latest[ref]
            return self._store.get(ref)

    @property
    def structures(self) -> tuple[Structure, ...]:
        """All revisions, in production order; immutable external view."""
        with self._lock:
            return tuple(self._store.get(receipt.result) for receipt in self._store.receipts)

    @property
    def receipts(self) -> tuple[Receipt, ...]:
        with self._lock:
            return tuple(self._store.receipts)

    def create(
        self,
        category: Ref | None = None,
        fields: dict[str, object] | Record | None = None,
        *,
        members: Sequence[Ref] = (),
        source: str = "structured input",
    ) -> Ref:
        values = fields_record(fields)
        refs = tuple(members)
        return self._commit(
            lambda store: store.publish(
                Role.DATA,
                category,
                values,
                refs,
                "create",
                inputs=refs,
                details=self._source(source),
            )
        )

    def compose(
        self,
        members: Sequence[Ref],
        category: Ref | None = None,
        fields: dict[str, object] | Record | None = None,
        *,
        source: str = "structured composition",
    ) -> Ref:
        """Composition is ordinary data construction with an explicit production action."""
        refs, values = tuple(members), fields_record(fields)
        return self._commit(
            lambda store: store.publish(
                Role.DATA,
                category,
                values,
                refs,
                "compose",
                inputs=refs,
                details=self._source(source),
            )
        )

    @staticmethod
    def _source(source: str) -> Record:
        if type(source) is not str or not source:
            raise ValidationError("Creation/teaching requires a nonempty attributed source")
        return record(source=source)

    def teach_category(
        self,
        name: str,
        fields: Mapping[str, FieldSpec] | None = None,
        *,
        parents: Sequence[Ref] = (),
        source: str = "structured teacher",
    ) -> Ref:
        schema = record({key: spec.to_record() for key, spec in (fields or {}).items()})
        values = record(name=name, schema=schema, parents=tuple(parents))
        return self._commit(
            lambda store: store.publish(
                Role.CATEGORY,
                None,
                values,
                (),
                "teach_category",
                inputs=tuple(parents),
                details=self._source(source),
            )
        )

    def teach_relation(
        self,
        name: str,
        source_interface: Endpoint | None = None,
        target_interface: Endpoint | None = None,
        *,
        inherit_source: bool = False,
        source: str = "structured teacher",
    ) -> Ref:
        values = record(
            name=name,
            source=(Endpoint() if source_interface is None else source_interface).to_record(),
            target=(Endpoint() if target_interface is None else target_interface).to_record(),
            inherit_source=inherit_source,
        )
        return self._commit(
            lambda store: store.publish(
                Role.RELATION, None, values, (), "teach_relation", details=self._source(source)
            )
        )

    def teach_operation(
        self, name: str, procedure: Procedure, *, source: str = "structured teacher"
    ) -> Ref:
        values = record(name=name, procedure=procedure.to_record())
        return self._commit(
            lambda store: store.publish(
                Role.OPERATION, None, values, (), "teach_operation", details=self._source(source)
            )
        )

    def bind(self, relation: Ref, source: Ref, target: Ref, evidence: Evidence) -> Ref:
        values = record(
            relation=relation, source=source, target=target, evidence=evidence.to_record()
        )
        return self._commit(
            lambda store: store.publish(
                Role.CLAIM, None, values, (), "bind", inputs=(relation, source, target)
            )
        )

    def distinguish(self, left: Ref, right: Ref) -> Difference:
        with self._lock:
            return compare(self._store.get(left), self._store.get(right))

    def ancestors(self, category: Ref) -> tuple[Ref, ...]:
        with self._lock:
            return self._store.ancestors(category)

    def schema(self, category: Ref) -> Record:
        with self._lock:
            return self._store.schema((category,))

    def relations(self, subject: Ref, relation: Ref) -> tuple[ResolvedClaim, ...]:
        """Resolve exact direct claims and explicitly declared source inheritance."""
        with self._lock:
            store = self._store
            subject_node = store.get(subject)
            definition = store.require_role(relation, Role.RELATION)
            category = subject if subject_node.role == Role.CATEGORY else subject_node.category
            ancestry = store.ancestors(category) if category is not None else ()
            found: list[ResolvedClaim] = []
            for node in store.nodes.values():
                if node.role != Role.CLAIM or node.fields["relation"] != relation:
                    continue
                origin = as_ref(node.fields["source"])
                if origin == subject:
                    found.append(ResolvedClaim(node.ref, origin, ()))
                elif definition.fields["inherit_source"] and origin in ancestry:
                    assert category is not None
                    via = store.inheritance_path(category, origin)
                    found.append(ResolvedClaim(node.ref, origin, via))
            return tuple(found)

    def assess(self, subject: Ref, relation: Ref, target: Ref) -> Assessment:
        """Report installed support; never equate it with external truth."""
        with self._lock:
            self._store.get(target)
            stances = {
                as_record(claim.fields["evidence"])["stance"]
                for resolved in self.relations(subject, relation)
                if (claim := self._store.get(resolved.claim)).fields["target"] == target
            }
            positive, negative = "positive" in stances, "negative" in stances
            if positive and negative:
                return Assessment.CONFLICT
            if positive:
                return Assessment.SUPPORTED
            if negative:
                return Assessment.REFUTED
            return Assessment.UNKNOWN

    def execute(
        self, operation: Ref, arguments: Sequence[Ref] = (), *, limits: Limits | None = None
    ) -> Ref:
        """Execute a taught procedure; all nested results share one atomic transaction."""
        budget = Limits() if limits is None else limits
        if not isinstance(budget, Limits):
            raise ValidationError("Execution requires explicit Limits")
        args = tuple(arguments)

        def run(store: Store) -> Ref:
            evaluator = Evaluator(store.get, budget)

            def invoke(target: Ref, inputs: tuple[Ref, ...], depth: int = 1) -> Ref:
                evaluator.tick()
                if depth > budget.max_depth:
                    raise ResourceLimitError("Invocation exceeded its call-depth budget")
                definition = store.require_role(target, Role.OPERATION)
                spec = as_record(definition.fields["procedure"])
                interface = as_tuple(spec["inputs"])
                if len(inputs) != len(interface):
                    raise ValidationError("Procedure arity mismatch")
                for ref, raw in zip(inputs, interface, strict=True):
                    if not store.matches(ref, as_record(raw)):
                        raise ValidationError(f"Input {ref} violates the procedure interface")
                update_index = spec["update"]
                update = None if update_index is None else inputs[as_index(update_index)]
                if update is not None:
                    old = store.get(update)
                    result_category = spec["result"]
                    if old.role != Role.DATA or (
                        result_category is not None
                        and (
                            old.category is None
                            or as_ref(result_category) not in store.ancestors(old.category)
                        )
                    ):
                        raise ValidationError("Update target violates its result category contract")
                    if store.latest[update.id] != update:
                        raise ValidationError("Cannot update a stale revision")
                causes: list[int] = []

                def nested(callee: Ref, nested_inputs: tuple[Ref, ...]) -> Ref:
                    result = invoke(callee, nested_inputs, depth + 1)
                    causes.append(store.producers[result])
                    return result

                fields = record(
                    {
                        name: evaluator.evaluate(as_record(expr), inputs, nested, target)
                        for name, expr in as_record(spec["fields"]).items()
                    }
                )
                if update is not None:
                    # Update fields are a patch; required prior fields remain present.
                    fields = record(dict(store.get(update).fields) | dict(fields))
                raw_members = spec["members"]
                if raw_members is None:
                    members = () if update is None else store.get(update).members
                else:
                    members = tuple(
                        as_ref(raw)
                        for raw in as_tuple(
                            evaluator.evaluate(as_record(raw_members), inputs, nested, target)
                        )
                    )
                category = None if spec["result"] is None else as_ref(spec["result"])
                if update is not None:
                    category = store.get(update).category
                return store.publish(
                    Role.DATA,
                    category,
                    fields,
                    members,
                    "transform" if update is not None else "execute",
                    update=update,
                    inputs=inputs,
                    operation=target,
                    causes=tuple(causes),
                    details=record(expression_steps=evaluator.steps),
                )

            try:
                return invoke(operation, args)
            except RecursionError as exc:
                raise ResourceLimitError("Python evaluator stack limit exceeded") from exc
            except (KeyError, IndexError, ZeroDivisionError) as exc:
                raise ExecutionError(f"Expression could not execute: {exc}") from exc

        return self._commit(run)

    def inspect(self, ref: Ref) -> Inspection:
        with self._lock:
            store = self._store
            node = store.get(ref)
            return Inspection(node, store.receipts[store.producers[ref] - 1])

    def provenance(self, ref: Ref) -> tuple[Receipt, ...]:
        """Dependency closure in production order, including definitions and members."""
        with self._lock:
            store = self._store
            selected: set[int] = set()
            pending = [ref]
            while pending:
                current = pending.pop()
                node = store.get(current)
                sequence = store.producers[current]
                if sequence in selected:
                    continue
                selected.add(sequence)
                receipt = store.receipts[sequence - 1]
                pending.extend(receipt.inputs)
                pending.extend(node.members)
                pending.extend(references(node.fields))
                if node.category is not None:
                    pending.append(node.category)
                if receipt.operation is not None:
                    pending.append(receipt.operation)
                pending.extend(store.receipts[cause - 1].result for cause in receipt.causes)
            return tuple(store.receipts[index - 1] for index in sorted(selected))

    def dumps(self) -> str:
        from .persistence import dumps

        with self._lock:
            return dumps(self._store)

    @classmethod
    def loads(cls, snapshot: str) -> Athanor:
        from .persistence import loads

        store = loads(snapshot)
        result = cls(namespace=store.namespace)
        result._store = store
        return result

    def save(self, path: str | Path) -> None:
        from .persistence import save

        save(Path(path), self.dumps())

    @classmethod
    def restore(cls, path: str | Path) -> Athanor:
        from .persistence import read

        return cls.loads(read(Path(path)))
