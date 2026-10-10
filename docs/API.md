# ATHANOR Goal 001 — public contracts

## Knowledge store

`Athanor(namespace=None)` creates an independent store. The default namespace is a
fresh UUID allocated once at initialization. Specify a namespace to reproduce exact
IDs, receipts, and snapshots from the same action sequence. Reusing a namespace
deliberately asserts a common lineage. Restoring preserves it; branch merging is
not implemented. References from a different namespace are rejected.

`Ref(id, revision=1, namespace="local")` identifies an exact local revision. Usually
use references returned by the kernel rather than constructing them. `get(ref)`
returns that immutable snapshot; `get(ref.id)` explicitly requests the latest state.

| Method | Behavior |
| --- | --- |
| `create(category=None, fields=None, members=(), source=...)` | Validates and creates data; an uncategorized structure has unconstrained field names. |
| `compose(members, category=None, fields=None, source=...)` | Creates an ordered composite using the ordinary data contract. |
| `teach_category(name, fields=None, parents=(), source=...)` | Installs immutable field/ancestry constraints. |
| `teach_relation(name, source_interface=None, target_interface=None, inherit_source=False, source=...)` | Installs typed endpoint and applicability rules. |
| `teach_operation(name, procedure, source=...)` | Validates and installs an executable procedure structure. |
| `bind(relation, source, target, evidence)` | Installs an attributed exact-endpoint claim. |
| `execute(operation, arguments=(), limits=None)` | Runs a procedure with atomic nested publication. |
| `distinguish(left, right)` | Separately reports identity/revision/state/fields/members/category/role differences. |
| `ancestors(category)`, `schema(category)` | Resolves full ancestry and compatible inherited field contracts. |
| `relations(subject, relation)` | Returns claim references with explicit inheritance paths; direct claims apply to exact revisions. |
| `assess(subject, relation, target)` | Reports supported/refuted/conflict/unknown installed evidence. |
| `inspect(ref)`, `provenance(ref)` | Returns the production receipt or dependency closure in production order. |
| `dumps()`, `loads(text)`, `save(path)`, `restore(path)` | Canonical versioned snapshots with strict restoration. |

Public `structures` and `receipts` properties return immutable views. Execution
contexts, evaluators, and locks are transient and are never serialized.

## Values and interfaces

Values are null, boolean, string, exact rational number, reference, ordered sequence,
or immutable `Record`. Input integers normalize to `Fraction`; floats and arbitrary
Python objects are rejected. Equality distinguishes boolean true from numeric one.
Records sort keys, preserve sequence order, and copy mutable input. Strings must be
valid Unicode code points so every admitted string can be serialized as UTF-8.
Recursive field records are supported; value nesting is bounded to detect
cyclic/excessive input. Structural nesting uses references and is independent of
this value-encoding bound.

`FieldSpec(kind="any", required=True, minimum=None, maximum=None)` accepts kinds
`any`, `number`, `string`, `bool`, `ref`, `sequence`, `record`. Bounds are exact,
inclusive, and require numeric fields. Categorized data uses a closed inherited
schema. There are no implicit defaults or conversions from textual numbers.

`Endpoint(role=None, category=None)` admits structures by role and/or category.
Category constraints address data categories or category ancestry. An operation
with `Endpoint(Role.DATA, parent)` accepts specializations. Relationship claims
are descriptive evidence and do not grant execution permission.

`Evidence(source, stance="positive", confidence=None, note="")` requires an
attributed source and stance positive/negative/uncertain. Confidence, when supplied,
is an exact number in [0, 1] and has no automatic probabilistic interpretation.
Uncertain evidence alone produces unknown assessment. Positive plus negative is
conflict; insertion order and confidence do not override disagreement.

## Procedure teaching

`Procedure(inputs, result, fields, members=None, update=None)` declares input
endpoints, a result category, expressions for result fields, an optional expression
for members, and an optional input index to update. New results use the declared
category. Updates preserve the input's identity and actual category, which must
satisfy the declared result category when present. Update fields patch existing
state; the entire final state is revalidated, including specialized constraints.
Without a member expression, new results have no members and updates retain members.
Updating a stale revision or a definition/claim is rejected.

Output fields evaluate in sorted name order. Each expression executes when encountered;
there is no implicit memoization or common-subexpression elimination. Nested calls
produce persistent results and receipts in the same transaction. Their returned
exact references may be read or composed. If a nested update makes an outer update's
input stale, the whole invocation fails. Definitions cannot be modified by a procedure;
teach a new definition under a new ID and explicitly adopt it.

## Expression builders (`athanor.programs`)

| Builder | Semantics |
| --- | --- |
| `literal(value)` | An immutable constant, including an exact reference. |
| `argument(index)` | Exact structure reference supplied at that input position. |
| `field(of, name)`, `has_field(of, name)` | Read/test a record field or a referenced structure's fields. |
| `members(of)` | Ordered exact references of a structure. |
| `item(of, index)` | Select a sequence item by nonnegative index. |
| `binary(op, left, right)` | add/sub/mul/div on rationals; lt/le on rationals; type-aware eq on values. |
| `choose(condition, then, otherwise)` | Exact boolean branch; only the selected expression executes. |
| `sequence(*expressions)` | Evaluate an ordered tuple. |
| `fold(items, initial, body)` | Iterate a sequence, evaluating body with the current item and accumulator. |
| `variable("item")`, `variable("acc")` | Fold-local variables; nested folds shadow them in their bodies. |
| `call(operation_ref, *arguments)` | Invoke an installed procedure on evaluated reference arguments. |
| `call("self", *arguments)` | Bounded recursive invocation of the containing procedure. |

Builders return immutable structural records. Structured input can provide the same
grammar directly; unknown keys/opcodes, invalid variable scopes, nonexistent called
operations, invalid input indices, and excessive AST depth are rejected at teaching.
Operand types, branch conditions, field availability, call arity, output constraints,
and current update revisions are checked during execution. Static termination/type
inference is not implemented. `Limits(max_steps=100_000, max_depth=64)` bounds one
entire invocation, including every nested call and fold iteration. These are execution
limits, not parameter/model-size limits or a hostile-input time/memory sandbox.

## Persistence and provenance limits

Snapshots include the namespace, every revision, definitions, claims, exact numbers,
and production receipts. Tagged JSON prevents user fields from impersonating serialized
reference/number tags. The digest covers canonical payload bytes. Restoration rejects
unknown versions, duplicate JSON/record keys, noncanonical rationals, unresolved or
foreign references, incompatible interfaces, invalid revision sequences, and invalid
causal receipt dependencies. It validates chronological structural contracts rather
than rerunning each historical computation. A rehashed fabricated snapshot can be
well formed; authenticate sources separately when that becomes necessary.

Save writes and flushes a temporary file in the destination directory, then replaces
the destination atomically. The parent must already exist. Multi-process writers,
directory fsync/power-loss guarantees, migrations, merge, and garbage collection are
outside this version. File I/O failure raises `PersistenceError`; invalid structural
input raises `ValidationError`; unavailable evaluation raises `ExecutionError`; budget
exhaustion raises `ResourceLimitError`. No rejected invocation changes the store.
## Optional neural observer

Goal 002 adds `athanor.neural` without changing the APIs above. See
[NEURAL_API.md](NEURAL_API.md) for observations, learned proposals, feedback,
dynamic self-map, supervised training, and combined checkpoint continuation.
