# Structural problems and acquired abstractions: 0.7 API

`runtime.observer.induction` is owned by the existing GroundedObserver. Its executions
use the cornerstone and its learning uses the GroundedTrainer's shared model and optimizer.
It does not receive the procedural environment or a hidden transformation program.

```python
from athanor import record
from athanor.grounded.checkpoint import restore, save

runtime, trainer = restore("artifacts/process-observer.athanor")
inquiry = runtime.observer.induction
concepts = inquiry.form_concepts(observations)   # tuple of native immutable Values
rule = inquiry.infer(examples, trainer, max_calls=2, max_candidates=384, beam=32)
if rule is not None:
    predicted = inquiry.apply(rule, new_input)
    # Supply a later measured consequence to revise alternatives and capability evidence.
    inquiry.apply(rule, new_input, expected=measured_output, measured=True)
    rule = inquiry.rules[-1]                   # current pinned revision
save(runtime, "artifacts/continued.athanor", trainer=trainer)
```

`examples` is a tuple of `(before, after)` Values. Prediction alone does not establish
transfer or reliable generalization. A rule requires at least three distinct consistent
demonstration inputs for adoption. One or two examples retain hypotheses. A supported
rule retains every fitting alternative; a query on which they disagree returns `None`.
A whole unknown outcome (`None`) is not positive transformation evidence.
New measured evidence can remove contradicted alternatives, select a consistent replacement
through the shared scorer, revoke rejected procedures and qualify dependent abstractions.
Old native revisions remain readable. Current inquiries require current owned references.

Native values are exact Fractions, strings, booleans, None, Records, tuples and pinned Refs.
A sequence is an ordered tuple. Collections use sequence operations such as `unique`;
ordering is preserved rather than silently asserting unordered equality. A small graph is
`record(nodes=(record(...), ...), edges=((0, 1), ...))`, with directed ordered index pairs.
A symbolic raster is a rectangular tuple of rows, with `None` denoting absence. Records
can contain other structures. Raster-to-graph conversion computes adjacency from cells;
it supplies neither semantic groups nor the appropriate solution distinction.

`recognize(concept, input)` executes its operational partition. Unavailable properties or
qualified concepts give unknown membership. `combine_concepts(first, second, observations)`
creates a native composition with pinned concept dependencies and separate evidence.
Three repeated copies of one observation cannot establish concept support.

## Objectives and variable binding

```python
goal = (
    record(kind="count", value=2),
    record(kind="equals", value=("second", "first")),
)
plan = inquiry.solve(("first", "first", "second"), goal, trainer,
                     max_depth=3, max_nodes=256, operations=None)
```

Conditions are `equals(value)`, `count(value)`, `ordered(field)`, `contains(value)`,
`concept(reference)` and `count_concept(reference)`, represented by Records with a `kind`
and the indicated argument. Concept goals evaluate the learned concept on the **initial**
structure. Current participants must belong to that eligible group; `count_concept` can
require preserving its cardinality. This avoids changing the meaning of relational roles
when a transformation deletes edges or accepting an empty result by vacuous membership.
Goal predicates bind a native `{initial, current}` frame. For direct validation use
`goal_test(predicate, current, initial=original)`.

`solve` returns a provisional native rule, `None` when the state already satisfies the
goal, or `None` with unknown status when bounded search fails. Inspect the native task
status to distinguish those outcomes. An available operation scope is optional. An
execution is successful only when its actual result satisfies the goal; a successful
single solution does not establish unrestricted applicability. Three distinct successful
measured inputs can support the learned plan within its recorded schema and goal scope.

Participants are bound by native arguments and scoped `item`, `index`, `root`, `peer`,
`peer_index` variables. The interpreter validates expression scope, typed endpoints,
exact index/property types, ordered relationship direction and every nested call. Missing
properties are unknown, not false. No arbitrary Python execution is involved.

## Application and headless commands

The viewer's problem editor and HTTP `/command` accept the same commands as
`Application.command`. `/state` is read-only and includes `induction` observations,
examples, objectives, concepts, candidate scores, alternatives, rules, outcomes,
capability counts and neural updates.

- `form_concepts`: `observations` array.
- `infer_rule`: `examples` array of `{before, after}`; optional `max_calls`,
  `max_candidates`, `beam`.
- `apply_rule`: current `reference` string and `input`; optional measured `expected`.
- `recognize_concept`: current `reference` and `input`.
- `choose_structural_test`: `inputs` array; neural selection precedes oracle feedback.
- `solve_problem`: `input`, `conditions`; optional `max_depth`, `max_nodes`,
  `operations` list of currently available native operation reference strings.

JSON arrays become native tuples. Integers stay exact. Use `{"$number":"1/7"}` for
rational quantities and `{"$ref":"namespace:id@revision"}` for pinned goal references.
Floating-point JSON input is rejected. Structural command results and inspection preserve
rational values with `$number` markers. Acquired reference results are printable strings;
use the explicit `$ref` marker when embedding one in a structural condition.

```powershell
.venv\Scripts\python.exe -m athanor.grounded.app --checkpoint artifacts\concept-observer.athanor --paused --port 8766
.venv\Scripts\python.exe -m athanor.grounded.app --checkpoint artifacts\concept-observer.athanor --headless --paused --steps 0 --problem problem.json
```

The problem file contains one command or 1–64 commands. Use a `save` command to persist
changes from a headless invocation. HTTP request bodies retain the existing 8 KiB limit;
larger bounded inputs can use the Python API or the headless file interface.

## Bounded contracts

Observations/examples: 1–32. Informative test inputs: 1–64. Inference depth: 1–4 calls,
candidate limit: 1–2048, beam: 1–128. Plans: depth 1–4 and 1–2048 evaluated transitions.
Values: at most 8192 visited elements and nesting depth 16. Graphs: 128 nodes and 1024
unique directed edges. Rasters: nonempty rectangles up to 64 by 64; conversion to a graph
also obeys the 128 occupied-node limit. Native execution uses a 20,000-step budget and
24-call depth; per-cell work consumes steps. Failed bounded search is not an absence proof.

Checkpoint format 5 stores concepts, tasks, acquired rules, revoked alternatives, native
history, active inquiry, shared parameters, recurrent state, optimizer and retention anchors.
Formats 1–4 remain readable; restored older models start with an empty structural inquiry.
