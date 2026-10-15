# Representation invention API — 0.8.0

`Runtime.observer.representations` is a front end of the existing induction, process,
grounded and recurrent machinery. Importing the native cornerstone still needs no Torch.
Representations execute only validated native Procedures with the common structural
value category; applications use exact canonical Values and revision-pinned Refs.

## Observations and execution

`observe(value, source=..., capture=None, sufficient=True)` returns an attributed native
observation binding. Repeated immutable supplied values can reuse that binding; this does
not identify world entities. Camera captures keep separate frame provenance. Pass an
observation Ref explicitly when reusing an observation. `camera(frame, eye=0,
measurements=False)` uses the existing learned foreground/appearance encoder, records
raw RGB and features, and supplies an ungrouped foreground raster. With `measurements=True`
it supplies a composite containing that raster and pixel feature/position measurements.
Record projection is then a candidate native process, rather than a supplied solution.
No World, scene graph, object identity, labels or hidden generator procedure is accepted.

`interpret(representation, value_or_observation_ref)` returns an interpretation receipt,
or None for unsupported, insufficient or incompatible input. The receipt binds observation,
representation revision and actual typed execution result. `raw(receipt)` reads that
result. Immutable observation/representation pairs can reuse a result; the reuse is
recorded explicitly. New evidence revises current participants and leaves historical pins.

## Invention, concepts and evidence

`invent(pairs, trainer, max_depth=6, max_candidates=96, beam=8)` receives tuples of
`(left, right, equivalent)` with a measured boolean relationship outcome. Inputs may be
canonical values or owned observation/interpretation Refs. Candidate representations are
built from observed field/type projections, relational predicates, induced groups,
numerical reductions, derived properties, stable ordering and acquired fragments.
No output representation or correct grouping is supplied.

Limits: 1–24 pairs, 1–6 constituent calls, 1–256 constructed candidates, beam 1–32.
Native observations retain the existing bounded shape/depth rules. Each representation
execution has a 250,000 expression-step and 24 call-depth budget. Exact algorithms use
the same cornerstone evaluator. Repeated operators, unchanged sampled results and
incompatible structures are pruned before expanding full candidates; pruning is bounded
search policy, not an impossibility proof.

The shared scorer ranks proposals and controls beam expansion. Measured partial
discrepancies guide refinement, but only correct final relationship predictions support
adoption. Support needs at least three distinct observed pairs, positive and negative
outcomes, sufficient observations, and no inconsistent prediction. Positive-only or
duplicate evidence remains hypothesized. The selected Ref may be None even when another
candidate fits: a neural proposal is subject to the evidence veto.

`relationship(left, right)` evaluates all surviving interpretations from the current
inquiry. Disagreement or insufficient observations returns None. `choose_test(pairs)`
uses predicted disagreement to admit informative options, then lets the shared recurrent
scorer select one. It returns two observation Refs or None when no informative available
test is established. Its proposal is recorded in the existing inquiry history.
`revise(left, right, equivalent)` incorporates a newly measured outcome, qualifies failed
interpretations and their dependent knowledge, and reselects from surviving alternatives.
Observation, prediction and external correspondence outcome remain distinct records.

`form_concept(representation, observations)` creates an unnamed native membership
procedure comparing the executable derived structure with a pinned observed prototype.
`recognize(concept, value_or_ref, expected=None)` returns boolean or unknown and records
actual outcomes; measured `expected` outcomes update transfer evidence. It does not assert
world identity or unrestricted category meaning. `compose(first, second, observations)`
creates a higher-level executable interpretation when output/input contracts match.
Successful execution alone gives provisional applicability. Dependencies remain reopenable.

Repeated suffixes become native fragments. Contracts retain fields actually used by their
computation; generic graph/collection algorithms can accept unfamiliar node properties.
Reuse in another acquired process supplies evidence for a fragment's utility. Shared
references reduce repeated procedure descriptions. Cached pinned results reduce actual
execution; shorter descriptions alone do not establish faster or more truthful computation.

## Goals

`solve(value_or_ref, conditions, trainer, max_nodes=64, max_depth=3, operations=None)`
uses the existing Induction goal compiler and planner. A bounded initial trial uses
available native operations. If it fails, recurrent selection prioritizes compatible
acquired interpretations for further native trials. Initial and revised tasks, available
operations, outcomes and budgets are retained. All trials share the declared node budget;
at most four interpretations are tried. Depth is 1–3 before interpretation, up to 4 after
composition; node budget is 1–2048. Failed search returns None/unknown. An already satisfied
goal also returns None, with `status == "goal_supported"`; inspect status to distinguish it.
Native solution rules retain the existing provisional-to-supported evidence requirements.

## Application commands

The viewer's problem editor, HTTP `/command`, and headless `--problem` file use the same
Application. Arrays become tuples; `{"$number":"1/7"}` preserves an exact rational;
`{"$ref":"namespace:id@revision"}` passes a pinned observation or interpretation Ref.

| Command | Fields |
| --- | --- |
| `invent_representation` | `examples: [{left, right, equivalent}]`, optional bounds |
| `observe_camera` | optional `eye`, `measurements`, raw stereo `frame` |
| `interpret` | `reference`, `input` |
| `representation_relationship` | `left`, `right` |
| `revise_representation` | `left`, `right`, measured `equivalent` |
| `choose_representation_test` | `pairs: [{left, right}]` |
| `form_representation_concept` | `reference`, `observations` |
| `recognize_representation_concept` | `reference`, `input`, optional boolean `expected` |
| `compose_representation` | `first`, `second`, `observations` |
| `solve_represented_problem` | `input`, `conditions`, optional `operations`, bounds |

`frame` has the SensorFrame fields `tick,width,height,left,right,focal,baseline,proprio,
executed`, with hexadecimal RGB bytes. Supplied frames are validated; no authoritative
labels are accepted. References in `operations` must be currently available native
operations. Existing world, process, concept, rule, save and restore commands still work.

```powershell
.venv\Scripts\python.exe -m athanor.grounded.representation_demo
.venv\Scripts\python.exe -m athanor.grounded.app --checkpoint artifacts\representation-observer.athanor --paused --port 8766
.venv\Scripts\python.exe -m athanor.grounded.app --checkpoint artifacts\representation-observer.athanor --problem my-problem.json --headless --paused --steps 0
```

The state API exposes actual observations, current alternatives, native dependencies,
derived structures, concepts, fragments, tasks, failures, search effort and learning.
Viewer inspection creates no new evidence. Full checkpoint format 6 adds these references,
cached result provenance and inquiry state; formats 1–5 remain readable.
