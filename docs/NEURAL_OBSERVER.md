# ATHANOR Goal 002 — precise integration design

Status: integration contract established before implementation and implemented/validated,
2026-10-10. The cornerstone requires no semantic extension.

## Boundary with the implemented cornerstone

The neural observer uses only public `Athanor` methods: immutable structural views,
exact `get`, `create`/teaching for observations and self observations, `execute`,
`inspect`, and canonical persistence. No foundation extension is required. Namespaces,
pinned revisions, closed schemas, immutable definitions, production receipts, and
atomic nested execution retain their Goal 001 meanings. A rejected action can append
an attributed self-observation; that is explicit new knowledge, not a partial execution.

An observation is an exact data reference plus an explicitly offered candidate pool.
The domain adapter supplies structured context, not the correct executable action.
The model reads a structural graph containing the observation, offered participants,
installed operations, their definitions/dependencies, relationship claims, and recent
self observations. Graph construction follows declared references and relationship
structure. It does not choose an applicable operation or prefilter arguments by
the operation's category interface. The cornerstone makes that final determination.

## Shared neural process

All structures use one encoder: role/type/shape features, stable hashed attribute
tokens, compressed exact-number features, and reference edges with typed contexts
and sequence positions. Arbitrary IDs, namespace strings, allocation order, revision
numbers, and display names of categories/operations do not enter learned features.
References remain exact handles outside the network. A fixed-size token feature
space shares parameters across new symbols; a new structure allocates activations,
not a new neural module or output neuron.

Sparse learned messages over reference edges feed one shared GRU cell. Repeating
that cell refines the activation field. The same cell also updates the pointer
decoder context. Operation choice and variable-length argument binding use a shared
candidate scoring function. The inventory, rather than a fixed class list, supplies
operation candidates; an explicit learned abstention candidate represents insufficient
information. Argument candidates are not supplied in target order.

Direct reference-edge descriptors also enter candidate pointer vectors. This preserves
sequence position after recurrent aggregation. The path/position projections are
shared and learned; the decoder does not look up a requested slot as an execution rule.

Width is chosen to carry attribute encoding, sparse relational context, and recurrent
state coherently at the small initial workload. It is configurable, not a model-size
ceiling. Neither a pretrained transformer nor separate world/self networks are used.

## Three separate stores of state

1. Installed knowledge: immutable cornerstone structures, definitions, claims, and
   explicitly recorded observations/self observations.
2. Learned parameters: encoder, message/gate functions, shared recurrent cell,
   pointer scorer, and small prediction readouts.
3. Runtime activation: per-reference vectors, global recurrent context, current
   observation, refinement count, recent activation history, and self-observation
   references. Runtime tensors are detached between inference calls; no accidental
   unbounded autograd history is retained.

The observer can refine its current field without a fresh external observation.
New knowledge enters by encoding new structures with the existing weights. Previous
vectors are aligned by exact reference, preserving revision distinctions.

## Proposal and feedback contracts

A proposal contains an exact operation reference (or abstention), an ordered tuple
of argument references, confidence, expected execution success, expected preference
satisfaction, and an expected structural outcome signature. Numeric output values
are never a neural execution target: installed arithmetic remains exact.

The adapter invokes `Athanor.execute` without silently repairing the proposal.
Successful results return their exact result and receipt. Rejections return the
actual exception category/message. Abstention is distinct from execution failure.

Expectation + observation gives three typed discrepancy channels:

- Predictive: expected versus observed structural outcome, available after success.
- Preference: externally supplied satisfaction versus expected satisfaction; missing
  evaluation remains unknown rather than being inferred from successful execution.
- Capability: actual execution acceptance versus predicted success; absent for
  abstention, where no capability was exercised.

No combined reward replaces these channels. Structured corrective instruction may
name a preferred operation/argument binding. It is explicit teacher feedback and is
recorded as such, never a hidden inference-time oracle.

## Operational self-map

The observer is itself an ordinary structure. Its self observations reference that
observer, its available operations, current observation, decision, confidence,
predictions, three discrepancy channels, activation summary/history, execution result,
receipt, and measured attempts/successes. They are encoded through the same graph
machinery as external structures. Their influence on later proposals is learned;
there is no separate hand-written reliability penalty or corrective selector.
Counters are measurements, not a second learned model or static biography.

## Teaching and training

The curriculum installs categories, exact procedures, intentions, and attributed
relationships through Goal 001 teaching. A separate supervised pipeline teaches the
network to use that knowledge. Labels name candidate references, not output classes.
Actual cornerstone executions supply capability/outcome targets, including rejected
probes. Preference targets and corrections are explicit curriculum evaluations.

Train with operation/argument pointer losses, an applicability readout, and separate
capability/preference/predictive losses. No target trains arithmetic approximation.
Training includes both fresh fields and carried activations, including the transition
from a prior decision to corrective self-evidence. A single AdamW optimizer updates
all learned machinery. Inference runs detached and does not silently optimize weights.
Held-out stores change namespaces, allocation/candidate order, values, category
specializations, compositions, and taught procedure definitions. After initial
training, install new knowledge without adding model parameters, then evaluate its
use. Demonstrate actual feedback correction and continued optimization, reporting
what changes with recurrent state, self observations, and learned weights.

## Checkpoint and failure contract

One atomically replaced checkpoint contains canonical cornerstone knowledge, model
configuration and tensors, runtime state, current observation, self-map references,
and optional optimizer/local shuffle state. Digests and strict format/shape/reference
checks precede restoration. Load tensor/primitive state with PyTorch's restricted
`weights_only=True` path; never load serialized model classes or Python source.
Checkpoint hashes detect corruption, not authenticity. Continuation preserves
optimizer state and exact activation tensors on the same deterministic CPU path.

## Critical review and deliberate limits

- An ID lookup/class head could memorize examples: exclude identity features and
  test renamed/reallocated/permuted and newly taught structures.
- A symbolic prefilter could do the selection: enumerate candidates without
  interface repair; the network proposes and the kernel rejects invalid actions.
- A decorative neural layer could coexist with scripted behavior: measure initial
  versus trained held-out proposals, parameter changes, real execution, and correction.
- Missing information could become a fabricated answer: learn explicit abstention
  and preserve missing preference/observation evidence.
- A static self description would not adapt: measured self observations and activation
  state recur into later processing; test their effects on proposals.
- A checksum could imply authenticated history: retain Goal 001's distinction between
  well-formed state and established provenance/truth.

This is supervised structural competence in a bounded curriculum. It does not establish
open-domain reasoning, calibrated out-of-distribution uncertainty, sensory grounding,
unrestricted continual learning, consciousness, or autonomous architecture refinement.
Active graph/history bounds are explicit runtime resources, not model-size ceilings.

## Verified learning evidence and remaining gap

The default deterministic CPU run uses width 64, four refinement steps, 256 lessons,
40 epochs and 640 optimizer updates. It has 107,980 parameters; this is a reported
implementation size, not a ceiling or objective. Held-out exact proposal accuracy
improves from 0/128 to 122/128; operation selection is 128/128. All 48 insufficient
cases abstain. Among 80 actionable cases, 74 have exactly correct proposals and 75
reach the expected numeric result (one incorrectly ordered binding is numerically
equivalent under a commutative procedure). The six
remaining errors are argument bindings, even though the chosen operation is correct.
All 48 additional new-definition cases produce the expected exact results.

The executable feedback episode starts with a model-generated proposal. The execution
succeeds but a changed preference is unsatisfied. Explicit corrective instruction is
recorded in the self-map; the learned model changes its next proposal and the kernel
executes the corrected result (-15/2). With the active self-evidence and carried field
removed, the original instruction again determines the proposal. This is bounded
learned use of feedback, not autonomous discovery of the user's preference.

A multiplication procedure is installed after optimization; neural pointers select
it and bind its inputs without extra parameters or optimization. The exact result is
-14. Save/restore preserves knowledge, activation tensors, pending decision and the
next internal computation. Continued supervised training changes restored weights.

No inference capability is inferred from loss alone. See `artifacts/neural-observer.report.json`
and the automated acceptance test. Softmax confidence and success predictions are not
calibrated truth estimates; broader capability calibration, multiple changing tasks,
contradictory evidence and neural retention require additional contracts and evidence.
