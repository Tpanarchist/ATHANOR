# ATHANOR — decisions and unresolved questions

Date: 2026-10-10. Decider: implementation workstream under Goal 001 authorization.
Status: Accepted for Goal 001; revisitable without implying scientific validation.

## ADR-001: versioned local structural envelope

**Context:** identity, composition, reflection, and inspectable change must agree.
**Decision:** immutable revisions, namespaced local IDs, exact references, compact
records, ordered members, and a small role discriminator.
**Alternatives:** mutable objects (cheap but historical composites drift); content
addressing alone (conflates equal state with identity); universal untyped graphs
(smaller kernel surface but ambiguous contracts).
**Consequences:** exact historical inspection and ordinary recursive composition;
storage grows. Fresh namespaces distinguish independent stores; an explicit namespace
and repeated action sequence reproduce identical snapshots. Restore preserves lineage.
IDs cannot be merged between stores implicitly.

## ADR-002: treat the three primitives as architectural vocabulary

**Context:** Distinguish, Bind, Transform must be evaluated rather than presumed.
**Decision:** expose their useful behavior, plus explicit teaching, creation,
composition, evaluation, and persistence mechanisms. No minimality theorem.
**Alternatives:** force all behavior into three opaque methods; many domain-specific
primitives. Both obscure where admissibility and computation actually happen.
**Consequences:** a small interpreter with real control flow/exact functions remains
necessary. Computational expressiveness says nothing about learning efficiency.

## ADR-003: safe installed expression data

**Context:** procedures must be teachable and survive restore.
**Decision:** fixed expression grammar, exact rational arithmetic, projections,
conditionals, sequence folds, nested/self calls; typed input/output interfaces;
bounded work and depth; staged atomic commit.
**Alternatives:** Python callbacks (easy but not self-contained persistence); Python
source/pickle (opaque and unsafe restoration); a full compiler (premature).
**Consequences:** reusable and recursive local procedures; limited language and
significant interpreter cost. Native extensions need explicit trusted installation.

## ADR-004: closed schemas and conservative inheritance

**Context:** taught specializations must reuse knowledge without copied edges.
**Decision:** resolve ancestors; aggregate compatible field contracts; require exact
agreement for duplicate fields. Relation definitions explicitly opt into source
inheritance. All claim evidence is returned; absence is unknown, disagreement conflict.
**Alternatives:** copy relationships into children (stale duplicates); assume every
relationship is transitive/inherited (unsound); silently override parent constraints.
**Consequences:** usable schema/relationship reuse; no automatic general theorem
proving, exceptions, defeasible inheritance, or confidence-based truth selection.

## ADR-005: standard-library validation and persistence

**Context:** explicit contracts and compact internals matter more than dependencies.
**Decision:** frozen typed records; boundary validation; strict tagged canonical JSON;
versioned format and digest; validation on restore; atomic file replacement.
**Alternatives:** Pydantic throughout (unneeded metadata/serialization cost here);
pickle (executable and weak portability); database (future scale milestone).
**Consequences:** no runtime dependency. A digest detects accidental corruption; it
is not authentication or proof of original truth. Failed save preserves the old file.

## ADR-006: definitions are reflectable but immutable

**Context:** the same operations should address external and internal representations.
**Decision:** definitions support comparison, binding, reference, and composition;
teach new definitions to extend behavior. Only data state can be revised in place.
**Alternatives:** unrestricted definition edits (invalidate extant interfaces), or
separate nonstructural operation registry (blocks first-class participation).
**Consequences:** safe operational reflection; no claim of autonomous self-improvement.

## ADR-007: optional shared recurrent neural observer

**Context:** installed structures need a real trainable activation/selection process.
**Decision:** optional PyTorch dependency; one feature encoder, sparse learned graph
messages, one GRU cell reused for field/context/decoder, variable candidate pointers.
Small readouts predict applicability, capability, preference and structural outcomes.
**Alternatives:** external teacher inference (violates local intelligence); symbolic
rule selection with decorative neural scores; fixed symbol embeddings/output heads;
separate world/self networks; mandatory full transformer. None is required here.
**Consequences:** actual learned use of newly taught nodes/operations with fixed
parameters. Hash collisions, bounded depth, graph closure costs and imperfect argument
binding remain limitations. Computational expressiveness does not prove efficient learning.

## ADR-008: preserve exact execution and qualified feedback

**Decision:** public-only cornerstone boundary, exact reference pointers, all installed
operations/offered inputs as neural candidates, no applicability repair/prefilter.
Kernel rejection remains atomic. Actual result/receipt/error returns to the observer.
Predictive, preference and capability discrepancies remain separate. Numeric output
values are never neural prediction targets. Correction is explicit teacher instruction.
**Consequences:** successful execution can still violate preference. Abstention is
learned and distinct from rejection. Confidence is uncalibrated; it cannot assert truth
or suppress contradictory cornerstone evidence.

## ADR-009: dynamic shared self-representation and recurrent training

**Decision:** an ordinary observer structure plus measured self-observations, active
field/context and activation history, all passed through the same graph process.
Train fresh and carried-field episodes. Direct query-edge descriptors also enter
pointer candidates to preserve positional context through repeated aggregation.
**Alternatives:** static biography; handwritten self reliability penalty; corrective
operation override; training only fresh states and assuming recurrent competence.
**Consequences:** demonstrated feedback-driven correction and internal processing
without new input. The initial failure of carried-state inference and loss of second
argument order required explicit recurrent curriculum and preserved relational context.
Broader task retention/interference and calibration remain unresolved.

## ADR-010: separate stores, combined resumable checkpoint

**Decision:** cornerstone knowledge, learned weights and temporary activations have
separate contracts. An atomic versioned checkpoint bundles those plus optional AdamW
state and local shuffle RNG. Restricted tensor loading, finite/shape/reference checks
and knowledge validation precede use. PyTorch remains optional for the foundation.
**Consequences:** same next decision/tensors and optimizer continuation on the exercised
CPU path. No executable Python callbacks/source/model classes are persisted. No
cross-version/hardware bitwise portability or checkpoint authenticity claim.

## Unresolved after the first neural model

- Grounding contracts: what observation supports which distinction/reference, and
  how contradictory observations affect interpretation without erasing provenance.
- Broader structural encoding and planning depth; learned attention/active graph budgets
  that remain explicit about excluded knowledge. Current encoding/selection is operational
  for the bounded curriculum, not established across all cornerstone semantics.
- Richer constraint refinement, generic polymorphism, units, dependent interfaces,
  exceptions to inheritance, and calibrated evidence aggregation.
- Explicit cross-store identity/unification and schema migration protocols.
- Retention/compaction of recursive execution histories without breaking references.
- Persistent indexes and transactional storage at useful scale.
- Evaluation of actual acquisition, transfer, correction, and capability calibration;
  these must precede claims of efficient learning or useful recursive intelligence.

These questions do not prevent the operational cornerstone and first neural observer.
They prevent treating this bounded supervised competence as grounded general intelligence.
