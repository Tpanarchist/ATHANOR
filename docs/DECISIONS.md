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

## Unresolved before the neural model

- Grounding contracts: what observation supports which distinction/reference, and
  how contradictory observations affect interpretation without erasing provenance.
- Neural encoding of graphs, revisions, interfaces, programs, and dynamic activation;
  selection of computational paths and budgeting of executed intermediate structures.
- Richer constraint refinement, generic polymorphism, units, dependent interfaces,
  exceptions to inheritance, and calibrated evidence aggregation.
- Explicit cross-store identity/unification and schema migration protocols.
- Retention/compaction of recursive execution histories without breaking references.
- Persistent indexes and transactional storage at useful scale.
- Evaluation of actual acquisition, transfer, correction, and capability calibration;
  these must precede claims of efficient learning or useful recursive intelligence.

These questions do not prevent an operational cornerstone. They prevent treating
Goal 001 as a complete neural architecture or grounded intelligence.
