# ATHANOR — the operational cornerstone

Status: Goal 001 contract, 2026-10-10. This is an engineering foundation for the
architecture hypothesis, not a derivation of intelligence from three axioms.

## Definitions established before implementation

A **structure** is a locally identified, immutable revision containing a role,
optional category reference, a compact record of values, and an ordered tuple of
member references. Its identity is an explicit kernel allocation. It is not
inferred from equality of contents, a name, or a claim about a real object.
A **reference** identifies a store namespace, local ID, and exact revision. Identity
across a change means the same ID with the next revision, created by an allowed
update. It is an engineering continuity rule, not proof of physical persistence.

A **symbol** is a structure used by an interpreter. Labels aid humans; operational
significance comes from schemas, relationships, projections, and procedures the
interpreter can apply. A label alone supplies no grounding or semantic truth.
External significance remains a teaching/observation problem.

A **category** declares a closed field schema and zero or more parent categories.
Inherited schemas are consulted at use time. Duplicate inherited field declarations
must agree exactly; conflicting declarations are rejected. Specializations may add
fields, but cannot silently override a parent's contract. Categories classify both
leaf data and composites; membership and field state remain separate.

A **relationship interface** declares allowed endpoint roles/categories and whether
claims on a source category apply to its descendants and instances. A **claim**
records two exact endpoints, a relationship, and attributed evidence. Positive,
negative, and uncertain claims may coexist. Absence of a claim means unknown.
Inheritance propagates only the declared relationship applicability; it does not
copy edges, establish physical causality, or make an attributed statement true.

A **procedure interface** declares input categories, a result category, and whether
the result is a new identity or a revision of an input. A procedure is itself a
structure. Its body is bounded expression data evaluated by a small interpreter.
It can read inputs/fields/members, calculate exactly, branch, fold sequences, and
invoke other procedures or itself. The result obeys the same structural contract
and can be passed to further procedures.

A **composition** is a structure with ordered references to existing revisions.
Composition gives another participant, not an automatic inference about its
meaning. A composite's category determines applicable operations; members alone
do not imply that a sum, conjunction, or causal model is appropriate.

## Primitive review

| Candidate | Operational interpretation | Sufficiency and limitation |
| --- | --- | --- |
| Distinguish | Allocate explicit IDs; compare identity, revisions, field state, and members | Computable discrimination is useful; it does not discover which differences matter to an observer. |
| Bind | Validate and store an attributed typed claim between structures | Expresses relationships; cannot turn statements into truth or supply an inference rule by itself. |
| Transform | Evaluate a declared procedure and atomically commit validated results | Implements permitted change; requires an interpreter, exact functions, and resource bounds. |

These are a compact vocabulary, not three sufficient executable instructions.
Allocation/reference resolution, schema validation, exact expression evaluation,
bounded control flow, and atomic publication are necessary mechanisms. Composition
is ordinary structure construction with members. Teaching is validated construction
of category, relationship, and procedure structures. No artificial claim is made
that comparison, evaluation, or allocation disappear into a single primitive.

## Eight foundational questions

1. The minimum computational distinction is an allocated identity and interpretable
   state shape. A fieldless structure is valid; an empty label is not an identity.
2. Meaning to this interpreter is the available validated operations and attributed
   relations. Meaning about external reality remains evidence qualified.
3. Interfaces are declared admissibility rules for endpoints, inputs, and outputs.
4. Structures participate by exact reference, checked against input categories;
   expressions operate on their state and recursively on members.
5. The composite has the same structural envelope and input contract as any data
   structure. No separate composite execution engine is needed.
6. Operation/category/relationship definitions are structures and can be compared,
   bound, referenced, and composed. Definition mutation is deliberately unavailable;
   new definitions can be taught and invoked. This is operational reflection, not
   unrestricted self-modification.
7. Identity allocation, reference resolution, validation, arithmetic, comparison,
   inheritance traversal, execution, and persistence are exact algorithms. Relevance,
   perception, uncertain interpretation, prediction, grounding, and proposal choice
   are future learning tasks; none are claimed here.
8. Updates preserve identity and category, leave prior revisions intact, validate
   the complete new state, and preserve the ability to resolve provenance and all
   existing references. Composition preserves member order and exact revisions.

## Invariants

- Immutable, canonical values; no caller-owned mutable state enters the store.
- Only local exact references; no forward references or silent alias unification.
- Deterministic local IDs and receipt sequence for a specified store namespace; no
  timestamps or global store. A fresh default namespace distinguishes independent
  lineages; specify the namespace to reproduce an identical snapshot. Restore
  preserves it. Deliberate reuse of a namespace asserts common lineage, not a merge.
- Every revision has a production receipt with operation/input references and causes.
- All changes made by an invocation, including nested calls, commit together or fail
  together. Failure consumes no identity and publishes no receipt.
- Definitions remain immutable. Data updates preserve category and increment revision.
- Known references and field constraints are checked before publication.
- All categories in the parent graph are checked; incompatible multiple inheritance
  is rejected, not resolved by an arbitrary traversal order.
- Relationship evidence is distinct from procedure permission. A claimed affordance
  does not bypass an operation's actual input or result contract.
- Resource limits bound interpreter work and call depth; they do not bound model size
  or claim efficient learning. Structural persistence contains no transient evaluator.
- A restore constructs a fresh store only after strict format and invariant checks;
  executable input is interpreted data, never Python source, pickle, or eval.

## Critical review before coding

**Counterexample: equal labels, different objects.** Explicit allocation prevents
label collision from merging identity. Equal numeric state does not prove identity.

**Counterexample: a child inherits incompatible parents.** Exact schema agreement
rejects ambiguous contracts. Richer constraint refinement is deferred.

**Counterexample: revise a leaf after composing it.** Exact member references keep
the original composite stable. A new composition must explicitly reference the new
revision. Live views may be introduced later as a separate contract.

**Counterexample: the final output is invalid after nested calls have run.** Staged
publication rolls back every intermediate result and its receipt.

**Counterexample: an installed procedure recurses forever.** Work/call limits stop
execution without corruption. This does not prove termination of arbitrary bodies.

**Counterexample: two teachers disagree.** Both attributed claims remain inspectable;
assessment reports conflict. Neither insertion order nor confidence silently chooses
truth. Numerical confidence is metadata, not calibrated probability.

**Counterexample: a schema or program is itself supplied to a data operation.**
Category/role checks reject incompatible use. Definitions remain first-class
structures for binding/composition, while protected fields cannot be rewritten.

**Conclusion of review:** coherent for the requested bounded foundation, with explicit
limits. Computational expressiveness establishes executable reuse, not efficient
learning, grounding, adaptive intelligence, or a sufficient neural architecture.
