# ATHANOR — decisions and unresolved questions

Date: 2026-10-10. Decider: implementation workstream under Goals 001–004 authorization.
Status: Implemented contracts; revisitable without implying scientific validation.

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

## ADR-011: meaning conventions are attributed structural knowledge

**Context:** spelling, identity, operational meaning and truth must remain distinct.
**Decision:** symbols and scoped referent bindings are ordinary native structures;
typed denotes claims carry evidence. Concepts reference definitions, roles, examples
and exact procedures. Synonyms share referents, not symbol identities. Explicit
correction supersedes specified conventions and preserves old evidence.
**Alternatives:** lexical similarity as equivalence, destructive dictionary replacement,
or treating a teacher assertion as world truth. These conceal uncertainty/provenance.
**Consequences:** explicit alternatives/conflicts, persistent acquired meaning, scoped
observations. Grounding is attributed local support, not authenticated external truth.

## ADR-012: learned meaning and ordered language proposals reuse the observer

**Decision:** bounded lexical observation preserves token/mention positions and taught
reference links; it never emits a selected operation or roles. Existing scorer also
scores variable concepts, reusing the same recurrent field. Evidence policy can veto
but cannot repair learned pointers. Newly installed knowledge allocates no neural weights.
**Alternatives:** phrase-command routing, operation-specific parsing, separate per-concept
networks, a pretrained language model in inference. None establishes the requested
shared acquisition/use mechanism.
**Consequences:** train both fresh and carried states across different prior observations,
real self-feedback, aliases, corrections, composed definitions and two concept families.
Early fresh-only success exposed history interference; curriculum coverage is a real
contract, not a justification for claiming broad generalization. Confidence remains
uncalibrated, and valid but wrongly ordered calls require examples/feedback to detect.
Possible referents are explicitly linked to the current observation as well as its
attributed binding records. This exposes an established interface directly to shared
relational communication and reduces stale-history interference. All alternatives
remain candidates; it does not choose an operation or reorder inputs.

## ADR-013: atomic lessons and scoped independent validation

**Decision:** public staged multi-call transaction; exact category reflection for taught
categorical algorithms; offline JSON procedures checked by exact examples and native
contracts; teacher empirical assertions remain uncertain without separate observation.
**Alternatives:** partial installation on failed checks; native membership classifier;
executing teacher Python; adding observation support from teacher prose alone.
**Consequences:** foundation remains independently usable; old semantics preserved.
Examples validate implementations against supplied checks, not universal proofs.
Observation provenance is attribution and scope, not authentication. Explicit support
records connect claims and witnesses. Original neural checkpoints remain restorable.

## ADR-014: separate native territory, restricted interface and acquired map

**Decision:** an independent authoritative Athanor store holds bounded 3D rigid
geometry, a spherical body, time and causal receipts. RGB stereo/proprioception is
the only inference port. The outside viewer and privileged training labels remain
separate authority paths. Learned map identities never copy territory references.
**Alternatives:** object-ID/coordinate observations, a full game engine, or using
the observer's beliefs as world authority. These fail the intended interface boundary
or add unnecessary scope. **Consequences:** actual pixels, occlusion, deterministic
native revision history, inspectable physics approximations and explicit hypotheses.
See GROUNDED_OBSERVER for the design and critical review recorded before implementation.

## ADR-015: sensory computation supplements the same recurrent intelligence

**Decision:** one shared CNN learns foreground/continuous appearance; shared neural
fusion combines pixels/proprioception with a compact native intention view through
the existing structural encoder. The same messages/GRU/operation pointers refine
and select. A conditioned readout of that field predicts physical consequences.
Exact connected components, descriptor association and stereo geometry operate on
learned measurements, preserving ambiguity. Distinct learned appearances separate
touching surfaces instead of blending object and occluder into one false identity.
**Alternatives:** a second world/self reasoner, one classifier per taught symbol,
end-to-end approximation of exact physics, or raw receipt-heavy query hashing alone.
**Consequences:** new taught names/instances allocate no parameters. Current query
evidence remains available alongside recurrent refinement on the sensory path.
No-sensory computation and original parameter names remain unchanged. Appearance
and identity are limited; learned descriptors plus exact association are not a learned
universal identity or object-permanence theory.

## ADR-016: consequential evidence, explicit updates and retention

**Decision:** validate native commands, then measure the actual physical outcome.
Keep prediction, preference and capability channels distinct; retain world/self/interface
alternatives when evidence is insufficient. Empirical action summaries are reusable
native knowledge, qualified by experience. Supervised perception/action learning,
current-parameter carried context, exact execution labels and measured feedback updates
share the existing model. Structural/semantic replay limits cross-domain interference.
**Alternatives:** treating a valid command as physical success, hidden scene solutions,
hand-authored reward as truth, or assuming successful fresh-state tests establish
continuous operation. **Consequences:** the continuous acceptance exercises occlusion,
active sensing, independent motion, hidden fault, transfer and resumed behavior.
Training includes isolated views and continuous movement episodes with up to four
detached preceding frames. Approach labels use learned stereo bearings rather than
an unavailable authoritative bearing. The sensory query encodes a compact current
interface, with full map/self structures retained as relational context. Replay
reuses the original trainer's full losses and carried-state contract with the same
model, optimizer and RNG; abbreviated pointer-only replay did not retain enough
structural competence. Truncated training is bounded; unrestricted continual retention and calibrated
causal attribution remain unresolved.

## ADR-017: atomic complete-state checkpoint and a local live application

**Decision:** an outer versioned archive embeds the original observer checkpoint,
native world and separate sensory/runtime/training state. Pending pre-action experience,
optimizer, RNG and frozen/trainable flags restore with parameters. The next recurrent
step, physical continuation and optimizer update are checked. A standard-library local
HTTP service serializes actual world/observer operations; browser canvases show real
state and raw cameras. Headless execution uses the same coordinator.
**Alternatives:** serializing executable Python, replacing old formats, cosmetic
animation, omitting pending experience, or adding a graphics dependency just for controls.
**Consequences:** existing Goal 001–003 formats remain readable. Distribution includes
the HTML asset. Same-platform deterministic continuation is exercised; hashes are not
authentication. Application access is local, not an authenticated remote product.

## Unresolved after Goal 004

- Broader perception/effect grounding and externally authenticated observation sources;
  local attributed support does not establish universal or externally verified truth.
- Calibrated stereo confidence, indistinguishable-object identity, multi-view motion
  inference, general evidence-seeking/planning, and realistic materials/lighting.
- Controlled online update policy, retention under longer interactions and calibrated
  world/self/interface attribution beyond the bounded training distribution.
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
They prevent treating bounded taught operational meaning as general intelligence.
