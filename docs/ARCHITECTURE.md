# ATHANOR — Recursive Intelligence Architecture

Athanor is the furnace of transformation: Foundation + Knowledge + Experience;
Activation → Transformation → Refinement. The hypothesis is that a relatively small
collection of reusable processes, composed across representations, may support
complex intelligence. Goal 001 makes the operational substrate concrete; it does
not establish that hypothesis experimentally.

## Intended system

```text
instruction / observations / demonstrations / consequences
                    ↓
domain interfaces and grounding (future)
                    ↓
shared recurrent structural encoding / activation / pointers (Goal 002)
                    ↓ proposed exact structural actions
kernel validation and bounded execution (Goal 001)
                    ↕
versioned structures / taught schemas / typed claims / reusable procedures
                    ↓
execution receipts / rejections → typed feedback / operational self-map (Goal 002)
                    ↓
supervised parameter updates / checkpoints (Goal 002)
grounding / broader calibration / continual learning (future)
```

The kernel owns computational admissibility, not truth. Knowledge is represented in
immutable structural revisions. Transient evaluators propose/execute changes against
that knowledge. Exact arithmetic and established procedures execute locally.
The neural component selects and interprets installed structures without replacing
deterministic execution with approximations. Its first operational self-map contains
measured decisions, successes/rejections, expectations and separate discrepancies,
corrective instructions, capability counts, and activation history. All are encoded
through the same recurrent machinery as external structures.

## Precise Goal 001 boundary

Implemented scope: compact immutable values and local references; data/category/
relationship/claim/operation structures; exact category inheritance; validated
attributed bindings and conservative assessment; first-class ordered composition;
small executable expression language; atomic nested execution and identity-preserving
updates; production receipts; structured teaching; strict deterministic persistence.

The implementation uses one structural envelope for every role. Role-specific
contracts remain explicit. Teaching may extend knowledge and procedures inside the
fixed interpreter; installing a new native primitive requires a deliberate engine
change. Installed procedures are data, survive restore, and do not need an external
model, dynamic import, or retraining.

Outside Goal 001: learned parameters, natural-language understanding, sensory grounding,
automatic category discovery, probabilistic inference, relevance ranking, continual
neural learning, learned self-map, autonomous self-improvement, chatbot/RAG interfaces,
agent orchestration, and digital-universe integration.

## Implemented Goal 002 connection

The optional PyTorch module reads immutable public structures and pinned references.
One structural encoder, sparse learned relational messages, and one reused GRU cell
refine a dynamic field and decode operation/argument pointers. Reference identities
remain boundary handles; they are excluded from neural features. New structures and
operation candidates reuse existing parameters. Direct query-edge context preserves
sequence positions at the pointer boundary, without an argument-selection rule.

The observer returns exact proposals to `Athanor.execute`; it never repairs them or
prefilters candidates by executable interfaces. The unchanged cornerstone validates
and executes atomically. Success returns the actual result and receipt; rejection
returns the actual error. Abstention represents a learned insufficient-information
choice. Success, preference satisfaction, and structural outcome prediction remain
separate. Explicit self-observations reenter the same encoder after feedback.

Supervised lessons install real knowledge through Goal 001 and train pointer use and
distinct feedback predictions from actual executions. Teaching and optimization are
separate actions. Temporary tensors, installed knowledge, and learned weights have
separate contracts. A versioned atomic checkpoint saves all three plus optional
optimizer/shuffle state, and restores the same next CPU computation. See
[NEURAL_OBSERVER](NEURAL_OBSERVER.md) for the integration review and
[NEURAL_API](NEURAL_API.md) for training/inference.

Goal 002 does not supply natural-language/sensory interpretation, proof search,
calibrated open-world confidence, autonomous neural training, unrestricted continual
learning, or controlled self-improvement. Structured correction is externally supplied.

Architecture-changing proposals should create new procedure/schema definitions,
receive tests and controlled review, and explicitly redirect future usage. Rewriting
the validator or interpreter from inside a taught program is not permitted.

## Cost and concurrency

Immutable histories share values/references in memory. Transactions stage shallow
copies of store indexes/receipts, so mutation has an O(N + H) copying cost in this version;
history and provenance grow with committed computation. Inheritance/claim lookup
traverses installed knowledge without persistent copied edges. No vector or neural
dimension is assumed. A single kernel serializes mutations; there is no distributed
identity, concurrent transaction merge, garbage collection, or durable database.

Before scale-up, measure workload cost and add indexes, dependency-aware retention,
and a durable transactional store behind the same contracts. Interpreter work limits
are not a time/memory sandbox: large arithmetic operands and hostile snapshot sizes
need separate controls before accepting untrusted remote input.

Neural features use fixed-size signed hash bins, so token collisions and compressed
number descriptors lose information. Exact values remain in the cornerstone. Active
graph construction follows references and incident claims; it currently scans all
knowledge and includes all installed operations. Graph closure can exceed the explicit
node budget, in which case processing fails rather than silently discarding knowledge.
For N active nodes, E reference edges, width D and K steps, recurrent work is roughly
O(K(N+E)D²), with O(ND+ED) activation memory per step (training retains K steps).
The bounded active self-history does not delete durable self-observations. Cross-episode
retention/interference, evidence conflict and relation inheritance have not received a
general neural competence guarantee. These are next integration questions, not reasons
to weaken exact cornerstone invariants.
