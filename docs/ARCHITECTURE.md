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
adaptive activation, interpretation and proposal selection (future)
                    ↓ proposed exact structural actions
kernel validation and bounded execution (Goal 001)
                    ↕
versioned structures / taught schemas / typed claims / reusable procedures
                    ↓
execution receipts and evidence → learning / calibration / self-map (future)
```

The kernel owns computational admissibility, not truth. Knowledge is represented in
immutable structural revisions. Transient evaluators propose/execute changes against
that knowledge. Exact arithmetic and established procedures execute locally.
The future neural component selects and interprets installed structures; it must
not replace deterministic execution with approximations. The self-map will eventually
include dynamic activation and measured capability/reliability patterns, beyond the
inspectable operation structures and receipts implemented here.

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

Out of scope: learned parameters, natural-language understanding, sensory grounding,
automatic category discovery, probabilistic inference, relevance ranking, continual
neural learning, learned self-map, autonomous self-improvement, chatbot/RAG interfaces,
agent orchestration, and digital-universe integration.

## Future neural connection

Use a read-only structural snapshot and stable references to encode candidate
structures, schemas, claims, and procedures. A model may return references and
structured action proposals to the same public API. Execution must validate current
inputs and produce explicit receipts. Rejected proposals become feedback, while
perception/observation evidence remains separate from successful execution receipts.
Saving neural state, activation history, optimizer state, and grounding observations
requires its own versioned contract. None is hidden in this foundation.

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
