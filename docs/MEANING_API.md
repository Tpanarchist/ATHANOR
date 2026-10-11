# ATHANOR 0.3 — semantic teaching and observer API

## Native knowledge (no PyTorch dependency)

Import `Teacher`, `Lexicon`, `Example` from `athanor.semantics`. Ordinary `Ref`,
`Record`, `Evidence`, `Procedure` and exact expression data retain Goal 001 contracts.
Bindings/observations/definitions/history are native structures and survive `Athanor.save`.

| Interface | Contract |
| --- | --- |
| `Teacher.define(name, definition, procedure, expressions=..., examples=..., roles=..., context=..., general=..., source=...)` | Atomically validates/installs the procedure, checks each exact example, creates a concept and explicit convention bindings, and records capability availability. Returns concept Ref. |
| `Example(arguments, expected)` | Exact argument references and complete expected result fields; wrong output rejects the whole lesson. Checks are not universal proofs. |
| `Teacher.specialize(name, parent, fields, expressions=..., source=...)` | Existing native category inheritance, plus taught expressions. Inherited constraints/relations are not copied. |
| `Teacher.alias(expression, referent, context=..., evidence=...)` | Installs an attributed convention. Synonyms have separate symbols but may share a referent. No lexical resemblance test. |
| `Teacher.correct(expression, supersedes, referent, reason=..., context=..., source=...)` | Explicitly supersedes exact bindings of the same expression/scope, retains them and their evidence, creates replacement and correction records. Other disputes remain live. |
| `Teacher.observe(category, measurements, expression=..., source=..., context=...)` | Validates a typed participant and stores an attributed observation witness and expression binding. Returns participant/witness Refs. |
| `Teacher.claim(relation, subject, target, source=..., stance=..., observation=...)` | Empirical assertion stays uncertain unless a separate matching relation observation supports its exact endpoints/stance. Support records retain witness references. |
| `Teacher.observe_relation(relation, subject, target, source=..., stance=...)` | Records a scoped observation independently of teacher assertion; callers are responsible for the external source's reliability. |
| `Lexicon.bindings(expression, context=None)` | Returns binding Ref, symbol Ref, referent Ref, scope and installed evidence assessment. Missing context retains alternatives; conflicting supplied context excludes a binding. |

Text normalization is Unicode NFKC, case folding and whitespace. Exact normalized
spelling is a convention key, never an assertion of equivalent meaning. Expressions
contain bounded words/spaces/hyphens. The exact lexical adapter recognizes the longest
installed span, retaining every competing binding of that span. It does not dispatch
an operation. Unverified convention evidence cannot authorize an interpretation.

`Athanor.transaction(work)` passes a staged kernel to the callback and publishes all
its successful mutations together. Exceptions discard structures, IDs and receipts.
Use the provided staged instance for nested transactions. External side effects are
not transactional. `category(of)` is a new exact expression returning a structure's
category Ref or null, enabling installed category traversal without a neural classifier.

## Learned language use

`SemanticObserver(Observer(kernel, model), quantity_category)` wraps the same Goal 002
observer. Numeric lexical literals become exact instances of the supplied quantity
interface. The initialization curriculum stores a native `language_anchor` for restore.

`interpret(text, context=None, distractors=())` tokenizes and creates an ordinary
utterance graph, then calls the shared neural field. `inspect()` returns:

- `status`: accepted, unknown, ambiguous, conflict, unresolved, or insufficient.
- `meanings`: context-compatible concept references, learned probabilities and evidence.
- `selected`: the neural concept pointer, possibly null.
- `decision`: learned operation and ordered input pointers, or explicit abstention.
- `reason`: the acceptance/veto explanation.

Evidence safeguards only veto. They cannot choose an operation, fill a slot or swap
inputs. Acceptance requires a unique supported convention, the learned concept pointer,
an agreeing learned operation pointer and grounded proposed inputs. This checks support
and execution eligibility, not universal semantic correctness. A confidently wrong
argument order can still be a valid kernel call; exact examples and execution feedback
are therefore essential. Confidence is not calibrated.
Grounding respects mention multiplicity: one textual occurrence cannot silently be
consumed twice. Explicit repeated mentions can refer to the same participant. Pending
interpretations are rechecked against current conflicts and adopted supersession before
acting; an earlier positive claim cannot override subsequently contradictory evidence.
Scoped conventions require their context before acceptance.

`refine(steps=None)` processes the same observation further through the shared field
and returns an updated interpretation. It supplies no answer or corrective operation.
The demonstration permits bounded additional refinement only for unresolved/insufficient
activation; evidence ambiguity, conflict and unknown knowledge still block execution.

`act(evaluate=None, correction=None)` validates/executes the accepted learned proposal
through the kernel and records measured feedback. Preference evaluation is optional;
success is not automatically preference satisfaction. The operational self-map retains
both execution results and semantic recognition records. These re-enter the same field.
Installation, recognition, actual success and predicted confidence remain distinct.

`sequence(text, context=None, max_refinements=3)` accepts 1..4 semicolon-separated clauses.
It permits bounded further processing for unresolved activation (0..8 refinements);
unknown, ambiguous and conflicting evidence still stop it. `it` explicitly
refers to the most recent successful result. A clause's resulting structure can become
the next clause's input. Each kernel call is atomic; previous successes remain if a
later interpretation stops. No general planning or cross-clause rollback is implied.

Bounds: 1..1024 characters, at most 64 lexical tokens, at most 64 characters per token;
signed integers, fractions and decimal literals parse exactly. Supported learned
grammar includes `add X and Y`, `the sum of X and Y`, `subtract Y from X`,
`difference X and Y`, `divide X by Y`, `quotient X and Y`, `is ENTITY a CATEGORY`,
and `does CATEGORY include ENTITY`, plus taught expressions in those structures.
Unknown content is explicit. Grammar words are a bounded lexical convention; there
are no operation-specific argument-reordering rules. Additional grammar needs teaching
examples and adaptation; an arbitrary definition paragraph is not an executable lesson.

## Training and persistence

`athanor.semantics.curriculum.lessons(count, config, seed=3000)` produces fresh and
active episodes for each native teaching case, including actual execution histories,
aliases, corrections, context, insufficiency, category knowledge and composed procedures.
It uses Goal 002 `Lesson`/`Trainer`. A lesson's optional `meaning` target trains the
existing shared scorer over variable concept candidates; no new neural head or symbol
parameters are allocated. The demo also replays earlier structural lessons.
History length and preceding task types vary independently of lesson mode; pronominal
examples reference the actual latest execution result. Category depth, field names/types
and observed instances also vary. The default demo first reproduces Goal 002's training;
`--base-checkpoint artifacts/neural-observer.athanor` extends the existing trained
checkpoint directly. Semantic training adds 1,000 optimizer updates at default settings.
The report separately measures retention on the original structural held-out cases.

`SemanticObserver.save(path, trainer=None)` refreshes the active graph and atomically
saves native knowledge, weights, field, context, histories and optional optimizer/RNG.
`SemanticObserver.restore(path)` returns observer and optional Trainer. Checkpoint
version 2 preserves meaning logits and accepts original version 1 neural checkpoints.
Native snapshots retain their format; PyTorch is unnecessary to restore/execute knowledge.
Restricted tensor loading, digest, finite-value, shape and reference checks remain.
Cross-device/version bitwise equivalence and authenticity are not claimed.

For a new supervised language example, first install its convention/procedure using
`Teacher`, then construct an ordinary language observation with `Language.encode`.
The instructor supplies a `Proposal` with exact ordered mention references and a concept
Ref. Build its `Frame` with the existing `GraphBuilder`; validate/measure the target
through `measure(kernel, Decision(...), evaluate=...)`, and pass
`Lesson(frame, target, target, feedback, meaning=concept)` to `Trainer.train`.
For example, teaching `remove Y from X` supplies references in order `(X, Y)` as a
supervised target. The language adapter retains `(Y, X)` textual mentions; it performs
no reorder itself. Additional varied examples teach the grammar through the same
scorer/recurrent parameters. The curriculum demonstrates this complete pipeline with
independently varied values, aliases, histories and native definitions.

## Optional offline AI lesson preparation

An external teacher may produce a JSON document. It is an offline author, not part of
Athanor inference. `athanor.semantics.lessons.import_lesson(kernel, text, names=None)`
validates and atomically installs it. It accepts no Python source or model callback.

Top-level fields: `format: "athanor.lesson.v1"`, nonempty `source`, `procedures` array;
optional arrays `specializations`, `mappings`, `claims`. Unknown/duplicate fields reject.
Each procedure supplies name, definition, expressions, procedure and nonempty examples;
optional roles/context. Procedure inputs each specify role/category; result/fields are
required; members/update are optional. Expressions use the existing portable grammar.
Examples supply argument references and complete expected result fields. Every example
is executed and compared exactly before adoption. A wrong example rolls back the pack.

Exact refs use `{"ref": ["s00000001", 1, "namespace"]}`; supplied/imported local names
use `{"named": "quantity"}`; rationals use `{"fraction": [3, 2]}`. A procedure name
resolves to its concept; `name.operation` resolves to its installed operation.
Forward/unknown names reject. Arrays/objects map to native tuples/Records; floats reject.

Specializations supply name, parent, fields, expressions; each field contract includes
kind/required/minimum/maximum. Mappings supply expression/referent and optional context/
stance. Claims supply relation/subject/target and optional stance, and remain uncertain.
The import surface cannot mint observation witnesses. Separately attributed observations
are required for empirical support. Import bounds: 256 KiB, arrays at most 64 entries,
native lesson nesting at most 32. These are input safety contracts, not model-size limits.

The integrated demonstration writes a complete executable `.lesson.json` example and
its `.knowledge.json` separately. The CLI is:

```powershell
athanor-teach lesson.json --knowledge existing-native.json --save taught-native.json
```

Exact execution validates an implementation against supplied checks and native constraints;
it does not prove a teacher's intended mathematical definition universally. Observation
attribution identifies a source/scope; it does not authenticate the source or prove the
measurement's external truth. Those boundaries stay visible in native evidence records.
