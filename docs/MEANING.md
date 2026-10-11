# ATHANOR Goal 003 — meaning, teaching and grounding contracts

Status: integration design and critical review established before model changes,
2026-10-10. Goal 001/002 remain the computational and neural substrate.

## Operational meaning, not identity or textual resemblance

A symbol is an ordinary data structure containing a bounded normalized expression.
Its identity is an exact namespaced reference, not its spelling. Normalization only
standardizes Unicode/case/whitespace; it does not establish semantic equivalence.
An explicitly taught meaning binding connects a symbol to a referent, context and
attributed evidence. Several expressions can have bindings to one referent. One
expression can have several bindings in different or overlapping contexts.

Referents may be data, categories, relationships or executable concepts. An executable
concept is an ordinary structure referencing its procedure, definition, role names,
examples and optional general concepts. Procedures remain exact installed expression
data. Semantic composition includes applying results to subsequent expressions and
teaching a new procedure that composes existing ones. A definition paragraph alone is
not executable understanding; usable meaning requires relationships, interfaces,
examples, observations or procedures.

Existence of a binding is not proof about the world. Stipulated expression conventions
and validated executable contracts are distinct from unverified empirical claims.
Grounding here means attributed structured observations linked to local participants,
not unrestricted sensory recognition or a philosophical proof of semantic aboutness.

## Exact boundary versus learned interpretation

The language adapter performs bounded Unicode tokenization, exact rational parsing,
explicit expression/reference grounding and contextual evidence lookup. It constructs
token structures and an utterance graph with mentions in textual order, not argument
role order. It never rewrites a phrase into a selected operation or silently swaps
arguments. Grammar words and taught symbols are visible data. Unfamiliar content remains
unknown. Argument candidates include distractors and are offered independently of
target order. Pronoun `it` refers explicitly to the previous successful result.

One existing encoder/GRU/message process handles these structures. The existing shared
pointer scorer also scores concept candidates; this adds no concept-specific network,
class head or symbol embedding table. Neural operation and ordered argument proposals
must agree with the activated concept's installed interface. Numeric values are
executed by the cornerstone, not predicted as language answers.

Interpretation returns multiple candidate meanings and neural activation probabilities.
An explicit evidence policy vetoes unsupported, uncertain, conflicting or unresolved
meanings, even when neural scores are high. It cannot select or repair an operation.
Context can narrow the admissible meanings and produce another recurrent processing
step. Unknown and contested interpretations remain different from confident acceptance.
This policy is deliberately conservative; probability alone cannot erase evidence.

Possible taught referents are explicit links on the utterance as well as attributed
binding structures. This preserves the observation-to-meaning interface through
recurrent communication when many past self-observations compete for activation.
Every alternative is linked; neither an operation nor argument roles are selected
by this representation. Requiring the neural field to reconstruct an already taught
reference through avoidable indirection degraded live continuity. Direct grounding
fixes that representation issue without adding model parameters or hidden selection.

## Teaching and validation

Definition, relationship, procedure, example, specialization, correction and uncertainty
are teaching functions built from shared cornerstone operations. Teaching records
source and stable references. Complete procedures are checked against interfaces and
exact examples before installation; passing examples are evidence, not universal
proof. Specializations use existing category inheritance and retain general contracts.
Failed multi-part lessons must leave knowledge, receipts and identity allocation intact.

The minimum foundation extensions are:

1. An atomic public transaction over ordinary kernel operations, to install a lesson
   and validate all its examples as one unit. No changed reference/execution semantics.
2. A `category(of)` reflection expression, complementary to `field` and `members`,
   allowing taught general categorical procedures to traverse existing parent links.
   The neural model does not implement or approximate category membership.

Meaning corrections explicitly supersede specified convention bindings and record
reason/replacement. Supersession is adopted instruction, not silent truth selection;
the old claims remain inspectable. Independent contradictory bindings remain contested.
Examples/corrections may also train uncertain grammar/role binding. Durable knowledge
installation and neural parameter adaptation are separate, explicit actions.

## Offline external teacher boundary

An optional offline JSON lesson pipeline translates teacher-prepared definitions,
portable procedures, exact examples, expression bindings and claims into these native
structures. It accepts no Python source, model callbacks or live teacher dependency.
Procedure lessons require independent exact example checks and kernel contracts.
The importer cannot fabricate observation receipts. Empirical assertions remain
uncertain unless supported by a separately supplied, appropriately scoped observation;
an AI assertion alone never becomes verified evidence. Provenance is attribution,
not authentication or proof of external truth.

## Self-map, state and persistence

Installed, recognized, successfully exercised and confidently predicted capabilities
remain distinct. Installation records availability; neural interpretation records
recognition; actual execution updates measured success/failure; learned readouts retain
their separate confidence/expectation meanings. The existing dynamic self-map and
three discrepancy channels continue through the same observer.

Domain knowledge, lexical conventions and observed interaction records are durable
structures. Learned weights/optimizer and temporary activation remain separate.
The versioned combined checkpoint includes concept logits and retains restoration of
Goal 002 checkpoints. A restored semantic interface reconstructs its lexicon from
native knowledge and continues without its teacher. Active history/graph limits are
explicit; no silently truncated semantic evidence.

## Critical review and unresolved boundaries

- A phrase-to-operation table would bypass learning: lexical bindings only expose
  candidate referents; learned concept/operation/role pointers are measured on fresh
  values, aliases, categories and procedures. No operation-specific parsing rules.
- A bag of words loses operand roles: preserve token and mention position edges and
  train both direct and reversed noncommutative expressions. Measure exact bindings,
  not merely commutative numeric answers.
- New vocabulary could require a new network: new bindings/nodes reuse fixed learned
  parameters. Complete operational definitions are tested with weights unchanged.
- High confidence could hide ambiguity: explicit competing meanings/evidence veto
  execution, and additional context changes the field. Alternatives remain visible.
- A correction could erase disagreement: adopted supersession is explicit; ordinary
  conflict remains recorded and blocks unsupported certainty.
- A teacher could assert facts as evidence: independent execution validates local
  mathematics; observed claims need separate observation attribution/scope.
- Multi-expression execution could imply all-or-nothing dialogue: kernel calls are
  individually atomic; successful earlier steps retain receipts if a later step stops.

The initial language is bounded English clauses over taught terms, exact quantities,
named observed entities/categories and explicit contextual qualifiers. This is neither
a chatbot nor retrieval over explanations. No open-domain grammar, learned perception,
calibrated open-world probability, universal theorem proving, autonomous curriculum,
or unrestricted continual learning is claimed. Scaling active relational closure and
resolving long-lived interference remain engineering work.
