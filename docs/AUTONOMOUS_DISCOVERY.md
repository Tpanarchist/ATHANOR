# Goal 005 design: autonomous discovery and continual learning

## Integration and authority

The Goal 004 observer already owns a native map, a shared recurrent model, pixel
grounding, motor procedures, measured self feedback, and a continued optimizer.
Missing mechanisms are inquiry selection, structured alternatives, operational
consolidation, and automatic bounded continuation. Extend that observer. The world
coordinator executes commands and supplies SensorFrame; discovery never receives World,
territory identities, mobility flags, geometry labels, or authoritative positions.

## Distinguish, bind, transform, compare, revise, repeat

Create native inquiry, candidate hypothesis, experiment, evidence, capability, and
acquired relationship structures. A hypothesis has a measured channel, conditions,
prediction family, fitted values, alternatives, supporting and contradictory evidence,
and a status. Fit general constant-effect and zero-effect alternatives to observations;
do not encode the simulator's effect values. An unresolved interface alternative
remains possible. Object channels compare learned stereo estimates after compensating
for measured body motion. Self channels use actual odometry and command receipts.

Provisional adaptation revises estimates immediately. Consolidation requires repeated
valid observations, low residual dispersion, and, for action-attributed object effects,
passive comparison observations. Missing/ambiguous sensing is unavailable evidence,
not a zero effect. Evidence from denied or mismatched commands cannot establish the
requested action's effect. Contradictions qualify the relationship and remain durable.
Scope is empirical and conditional; intervention comparisons do not prove universal
causality. Equivalent relationships reuse existing structures and operations.

Compile a supported estimate into an ordinary exact native Procedure. It checks its
observed conditions, consumes new measured state, and produces a predicted next state
and evidence status. Application outside scope returns unknown. Compiled procedures
use fitted coefficients; installing the procedure does not teach the relationship.

## Inquiry and experiments

Generate bounded candidate experiments from available motor interfaces, active
uncertainties, observed geometry, passive controls, capability reliability, resource
limits, and user focus. Supply their uncertainty, relevance, expected distinguishing
benefit, reliability, cost, and risk to the existing encoder, recurrent field, and
shared neural scorer. The model chooses a candidate or abstains. Exact constraints
can veto a proposal but cannot replace it with a solution. Candidate predictions and
the selected neural scores are inspectable. Train this shared scoring competence on
generic experimental priorities, without scene physics or the discovery answer.

Start/pause/resume/stop and action budgets are explicit. No action occurs while paused,
stopped, or exhausted. Directed inquiry supplies a focus, not an answer. Passive
observation and safe abstention are available. The self-map includes current uncertainty,
measured capability reliability, evidence gains and conflicts; these change future
experiment features and recurrent activations.

## Neural and knowledge consolidation

Fast hypothesis/map changes do not imply parameter changes. Periodic neural updates
reuse GroundedTrainer and its optimizer, supervised only by measured consequences and
bounded information benefit. Retention regularization/replay protects prior proposals;
actual shared parameter updates and temporary activation changes are reported separately.
Persist optimizer moments, RNG, counters, pending experience, inquiry controls, exact
native references, hypotheses, and scoring competence. Old outer formats 1/2 load with
discovery inactive; old inner formats remain readable.

## Activation and runtime

Replace full-store graph construction scans with an incremental structural index over
types, kinds, reference incidence, and operations. Explicit operation scopes allow
discovery to activate motor/relevant acquired procedures without every installed
operation. Keep default operation behavior compatible. Native revisions and provenance
remain durable even when inactive. Resource overflow still fails explicitly.

Extend the existing local viewer and headless runtime with inquiry controls, hypotheses,
predictions, experiment rationale, evidence, acquired procedures, self-map, neural
updates and activation. Inspection must not choose a new experiment or change state.

## Implementation and verification plan

1. Add incremental indexing and preserve default graph contracts.
2. Add native hypotheses, condition-scoped estimation, evidence updates and compilation.
3. Add neural experiment ranking and generic priority training with retention.
4. Integrate bounded runtime and automatic continual updates with outer checkpoint v3.
5. Extend application/viewer/headless interfaces and version to 0.5.0.
6. Run a continuous discovery session with untaught object response, passive controls,
   reuse in a changed arrangement, further learning, save/restore, and matching continuation.
7. Test conflict, missing evidence, indistinguishability, budgets, hidden-input isolation,
   native execution, optimizer continuation, selective activation and prior milestones.

No separate reasoner, language retrieval, scene-specific diagnostic sequence, new neural
module per capability, or hidden-world inference path is introduced. The initial
inductive family is intentionally bounded; unsupported laws remain unknown.
