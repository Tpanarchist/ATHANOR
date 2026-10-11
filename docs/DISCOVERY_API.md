# Autonomous discovery API — 0.5.0

Use Python 3.14 with the existing `neural` extra. Runtime uses no external model.

## Control and resources

```python
from athanor.grounded.checkpoint import restore, save

runtime, trainer = restore("artifacts/discovery-observer.athanor")
runtime.start_discovery(24)  # all motor interfaces
runtime.start_discovery(12, focus="object", allowed=("wait", "interact"))
step = runtime.autonomous_step()
runtime.observer.discovery.pause()
runtime.observer.discovery.resume()
runtime.observer.discovery.stop()
save(runtime, "artifacts/session.athanor")
```

Starting an inquiry creates a budget and changes relevance, retaining knowledge,
evidence, capabilities, parameters and activation. Focus is `object`, `self`, `interface`,
or None; it supplies no answer. Allowed actions belong to the existing motor interface.
Budgets range from 1 to 10,000. Autonomous calls act only when runtime/inquiry controls
permit it. Abstention pauses without advancing the world or consuming an action.
`Runtime.step(Action(...))` is explicitly external and labeled accordingly.

`Discovery.learning=False` pauses slow neural consolidation; inquiry pause stops the
autonomous action/adaptation loop. Every fourth autonomous action normally triggers a
measured parameter update. `neural_updates` is separate from native revisions/refinement.

## Native hypotheses and acquired procedures

`GroundedObserver.discovery.estimates` contains channel/action/scope, observed appearance
group, samples, input conditions, support, conflicts, excluded measurements, a pinned
native hypothesis, and optional relationship/procedure handles. Kernel inspection and
provenance apply to every immutable revision.

Alternatives include zero effects, fitted conditional constant effects and unresolved
interface/measurement explanations. Missing or ambiguous stereo, unmeasured view rotation,
mismatched command execution and stale-image/odometry discrepancy stay unavailable rather
than becoming zero object effects. Interface hypotheses compare independently measured
image change and body displacement; they do not receive hidden fault labels.

Object action consolidation requires three valid observations, low dispersion, and two
stable passive controls. Self/interface predictions require three valid observations.
Acceptance tolerances are 0.06 for camera effects and 0.015 for body/interface measurements;
these are engineering bands, not calibrated probabilities. Conflicts qualify current
reuse while retaining historical support and procedures. A conflicting passive baseline
also qualifies dependent acquired object predictions and prevents further consolidation.

```python
d = runtime.observer.discovery
effect = next(v for v in d.estimates.values() if v.channel == "object" and v.status == "supported")
result = d.predict(effect, (0.08, -0.05, 0.65), appearance=effect.appearance)
if result is not None:
    print(runtime.observer.knowledge.get(result).fields)
```

This executes an ordinary native Procedure with exact fitted coefficients. Its inputs
include action, scope, validity, appearance and numbered state values. Outputs contain
status, pinned hypothesis, numbered effects and next-state predictions. It checks scope,
appearance and observed camera input ranges with tolerance; outside them values are unknown.
The immutable procedure represents its compilation revision; `Discovery.predict` additionally
checks current evidence before reusing it. Inquiry itself executes these procedures and
links their results as `acquired_prediction` in experiments. New measured contexts raise
the benefit of checking the prediction. Logs/text retrieval cannot substitute for execution.

## Neural selection and selective activation

Native experiments carry operations, targets, alternatives, uncertainty, relevance,
benefit, measured reliability, cost and risk. Compact experimental features pass through
the existing structural encoder and shared scorer, conditioned by the recurrent context.
Names, IDs and territory properties are excluded from priority features. The model chooses
an experiment or abstains. Constraints veto but never choose replacements. Scores/confidence
are inspectable and do not represent calibrated truth or discovery probabilities.

`GraphBuilder.build(..., operations=(...))` declares an active procedure interface.
Default behavior retains all installed operation candidates. `KnowledgeIndex` incrementally
indexes roles, kinds, categories and incident claims through `Athanor.structures_since`
and `structure_count`. Discovery activates motors and linked acquired predictions.
Audit links are omitted from compact discovery activation views, remaining intact in exact
native structures. Index/storage and transactional costs still grow with durable history.
This is native executable activation, with no text retrieval service.

## Persistence and application

Outer format 3 retains controls, hypotheses, samples, exact handles, selections, policy
and counters. The sensory payload retains owned AdamW moments, RNG, pending pre-action
frame/prior, trainability and update count. Inner format 3 adds explicit operation scopes;
unscoped observers write format 2. Inner and outer formats 1/2 remain readable. World
format 1 restores the default response; format 2 saves configurable bounded territory
physics. That property never appears in SensorFrame or observer features.

Application commands add `explore`, `investigate`, `pause`, `resume`, `stop_inquiry`,
`pause_learning`, `resume_learning`, and `train_inquiry`. Existing teaching, instruction,
manual actions and save/restore remain available. `/state` exposes hypotheses, experiments,
evidence, acquired procedures, scores, activation, self-map and neural updates. Inspection
does not mutate inquiry or knowledge.

```powershell
.venv\Scripts\python.exe -m athanor.grounded.discovery_demo
.venv\Scripts\python.exe -m athanor.grounded.app --checkpoint artifacts\discovery-observer.athanor
.venv\Scripts\python.exe -m athanor.grounded.app --checkpoint artifacts\discovery-observer.athanor --headless --explore --focus self --steps 8
```

See [DISCOVERY_TRAINING](DISCOVERY_TRAINING.md) and [AUTONOMOUS_DISCOVERY](AUTONOMOUS_DISCOVERY.md).
