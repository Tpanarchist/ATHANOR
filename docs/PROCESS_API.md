# Recursive processes: API and training (0.6.0)

`Runtime.start_process_discovery(budget=12, max_calls=2, interface="body")` constructs
bounded native call trees and starts existing neural-ranked motor inquiry. `camera`
excludes ambiguous/unavailable stereo correspondence. `numeric` accepts externally
measured exact pairs. `Processes.start(..., max_candidates=384)` supports 1–4 calls and
1–2048 candidates, with at most 64 times that many construction attempts. Exhaustion
is a resource limit, never an absence proof.

All expressions are ordinary Procedures. The initial interface is typed Data with a
number `value`; existing compatible arithmetic is reused or taught. The unit quantity
is established arithmetic, independent of environmental answers. `offer(operation_ref)`
accepts taught conditional, accumulated or other compatible 1–3 argument constituents.
Acquired processes automatically enter the same library by pinned reference. Wider
types require explicit adapters; unrestricted program generation is unsupported.

`observe(Fraction(before), Fraction(after), source=..., witness=...)` retains native
receipts, evaluates candidates through the kernel, and revises hypotheses. Three
distinct inputs are the minimum support threshold. All errors count. `rank()` uses the
existing encoder/scorer and recurrent context. `adopt()` chooses the neural-ranked
candidate, then evidence vetoes unsupported choices without substituting an answer.
Equivalent explanations remain separate provisional hypotheses.
The physical Runtime calls adoption after learning from each new process measurement;
the continuous acceptance does not choose or adopt a physical answer in its driver.

`choose_probe(tuple_of_fractions)` uses the same shared model to choose a new stimulus
in an advertised domain of at most 64 values. Candidate disagreement and cost influence
features. Selection precedes observation; no private territory law enters this selector.

`learn(owned_grounded_trainer, steps=40)` updates the existing shared scorer and optimizer
from measured error, complexity and reliability. Structural, semantic, current grounded
and inquiry scores constrain drift; parameter changes per step are bounded. Other model
parameters remain frozen during this narrow consolidation. Runtime training occurs on new
receipts and respects pause-learning. Retention is measured protection, not an unlimited
interference guarantee. No acquired-process parameters are added.

`apply(operation, value, expected=optional_measurement)` binds and executes an adopted
process. Numeric execution is exact; body evidence tolerance is 0.001 and camera tolerance
0.06. A fresh mismatch creates a contradiction and qualifies acquired dependents and
mappings recursively. `qualify(operation, reason, witness=...)` also permits explicitly
attributed invalidation. Old pinned revisions remain inspectable. Capability counts
distinguish recognition, binding, execution and transfer attempts/successes; installation
alone is not mastery.

`correspond(operation, [(before_record, after_record), ...], source=..., tolerance=0)`
enumerates typed numeric leaves through Records/sequences. Names have no semantic
privilege. The shared model ranks candidates; evidence requires three distinct inputs
with preserved predictions. Multiple fitting bindings remain uncertain until further
evidence distinguishes them. A mapping records learned selectors, typed interfaces,
pinned process, native input adapter/application/output binding, observed domain,
constant context guards, preserved quantity relation, packets and tolerance. Interface
installation is one kernel transaction; failure installs no partial binding while
retaining attributed proposal evidence.

`apply_mapping(mapping, packet, observed=optional_record)` runs the same process through
its new typed input binding. Wrong/missing types, changed context guards, invalidated
support or values outside the observed range withhold reuse. A one-pass correspondence
does not authorize a two-pass device. `materialize_mapping(mapping, packet, template)`
reconstructs the output interface through native Record/sequence expressions; the template
supplies unrelated channels and the selected output leaf is computed. This is a directed
quantity-preserving correspondence, not an asserted inverse of every process.

## Viewer and headless commands

`Application.command` and local `/command` expose:

| Command | Inputs | Result |
|---|---|---|
| `discover_processes` | budget/max_calls, interface | construction and inquiry |
| `process_observation` | exact before/after strings | attributed external measurement |
| `process_probe` | bounded exact value strings | model-selected stimulus |
| `train_processes` | none | 40 real shared updates |
| `adopt_process` | none | supported selected process or uncertainty |
| `correspond_process` | pinned reference, before/after packet pairs | binding or uncertainty |
| `apply_process` | pinned reference, exact value string | native result |
| `apply_mapping` | pinned mapping, packet, optional observed/template | native bound/materialized result |

JSON packet quantities use integers or `{"$number":"3/5"}`. Floats are rejected rather
than silently approximated. References use `namespace:s########@revision`. External
measurements are attributed as external. Pause/resume/stop and pause-learning controls
apply to both inquiry paths. State inspection displays actual model state without mutation.

```powershell
athanor-world --checkpoint artifacts\process-observer.athanor --paused
athanor-world --checkpoint artifacts\discovery-observer.athanor --headless --processes --steps 12
athanor-process-demo --base-checkpoint artifacts\discovery-observer.athanor --checkpoint artifacts\process-observer.athanor
```

Grounded checkpoint format 4 preserves candidates, library, dependencies, mappings,
support/contradictions/revisions, parameters, recurrent state, optimizer, resources,
probes, capability counts and retained score anchors. The next process-learning update
continues identically after restoration. Formats 1–3 remain readable with empty process inquiry.
The native store and inner observer formats are unchanged. Persisted execution requires
neither a teacher nor replay of original observations.
