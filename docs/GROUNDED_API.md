# ATHANOR 0.4.0 — grounded runtime API

Python 3.14; `athanor.world` uses only the standard library and the native cornerstone.
`athanor.grounded` requires the optional `neural` extra. Inference is local.

## Run the application

From `E:\ATHANOR`:

```powershell
uv sync --locked --extra dev --extra neural --python 3.14 --cache-dir .tools\uv-cache
.venv\Scripts\athanor-grounded-demo.exe --base-checkpoint artifacts\meaning-observer.athanor --save artifacts\grounded-observer.athanor
.venv\Scripts\athanor-world.exe --checkpoint artifacts\grounded-observer.athanor
```

Open `http://127.0.0.1:8765`. The loaded world initially pauses. Start/Pause/Stop
control actual simulation advancement; Advance one step advances one 0.1-second
tick. Think runs the same recurrent observer again without advancing territory.
The exterior projection, both RGB images, working identity/depth hypotheses,
learned proposal, actual consequences and active self-map all read the live runtime.
External motion controls are explicitly labeled; they are not learned decisions.

Teaching names the indicated visible percept index (zero based). Indices describe
current image regions, not authoritative object IDs. Teaching requires resolved
stereo evidence. Appearance membership and persistent identity remain hypotheses.
Instruction grammar is `approach <taught expression>`, `observe`, `search`, or one
of `forward`, `backward`, `left`, `right`, `turn_left`, `turn_right`, `look_up`,
`look_down`, `interact`, `wait`. Movement selection is a neural proposal, including
for a direct bounded instruction. Unknown or multiply bound target names veto
motor execution. Search is an acquired bounded exploration convention, not a
general path planner. No natural-language scene oracle is installed.

Save/Restore uses the chosen checkpoint path. Learn from last consequence updates
actual parameters using the saved pre-action frame and state, measured outcome and
AdamW; it does not mark a sensory hypothesis true. Training is explicit. If an
instruction changes after an action, its new frame cannot replace that action's
original learning context. This includes the activation that entered the decision,
rather than its output activation. The pending experience also survives a restart.

```powershell
.venv\Scripts\athanor-world.exe --checkpoint artifacts\grounded-observer.athanor --headless --steps 10
.venv\Scripts\athanor-grounded-demo.exe --resume artifacts\grounded-observer.athanor --save artifacts\grounded-resumed.athanor
```

The headless path uses exactly the same `Runtime.step`. It prints actual state and
does not implicitly overwrite the checkpoint. Resume performs a further explicit
supervised update and saves the optimizer/shuffle state. These are controlled
updates; retention under unlimited online learning is not established.

## Territory and sensor contracts

```python
from athanor.world import Action, World

world = World(namespace="territory", resolution=24)
participant = world.add_object((0.6, 0, 3), color=(225, 45, 35), movable=True)
event = world.step(Action("forward"))
frame = world.sense()
world = World.loads(world.dumps())
```

The world holds a separate native `Athanor` store. Its body and rigid objects have
stable local identity and pinned revisions. `world.knowledge.inspect(event)` and
`provenance(world.body)` expose actual production. `relocate` is an authority/editor
API; it is not available on `GroundedObserver`. Coordinates are bounded by
half-extents `(6, 3, 6)`. Sphere `size` repeats its radius; box `size` holds three
half-extents. Prescribed velocity is bounded by two units/second. Objects may be
explicitly installed in overlapping arrangements; the initialization API does not
claim a solved physical scene.

`SensorFrame` contains tick, dimensions, immutable left/right RGB bytes, focal
length, baseline, executed interface command and six proprioceptive values:
local displacement `(dx, dy, dz)`, yaw change, contact indicator and measured
body-movement indicator. It contains no identity handles, class labels, segmentation,
depth image, absolute position, object velocity, hidden fault label or native world
reference. `render(labels=True)` is a separate training/authority API and is never
called by observer inference. Repeated internal refinement consumes the same frame.

Raw ray/sphere and slab/ray/box intersections provide nearest-hit occlusion. Body
motion uses ten 25-mm substeps per action. Static sphere/sphere, sphere/axis-aligned
box and box/box overlap are analytic. Object velocity and interaction pushes use
discrete proposed-state collision tests; very thin geometry can require finer
integration. Rendering has flat materials, no learned lighting or texture. No gravity,
inertia, angular object dynamics or full rigid-body solver. Body translation is
planar; camera look angles reveal three-dimensional positions. Quantized native
coordinates and fixed step/order make same-platform continuation deterministic.

## Grounded observation and execution

```python
from athanor.grounded.checkpoint import restore, save

runtime, trainer = restore("artifacts/grounded-observer.athanor")
observer = runtime.observer
observer.teach_visible("marker", index=0)
observer.instruct("approach marker")
observer.observe(runtime.world.sense())  # coordinator supplies the restricted frame
step = runtime.step()  # actual neural proposal, kernel, world, feedback
print(step["intended"], step["executed"], step["prediction"], step["observed"])
save(runtime, "artifacts/my-grounded.athanor", trainer=trainer)
```

`GroundedObserver` has no territory member. `PixelObserver` extends the existing
`Observer` and uses its graph builder, structural encoder, relational messages,
GRU, operation pointers, activation history and self-map. A shared RGB CNN learns
foreground and continuous appearance. Exact connected components over learned
foreground/appearance split distinct touching surfaces; reciprocal stereo matching
and calibrated triangulation estimate relative position. Border clipping, multiple
matches and weak/invalid disparity preserve partial/unknown states. Confidence is
uncalibrated; centimeter precision must not be inferred from floating coordinates.

Map identities are created by the observer, and associated conservatively through
learned appearance. Indistinguishable duplicates cannot be uniquely identified.
Hidden tracks retain their last supported identity/odometry-adjusted estimate with
decaying confidence. This is an explicit persistence hypothesis, not proof of current
existence. Motion evidence compares camera estimates against measured own motion;
measurement error, camera movement and independent object movement remain competing
explanations. A small active relational window and bounded self-map influence the
same recurrent field; durable history is not all placed in the active graph.

Motor procedures are native taught exact command constructors. Only the authorized
motor port can affect territory. Invalid arguments are rejected atomically and
recorded as self evidence before any world effect. Physical success is measured
afterward, separately from a valid native command. The taught `within reach` procedure
uses estimated map coordinates, and rejects unresolved position/status instead of
turning missing depth into zero-distance certainty. An exact result about an estimate
is not a true assertion about territory.

Feedback retains intended/executed command, six predicted/measured consequences,
prediction discrepancy, physical capability success, optional target-progress
preference, world/self/interface alternatives and evidence. No movement preserves
multiple causes. Contact and odometry add support without secretly consulting motor
or camera fault settings. Consolidated action counts, visual-change means and empirical
success rates are native knowledge qualified by observed experience.

## Training and persistence

`visual_lessons` creates independent raw scenes, with privileged masks used only for
supervised training. `train_vision` learns shared foreground/continuous appearance.
`embodied_lessons` varies actual camera views, object geometry, clutter, taught names,
library size, faults, independent motion and prior interaction histories. Labels
use resolved learned stereo measurements for the approach bearing; teacher authority
can establish visibility but does not replace the observer's measured inputs.
`continuous_lessons` adds real eight-step episodes with teaching, exploration,
changing intent and hidden actuator faults. Proposals never receive scene labels
or authoritative coordinates at inference.

`GroundedTrainer.train` fits neural operation pointers and measured physical
consequences. Priors are regenerated with current parameters using up to four
previous observation frames; their activations are detached at the current training
boundary. Structural/semantic replay calls the existing native trainer with its
complete ordered-role, applicability, meaning and three feedback losses, carried
state, the same network, optimizer and owned shuffle RNG. Each representation family
is sampled separately so paired language examples cannot drown structured use; both
have acceptance gates.
Visual layers are frozen during action training; further explicit
visual teaching can unfreeze them using `train_vision`. Sensory fusion includes a
compact intention view encoded by the **existing shared structural encoder**, so
receipt/track metadata does not bury the current observation's intent. The grounded
query also uses this compact interface, while full receipt and track graphs remain
available through relational messages. The decoder retains current query evidence
alongside recurrent refinement. The original
no-sensory model path and all existing parameter names remain unchanged.

`learn_transition(before, after, action, frame=..., prior=..., preference=...)`
learns a measured outcome without simulator labels. Optional bounded preference
applies a small consequential policy-gradient term; prediction error and capability
evidence remain separate. The acceptance uses measured target progress, not an
invented reward. This is a first controlled feedback update, not a robust general
reinforcement-learning algorithm.

An atomic outer ZIP contains native world, original format-2 observer checkpoint,
sensory tensors, runtime map/frames, self/history/consolidations, optional optimizer,
trainability flags, local shuffle RNG, inference input activation and the pending
training transition. Grounded archive version 2 adds exact decision-input preservation;
version 1 remains readable with its older output-state learning convention. Digests,
native validation, tensor checks, exact references and member budgets reject malformed
input. `torch.load(weights_only=True)` avoids executable persisted classes. Digests
detect corruption, not instructor authenticity or historical truth. Old native and
Goal 002/003 neural checkpoint formats remain independently readable. The new outer
format is for grounded runtimes, not a replacement for those formats.

`Application` serializes all runtime operations with one lock. Direct `World` and
`Runtime` calls require a single coordinator. The local HTTP viewer is intended for
localhost; it is not an authenticated remote-control service. Active graph budgets
fail explicitly on overflow; native histories/index copying and Python rendering
remain scalability limits. See GROUNDED_OBSERVER, DECISIONS and VALIDATION for the
architectural review, measured results and remaining work.
