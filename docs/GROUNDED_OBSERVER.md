# Goal 004 integration design — territory, interface and map

Design/critical review recorded before implementation, 2026-10-10.

## Contracts

Territory is a separate native Athanor store: bounded 3D geometry, fixed time steps,
spherical body, dynamic objects and causal event receipts. Its identities and positions
never enter inference. A coordinator owns territory; the observer accepts only a
SensorFrame containing two RGB byte arrays, camera calibration and limited proprioception
(elapsed time, relative body displacement/rotation, contact and actuator receipt).
The outside viewer may inspect territory; its data path is distinct from the sensor path.
Camera faults are represented at the interface, not secretly labeled in camera images.

Map is another native store. Learned foreground segmentation provides image regions;
learned appearance features support conservative temporal/stereo correspondence.
Known projective geometry estimates relative 3D positions from camera measurements.
Tracks are observer-created hypotheses, never simulation IDs. Occlusion preserves a
last-supported hypothesis with increasing uncertainty, not a claim of current visibility.
Competing matches remain ambiguous. Exact physics and rendering do not infer beliefs.

An additional shared convolutional sensory encoder is justified by spatial RGB inputs.
It is not a second recurrent reasoner. Its pooled activation is injected into the existing
structural model's query before the same message/GRU refinement and operation pointers.
Fusion also receives a compact intention interface view through the same structural
encoder. Receipt/track metadata remains in the relational graph but does not dilute
the intent encoding. Current query evidence is retained alongside recurrent refinement
in the grounded decoder. No additional intent or policy network is created.
No sensory input leaves old inference unchanged, including parameter/checkpoint names.
Physical prediction is a readout of that shared field plus intended action, separate
from the existing structural-result prediction. A shared appearance descriptor handles
new instances; taught visual prototypes do not allocate concept-specific networks.

Installed exact action procedures produce validated command structures. The world port
validates and executes those commands separately, records actual consequences and returns
another sensor frame. Kernel execution success and physical action success are distinct.
Learned action selection has no access to collision tests, world IDs or authoritative
coordinates. Curriculum preparation may use them for labels; this is a separate module.

Teaching a name records a concept plus attributed perceptual examples. Installing it
does not establish recognition competence. General relation procedures use measured
map structures; physical recognition and applicability require learned perception.
An unfamiliar causal relationship remains an outcome hypothesis supported by experience,
not a named rule silently installed as truth. Predictive, preference and capability
discrepancies remain separate. Failed visual predictions preserve world/self/interface
explanations until contact, odometry or further views provide discriminating evidence.

## Exact and approximate computation

Use analytic ray/sphere and ray/box intersections with a z-buffer equivalent (nearest
ray hit), calibrated stereo and deterministic fixed-step collision/integration. Simple
rigid objects have prescribed velocity; interaction applies a bounded push. No gravity,
rotational inertia, deformable bodies or photorealistic illumination. Floating projection
and trigonometry are quantized before native state publication; reproducibility is within
the same implementation/platform, not a cross-hardware physics proof.

Segmentation, appearance, contextual action selection and consequence prediction are
trained. Connected components, calibrated triangulation, conservative hypothesis
association and evidence bookkeeping are exact algorithms over learned measurements.
Component growth also checks learned appearance to distinguish touching surfaces.
Neither region labels nor authoritative object coordinates are supplied at inference.

## Persistence and application

Separate world knowledge, observer knowledge, shared parameters, sensory parameters,
working tracks/frames, optimizer/RNG and bounded active self-map. Durable events are
inspectable but not all active. A versioned atomic outer checkpoint embeds the existing
observer checkpoint and native world snapshot, plus validated sensory/runtime data.
The pending pre-action frame/state and trainability flags also restore, so further
optimization uses the original experience, not the most recently viewed instruction.
Old Goal 001–003 formats stay readable. Headless and interactive paths use the same
runtime. A local standard-library HTTP viewer renders actual cameras, exterior state,
predictions and histories; it can pause/step, teach, instruct, save/restore and train.

## Critical review and decisions (ADR-014)

Chosen over a full game engine: compact analytic 3D simulation with native receipts,
because the requirement is observable learning, not graphics complexity. Chosen over
privileged object observations: raw stereo RGB plus learned segmentation, with an
inference boundary test. Chosen over a separate world/self network: sensory input and
readouts on the existing shared recurrent field. Chosen over end-to-end learned physics:
exact world execution and learned consequences. Chosen over unconditional identity
matching: conservative hypotheses and explicit ambiguity, because identical objects
cannot always be distinguished from images. Chosen over a new UI dependency: local
browser canvas and standard-library server, preserving Python 3.14 compatibility.

Unresolved before scale: calibrated confidence, indistinguishable-object identity,
larger-scene association, authenticated real sensors, distribution shift, continual
learning interference and cross-platform replay. Acceptance must report these limits
and actual before/after training changes rather than claim universal perception.

## Implemented review outcome

The world and map use pinned revisions, native procedures and production receipts.
Old foundation/meaning contracts and the no-sensory neural path are preserved.
Unknown depth cannot justify an exact nearness result. Hidden tracks are actively
available as uncertain hypotheses; repeated views/internal steps do not apply the
same odometry twice. Movement residuals compensate measured translation/yaw while
retaining measurement error as an alternative.

Training explicitly separates perception installation, appearance/name teaching,
neural selection competence, physical prediction and measured capability history.
Current-parameter carried-state training includes real continuous episodes with up
to four preceding frames. Current intent has a compact structural interface while
map/receipt graphs remain available for relational messages. Replay reuses the
existing trainer's complete lesson and feedback contract with one model/optimizer.
These choices address history/interference failures caught during implementation.
They do not establish
unrestricted continual learning. The image adapter is frozen during action training;
explicit further visual lessons can unfreeze and adapt it.

The runtime has a local browser viewer and the same headless port, with explicit
training controls. The continuous acceptance passes through actual learned proposals,
real faults and sensor feedback. Focused tests cover the privileged-state boundary,
safe rejection, stereo uncertainty, real parameter updates and identical resumed
training. See GROUNDED_API for exact operational contracts and VALIDATION for results.
