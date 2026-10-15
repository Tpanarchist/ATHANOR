# ATHANOR Goals 001–005 — verified implementation

Date: 2026-10-11. Windows, CPython 3.14.6, PyTorch 2.14.1+cpu.
The core and world remain independently usable without PyTorch.

## Goal 005 autonomous discovery and continual learning

Version 0.5.0 reuses the structural encoder, recurrent field and shared scorer for
experiment selection. No neural parameter module was added for inquiry. The public
acceptance run restores the Goal 004 observer and optimizer, teaches general inquiry
priorities, then enters a territory whose interaction response was not supplied to
the observer. Only restricted camera/body consequences inform acquisition.

The contact investigation permits `interact` and `wait`; the model selects their
order, including passive controls. Continued capability inquiry permits all ten
existing motor actions. These bounds are explicit experiment constraints. There is
no prescribed action sequence or controller supplying the environmental answer.

Measured results in `artifacts/discovery-observer.report.json`:

| Measurement | Result |
| --- | --- |
| Untaught territory response, held only by world authority | 0.12 |
| Learned camera-derived effect after three interactions and passive controls | (0, 0.00049024, 0.12292990) |
| Dispersion of initial effect samples | 0.01428286 |
| Acquired native knowledge | Conditional executable procedure and relationship, with hypothesis/evidence references |
| Prediction error in changed arrangement | 0.01259135 camera-coordinate units |
| Automatic acquired-procedure reuse | Native predictions attached to new experiment candidates |
| Continued neural consolidation | Four updates; shared encoder/message/recurrent/scorer parameters changed |
| Held-out structural proposals and abstention, 48 mixed cases | 100% before inquiry teaching, after teaching and after continued learning |
| Held-out semantic proposals, meanings and ordered roles, 48 mixed cases | 100% at all three stages |
| Checkpoint continuation | Next four decisions, native knowledge, world state, parameters and optimizer continuation match |
| Generic synthetic inquiry priority accuracy | 62.5% after 1,600 teaching updates |

Structural exact-result/executable rates are 19/48 overall because the set includes
29 intentional abstention cases; complete expected proposals are 48/48. Priority
accuracy is a separate measurement, not a calibrated uncertainty or general planning
guarantee. The acquired operation predicts a bounded conditional constant effect,
requires matching appearance and measured input bounds, and returns unknown outside
its scope. Contradictions qualify the hypothesis and withhold current reuse while
preserving historical procedures and evidence.

Eighteen focused discovery tests exercise provisional evidence, passive comparisons,
conflicts, unavailable sensing, native execution, budgets, neural abstention, identical
observations with different hidden physics, action-name-independent priority encoding,
capability feedback, pure viewer inspection, invalid rehashed checkpoints and indexed
activation. The acceptance integration test includes transfer, retained competence,
shared parameter updates and restore across an optimizer boundary. Existing grounded
contracts also pass all 20 tests. The 0.5.0 wheel and source distribution build
successfully; the viewer and headless coordinator use the same runtime.

After the final interface-focus and passive-conflict fixes, all 164 checks, including
the discovery integration acceptance, pass in 19.814 seconds. The repository contains
167 named tests, including three longer prerequisite learning acceptances. Ruff
lint/format, strict mypy (40 source files) and Git whitespace checks pass. A separate
installed-wheel environment restores acquired knowledge, executes an autonomous action,
saves/restores again and matches the next decision. Every packaged Python/viewer source
matches the workspace, and the source archive includes the new documentation/tests.
Without PyTorch, the installed wheel passes all 93 foundation/world tests and explicitly
skips the 74 optional neural tests.

The complete regression run passed all 165 tests present when it started in
1,693.400 seconds, including fresh Goal 002, 003 and 004 learning acceptances.
The final 164-check run includes the two subsequently added interface-focus and
passive-conflict regressions. Their combined coverage is all 167 current named tests;
`artifacts/goal005-coverage.json` records the explicit coverage sets. No prerequisite
training acceptance is omitted from this coverage.

Reproduce the integrated run:

```powershell
.venv\Scripts\python.exe -m athanor.grounded.discovery_demo
.venv\Scripts\python.exe -m athanor.grounded.app --checkpoint artifacts\discovery-observer.athanor --paused
.venv\Scripts\python.exe -m athanor.grounded.app --checkpoint artifacts\discovery-observer.athanor --headless --steps 5 --explore --focus interface
```

See `DISCOVERY_API.md`, `DISCOVERY_TRAINING.md` and `AUTONOMOUS_DISCOVERY.md` for the
runtime controls, evidence contract and induction limits. This run establishes bounded
functional acquisition and continuation. It does not establish arbitrary nonlinear
law discovery, universal causal explanations or indefinite retention.

## Goal 004 grounded learning and operational evidence

Version 0.4.0 extends the existing encoder/message/GRU/scorer with a shared two-eye
RGB adapter and physical-consequence readout. The existing recurrent model still
has 107,980 parameters; the sensory extension has
15,002. These dimensions follow the existing 64-dimensional
field and fixed embodied interface, not a parameter ceiling or new network per
concept. The shared model's parameter change from Goal 003 has L2 norm
40.1898.

The public training run restores the original Goal 003 checkpoint. It trains visual
foreground and continuous appearance on 240 randomly prepared scenes for 80 passes,
then action selection and six measured consequences on 400 isolated situations plus
128 valid continuous episode frames for 80 passes.
The 160-frame episode budget skips examples lacking resolved stereo teaching
evidence rather than injecting an authoritative object label. A separate 40-scene perception set and
100-situation operation set use different seeds. Appearance/name teaching, variables,
namespace/allocation, geometry, clutter, library size, actual motion and hidden faults
vary. Labels use teacher authority only during preparation; the ordinary observer
receives raw RGB and limited proprioception, never IDs, hidden coordinates, object
classes, segmentation masks or fault labels.

| Measurement | Before learning | After learning |
| --- | --- | --- |
| Foreground intersection-over-union on 40 held-out scenes | 0.00000 | 0.98543 |
| Normalized continuous appearance mean absolute error | 0.36438 | 0.05904 |
| Raw operation proposal on 100 held-out situations | 0% | 97% |
| Six-channel physical consequence mean squared error | 0.234588 | 0.014434 |
| Complete retained structural proposals, 128 Goal 002 cases | Prior Goal 003 model: 128/128 | 128/128 |
| Complete retained meaning/ordered-role proposals, 100 Goal 003 cases | 99/100 | 100/100 |
| Additional 40-case meaning/ordered-role check | Not measured here | 39/40 |

These are bounded correctness and transfer checks, not a comparative campaign or an
open-domain claim. Operation results precede execution/evidence vetoes. Remaining
proposal errors are real errors; deterministic kernel validation alone cannot prove
that the neural interpretation was appropriate.
The additional 40-case check retains one incorrect ordered mathematical proposal;
the perfect 100-case retention set does not imply general error-free interpretation.

The same original trainer supplies structural and semantic replay, including ordered
arguments, applicability, meaning, carried activations, and distinct prediction,
preference and capability losses. One shared model, AdamW optimizer and owned shuffle
RNG serve grounded and native learning. Continuous lessons regenerate up to four
preceding activations using current parameters and detach them at the current update
boundary. Sensory queries encode compact current intent; full map/self structures
remain available for relational messages. The visual layers are frozen during action
training and can be adapted through further explicit visual lessons.

The continuous runtime scenario performs actual learned proposals, not manual motor
corrections or a scene-specific fallback:

- Raw stereo distinguishes unfamiliar visible participants. A teacher points to an
  observed region and names its appearance kind `beacon`; no parameters change.
- The acquired concept activates the learned approach policy, reducing actual target
  distance by 0.4937 world units in three steps.
- The installed exact `within reach` procedure executes on learned map measurements.
  Its Boolean result concerns those estimates; unresolved depth cannot justify it.
- An independently moving object produces map/odometry residuals. Measurement error
  remains an alternative to independent motion.
- An inserted occluder leaves 1 uncertain hidden identity hypothesis.
  The learned bounded search convention changes viewpoint, exposes camera evidence,
  and reuses a supported identity hypothesis after reappearance.
- A hidden actuator failure yields actual zero movement, a failed physical capability
  record, and competing obstruction, actuator and observation-interface explanations.
  The fault setting is not supplied to the observer.
- The taught kind transfers to a different valid arrangement, reducing actual target
  distance by 1.2137 units, including learned forward actions.
- Measured prediction discrepancy and a separate observed-depth progress preference
  (0.443409) change neural parameters through an actual experience
  update. This does not establish an inferred cause as true.
- Territory and acquired-map snapshots restore exactly. The next recurrent proposal
  and the next actual physical step match after restore. No territory references enter
  observer knowledge. The teacher is absent from inference.

The public checkpoint has 5,281 optimizer updates and occupies
1,483,991 bytes. Its runtime preserves installed knowledge, parameters,
activation field, input activation used for the pending decision, raw sensory frames,
working tracks, feedback history, optimizer and RNG. Replaying a saved experience
reproduces the original neural logits; further training resumes bitwise in the focused
optimizer test. Archive version 2 adds exact inference-input preservation and reads
the earlier Goal 004 format 1. Goal 001–003 checkpoint formats remain readable.

Goal 004 validation covered **all 149 named tests at that milestone**: a regression run passed 146 tests,
including both original full learning acceptances; after final contract changes a
fresh run passed all 146 current non-training contracts; the fresh complete Goal 004
training acceptance passed in 1171.870 seconds. Combined named-test coverage is
checked explicitly. All original 116 tests remain. Ruff lint/format, strict mypy and
Git whitespace checks pass.

Independent installed-wheel checks on Python 3.14 pass 93 native tests without
PyTorch; 56 optional neural tests are explicitly skipped. The installed neural wheel
passes all 20 new grounded contracts, restores the earlier neural/semantic checkpoints,
executes the acquired `gap` procedure to 22, and advances the same headless runtime.
Wheel source/viewer bytes match the workspace; source archives include documentation,
tests and viewer HTML and exclude generated artifacts/caches/environments.

The interactive viewer was exercised through its actual controls: Think leaves the
world clock unchanged; learned stepping advances it; learning advances the optimizer;
Save/Restore recovers exact clock and learned state after external motion; camera
teaching installs a new named concept and activates a learned proposal. Start/Pause/
Stop affect actual simulation advancement. External controls are explicitly labeled.
The viewer displays authoritative exterior state separately from the two actual camera
images, map hypotheses, predictions/consequences and self-map. Headless execution uses
the identical runtime. See `artifacts/grounded-viewer.jpg` for the live application.

Evidence: `artifacts/grounded-observer.report.json`,
`artifacts/grounded-retention.report.json`, `artifacts/grounded-observer.athanor`, its
`-parameters.pt` training state and `-continued.athanor` continuation. These are local
generated evidence, intentionally excluded from package distributions and Git.

Limits: flat colors and simple spheres/boxes; 24-pixel stereo in the exercised model;
coarse and sometimes fragmented/truncated perception; appearance-based identity
hypotheses rather than general object identity; uncalibrated confidence; explicitly
taught persistence and calibrated geometry rather than a newly discovered theory of
object permanence; bounded motor goals/search rather than general planning; empirical
associations and competing causes rather than universal causal inference; planar body
translation, prescribed velocity and discrete object collision rather than full
rigid-body dynamics; controlled replay rather than unrestricted continual learning.
The grounded map has its own native knowledge scope; original symbolic libraries
remain separately usable rather than being silently unified across namespaces.

The next work should resolve calibration/abstention, indistinguishable-object tracking,
longer-history retention, import/unification and scalable active-graph/history costs
before broadening the world. No general-intelligence, consciousness, subjective
experience, real-camera transfer or cross-platform physics-replay result is inferred.

## Goal 003 meaning, teaching and grounding evidence

The 0.3.0 implementation extends the existing shared encoder/message/GRU/scorer.
It retains 107,980 parameters: concept scoring, new words, new procedures and new
specializations allocate no additional neural parameter sets. The verified standalone
run restored Goal 002's 640-update checkpoint, then trained on 160 two-family native
teaching cases (320 fresh/active lessons) plus 80 structural replay lessons for 40
passes: 1,000 additional optimizer updates. Parameter change from the prerequisite
has L2 norm 20.9873. The run took 356.54 seconds on the exercised CPU machine.

| Held-out learned measurement | Before semantic training | After semantic training |
| --- | --- | --- |
| Operation/abstention pointer (100 cases) | 13/100 | 100/100 |
| Concept meaning pointer (including null) | 66/100 | 100/100 |
| Complete operation and ordered input proposal | 13/100 | 99/100 |
| Insufficient/unknown/ambiguous abstention | 10/30 | 30/30 |
| Ordered noncommutative proposals | 3/30 | 29/30 |
| Actionable mathematical proposals | 3/54 | 53/54 |
| Actionable categorical proposals | 0/16 | 16/16 |
| Retained Goal 002 structural proposals | Original milestone: 122/128 | 128/128 |

These measurements precede semantic evidence vetoes. One valid held-out noncommutative
proposal still binds an input incorrectly. Kernel admissibility does not establish
correct interpretation. Values, allocation/operation/candidate order, namespaces,
aliases, category depth/field shapes and histories vary independently; no identities
are encoded. These are bounded correctness/transfer checks, not an open-domain or
comparative benchmark claim. Retention includes all 48 expected abstentions and all
80 actionable exact results; the reported execution fraction 80/128 includes abstentions.

The integrated episode uses actual neural proposals and exact kernel execution:

- `subtract 3 from 17` produces 14 with reversed textual operands correctly bound.
- `does bird include robin` produces true; `is turtle a bird` produces false.
- Unknown `gap` abstains. A validated offline lesson then installs separation using
  the existing difference procedure, checked in both input orders.
- `gap 19 and 7` produces 12; its synonym `separation 5 and 24` produces 19.
  `add it and 2` uses that resulting structure and produces 21.
- A newly taught Finch specialization and observed wren reuse the general membership
  procedure: membership in Animal and Finch both produce true.
- `balance` retains two meanings and abstains without context. Supplied difference
  context changes activation; one further internal refinement accepts the correct
  learned proposal. No answer/operation is supplied by refinement.
- `delta` first uses its taught sum convention (26). An explicit correction preserving
  the prior binding changes its subsequent meaning to ordered difference (12).
- Conflicting interpretation evidence blocks execution. Teacher assertions about
  universal bird flight remain unknown. A separate local observation supports only
  the observed participant's flight, without generalizing it to the category.
- A two-clause composition produces 14 then 16 through result reuse. All new teaching,
  synonyms, correction and specialization use exactly unchanged weights.
- Save/restore preserves knowledge bytes, meaning interpretation, field/context and
  the same next recurrent computation. A restored observer independently uses the
  acquired `gap` concept on 31 and 9, producing 22. Continued supervised training
  changes restored parameters and advances the optimizer to 1,642 updates.

The default no-base demonstration reproduces Goal 002 prerequisite training from
scratch before extending it. Its complete acceptance test also passed. The full
regression run passed 114 tests, including both complete training demonstrations and
all original 82 tests, in 534.317 seconds. Two final native validation tests were then
added for type-aware context and exact referent/role contracts. A fresh 114-test run
of every current non-training contract passed; combined named test evidence covers
**all 116 current tests**, including the two full learning acceptances. Ruff check,
format check and strict mypy pass. These details distinguish fresh evidence from
an inferred total; no training acceptance was substituted by an artifact-only test.

Installed-wheel checks exercise native teaching, persistence, exact execution and
actual offline lesson CLI import in an environment without PyTorch. The optional
neural wheel restores both the original format-1 observer and the new semantic
checkpoint and executes acquired knowledge locally. The source archive includes
updated documentation/tests and excludes caches, environments and generated artifacts.

Generated evidence: `artifacts/meaning-observer.report.json`, trained checkpoint,
its `-trained` parameter/optimizer checkpoint, its `-continued` checkpoint, native
`.knowledge.json`, the prepared `.lesson.json` and `.before-teaching.json` input
snapshot. These assets are intentionally ignored by Git and excluded from distributions.

| Goal 003 requirement | Authoritative evidence |
| --- | --- |
| Resolve meaning architecture before modification | MEANING design/critical review; ADR-011 through ADR-013 |
| Symbol/referent identity, synonyms and contextual alternatives | Lexicon/native binding tests; learned alias/context cases and live correction |
| Semantic activation and ordered roles | Same model/scorer; raw held-out pointer measurements; reversed subtraction and categorical role binding |
| Usable structured teaching and specialization | Teacher/Example/native kernel contracts; new executable separation and Finch/wren episode |
| Example validation and safe failure | Exact example checks; failed and late-failing packs preserve snapshots, IDs and receipts |
| Unknown/conflicting interpretations | Explicit evidence vetoes; ambiguous meanings remain visible; late conflict/supersession blocks pending interpretation |
| Observation grounding and empirical uncertainty | Attributed scoped witness/support records; unverified assertions stay unknown; importer cannot mint observations |
| Optional external-AI preparation | Strict offline JSON translation and actual base-wheel CLI import; no Python source or live teacher calls |
| Self-map distinction and feedback | Installation/recognition records versus real attempts/results; active shared self-map; three discrepancy channels |
| Persistence and continued learning | Exact native/semantic checkpoints, next-step equality, optimizer/RNG equality test and changed continued weights |
| Preserve Goals 001/002 | Original 82 tests pass, original checkpoint restores, structural competence retention measured |
| Independent foundation and package | Zero runtime dependencies for native foundation/teaching; isolated wheel checks on Python 3.14 |

Limits remain: restricted grammar, explicit structured definitions rather than free
prose compilation, attributed rather than authenticated sensory evidence, uncalibrated
confidence, bounded active history/graph, no unrestricted continual learning, and no
consciousness or general-intelligence claim. Larger knowledge needs indexed activation,
retention and conflict/import policies; broader language needs diverse teaching and
meaning/role evaluation rather than merely more parameters.

## Goal 002 learning and integration evidence

The neural implementation uses PyTorch 2.14.1 on the exercised CPU path with Python
3.14.6. The kernel implementation and its original 61 tests are preserved; its only
source change at that milestone was the package version. Goal 003 subsequently adds
the documented transaction/category-reflection extensions. The neural dependency is optional. A separate
base-wheel environment has no PyTorch and imports/runs the cornerstone independently.

The default runnable learning demonstration trains from scratch on 256 structurally
taught lessons for 40 epochs (640 optimizer updates), using width 64 and four recurrent
refinement steps. Actual parameter change has L2 norm 17.5171. The final separate
training losses are selection 0.0000811, arguments 0.0029533, applicability 0.0001302,
capability 0.0268479, preference 0.0112084 and structural prediction 0.0001241.
These losses do not substitute for execution/proposal evidence.

| Held-out measurement | Before training | After training |
| --- | --- | --- |
| Correct operation, including abstention (128 cases) | 17/128 (13.28%) | 128/128 (100%) |
| Exactly correct operation and ordered arguments, including abstention | 0/128 | 122/128 (95.31%) |
| Abstention for unsupported/missing/incompatible input (48 cases) | 0/48 | 48/48 |
| Exactly correct proposals among actionable requests (80 cases) | 0/80 | 74/80 (92.5%) |
| Expected exact numeric result among actionable requests | 1/80 | 75/80 (93.75%) |
| Separate new-definition set (48 cases, including multiplication) | Not a pretraining target | 48/48 exact proposals/results |

Six held-out proposals still bind an argument incorrectly. One nevertheless produces
the expected value under a commutative procedure, which is why numeric correctness is
not treated as proof of correct binding. Evaluation changes values, namespaces,
allocation order, candidate order, specializations and nested compositions. It is a
bounded structural curriculum, not a comparative benchmark or open-domain claim.

The end-to-end runtime episode begins with a learned proposal. The execution succeeds
but changed preference is explicitly unsatisfied. Recorded corrective feedback changes
the next neural proposal; exact execution yields -15/2. Clearing the active self-map
and carried field restores the original instruction's choice. Multiplication taught
after training is selected/bound by the same model, produces -14, and adds zero neural
parameters. Further internal steps change activation without a new external input.

Combined checkpoint restoration preserves knowledge bytes, field/context tensors,
the pending decision and the next recurrent computation. Continued optimization changes
restored weights. The separate optimizer/shuffle test checks bitwise equality of the
next updates after restoring AdamW and the local shuffle RNG.

Generated evidence is `artifacts/neural-observer.report.json`, the trained checkpoint
`artifacts/neural-observer.athanor` and its `-continued.athanor` continuation. The reported
complete run took approximately 95 seconds on the development machine. Generated files
are local artifacts, not installed fixtures.

Final verification of version 0.2.0:

- **82 tests passed**, including the original 61 cornerstone tests, 20 focused neural
  contracts and one complete learning acceptance test (101.2 seconds).
- Ruff lint/format checks and strict mypy passed for all 18 package source modules.
- Locked `[dev,neural]` installation succeeded offline from the resolved cache.
- Built `athanor-0.2.0-py3-none-any.whl` and `athanor-0.2.0.tar.gz`; the source archive
  includes neural documentation/tests and excludes caches/generated artifacts.
- In a separate base-wheel environment, the 61 cornerstone tests passed with all 21
  neural tests explicitly skipped; `import athanor` did not import PyTorch. The original
  end-to-end cornerstone demonstration also passed from the wheel.
- In a separate wheel environment with the neural extra, all 20 neural contract tests
  passed. Its isolated CLI restored the trained checkpoint and completed two further
  optimizer updates, saving `artifacts/wheel-resumed.athanor` at update 642.

Neural contract coverage includes identity/display-name exclusion, candidate permutation,
new inventory without new parameters, actual optimizer updates, repeated internal steps,
rejection rollback, unknown versus unsatisfied preference, abstention versus failure,
dynamic self-capability observations, exact checkpoint/optimizer continuation, stale graph
save rejection and corrupted/unknown/nonfinite checkpoint input. One automated acceptance
test trains and executes the complete demonstration. See NEURAL_OBSERVER for the precise
architectural limits; no calibration, general grounding, CUDA performance or unrestricted
continual-learning result is inferred from these checks.

## Original Goal 001 verification evidence (0.1.0)

- Locked development installation: `uv sync --locked --extra dev --python 3.14
  --offline --cache-dir .tools\uv-cache` completed successfully using downloaded cache.
- Correctness: **61 automated tests passed** in the editable development installation.
- Distribution correctness: the same **61 tests passed** against the built wheel in
  a separate Python 3.14 environment, using isolated Python mode and the absolute test
  directory. That environment contains Athanor and no third-party runtime dependencies.
- Ruff lint passed; Ruff formatting passed; strict mypy passed for all nine package
  source modules.
- A source archive and `athanor-0.1.0-py3-none-any.whl` were built. The source archive
  includes architecture/API documents and tests; development caches are excluded.
- The end-to-end demonstration ran from the wheel with isolated Python mode,
  successfully saving, restoring, executing restored knowledge, teaching another
  procedure, and restoring that new teaching again.

Those original checks claimed no comparative performance, neural-learning, Linux/macOS portability, grounded
intelligence, or adversarial resource-isolation result is claimed by these checks.

## Demonstration results

See `artifacts/demo-report.json` for the actual run report, `artifacts/cornerstone.json`
for the saved knowledge, and `artifacts/cornerstone-reused.json` for the store after
restoration and further teaching. These are generated artifacts, not package fixtures.

| Observation | Result |
| --- | --- |
| Identity-preserving specialized update | Same local ID; revision 1 → 2; specialized category and label preserved |
| Inherited applicability | One installed claim, resolved through the specialization's ancestor path |
| Recursive exact total of nested composition | 11/2 |
| Composition containing a computed result | Valid participant; further recursive total is 6 |
| Restored recursive total | 11/2 |
| New procedure invoking the restored recursive procedure | 11 |
| That newly taught procedure after a second restore | 11 |
| Unsupported input and constraint-violating update | Rejected; complete snapshot unchanged |
| Unasserted applicability | unknown |
| Positive and negative attributed applicability | conflict |
| Production dependency closure for the first recursive total | 18 receipts |
| Saved revision count before restore/reuse | 27 |

The total rule is explicitly taught. Its behavior is not inferred merely because
members happen to be numbers. The instructor's applicability statements are attributed
claims, distinct from input/output execution contracts and external truth.

## Requirement coverage

| Goal 001 requirement | Implementation and evidence |
| --- | --- |
| Reason through the cornerstone before coding | CORNERSTONE definitions, eight questions, primitive review, counterexamples; DECISIONS alternatives and unresolved questions |
| Canonical compact structural representation | Ref/Record/Structure; category/role/state/members; exact revisions and immutable values |
| Distinguishable structures | Allocated local identity plus namespace; explicit identity versus state comparison tests |
| Typed valid relationships | Endpoint declarations; validated attributed claims; direct/inherited lookup and conflict tests |
| Rule-governed transformations | Taught Procedure expression records; exact interpreter and full-result validation |
| First-class recursive composition | Same data envelope; composites and computed results used in further composition/execution |
| Reusable operations represented internally | Immutable operation structures; projection/reflection tests and nested invocations |
| Safe invalid-operation handling | Full snapshot equality after failed calls, nested creations/updates, resource failures, stale updates, bad output |
| Knowledge installation after initialization | Public category/relation/operation teaching and binding; taught procedures reused by later procedures |
| General/specialized knowledge reuse | Full ancestry resolution, compatible schema aggregation, inherited relation paths without copied claims |
| Save/restore and reuse | Exact roundtrip; restored execution; teaching after restore; another restore |
| Inspect production | Per-revision receipts and transitive dependency inspection |
| Determinism | Explicit namespace/action sequence gives identical IDs and snapshot bytes; failed transactions consume no identity/receipt |
| Python 3.14 installable package | Built wheel/source archive, editable install, clean wheel install and isolated tests/demo |
| Knowledge separate from transient execution | Store retains revisions/receipts; evaluators, budgets and synchronization objects stay transient |
| Future neural connection and limitations | ARCHITECTURE proposal/validation boundary and ROADMAP milestones; no hidden AI capability |

## Failure cases exercised

Tests cover caller mutation; invalid/cyclic/deep values; Unicode values that cannot
be serialized; boolean/numeric confusion; schema/endpoint violations; multiple-parent
conflicts; foreign/missing/forward references; stale/protected updates; invalid
expression grammar and variable scope; arity/field/index/operand failures; lazy branch
behavior; unbounded recursion/work; nested rollback; production provenance; atomic
save failure; duplicate JSON and record keys; unknown format versions; digest mismatch;
noncanonical rational encodings; receipt/result/action/cause/revision corruption;
invalid persisted program grammar and execution interfaces, including rehashed input.

Restore validates chronological structural contracts, not historical computational
authenticity. Snapshot hashes do not authenticate instructors or establish external
truth. Resource limits do not bound large arithmetic operands or hostile snapshot size.

## Reproduce the checks

From `E:\ATHANOR`:

```powershell
uv sync --locked --extra dev --extra neural --python 3.14 --cache-dir .tools\uv-cache
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\ruff.exe check .
.venv\Scripts\ruff.exe format --check .
.venv\Scripts\mypy.exe
uv build --cache-dir .tools\uv-cache
.venv\Scripts\athanor-demo.exe --save artifacts\cornerstone.json
.venv\Scripts\athanor-neural-demo.exe --save artifacts\neural-observer.athanor
```

For an independent distribution check, create a second Python 3.14 environment,
install only the built wheel, and run its Python with `-I -m unittest discover -s
E:\ATHANOR\tests -q`. The tests use the standard library and import the installed
package; they do not require an external model or development tool dependency.

## Goal 006: recursive processes (0.6.0, 2026-10-11)

The continuous acceptance restores the actual Goal 005 checkpoint, its knowledge, model
and owned optimizer. A private territory law changes actual body movement; the observer
sees only stereo images and limited body feedback. It constructs 272 native call trees
from established arithmetic and a unit quantity, without receiving the territory law.
Four neural-selected forward actions give an initial history measurement and three
distinct input/output pairs. The Runtime automatically adopts the neural-selected
supported composition `F(q) = q/(q+1)`;
contradictory alternatives retain measured errors and qualified status. A subsequent
physical prediction on a new input has error **1.1111111475e-9** (tolerance **0.001**).
This evidence supports the observed interface scope, not a universal causal law.

The second interface supplies ordered nested numeric packets with independent varying
channels. Neural-selected stimuli precede their measurements. The shared model/evidence
select input `channel[1]` and output `readings[0]`, then install typed native input and
output bindings around the **same** acquired process. A held-out input **1/6** transfers
exactly; native reconstruction preserves unrelated output channels. False consequences
fail correspondence assessment. Out-of-range inputs and a changed one-pass/two-pass
context withhold reuse. Ordinary tests also retain ambiguity for co-varying selectors.

The acquired process joins the constructor by pinned reference. A new 324-candidate
inquiry receives model-selected measurements from a two-stage interface. It acquires
`F(F(q))`, which computes exact **1/4** at a fresh input **1/2**. The expression invokes
the existing process twice; it does not copy its implementation. **200** measured-error
consolidations change the existing shared scorer's weights/biases. The shared model's
parameter count stays **107,980**. Structural and semantic proposal accuracy remains
**48/48** in each retained evaluation before and after adaptation; abstention and measured
semantic ordering are also retained. These are bounded preservation checks, not a
comparative benchmark or an unlimited continual-learning guarantee.

The session saves and restores native constituents, evidence/revisions, mappings, inquiry,
self-capability counts, active neural state and optimizer. Restored procedures and mappings
execute without a teacher or replaying the original sequence. A focused test additionally
clears the transient observation lists before reuse. A separately installed 0.6 wheel
restores the same checkpoint, computes 1/4, saves, and restores again with exact native
process/world state and optimizer update count **5,485**. Torch is supplied from the existing
local neural environment for that isolated package check; the package import resolves to
the installed wheel, not the source checkout.

Evidence files generated under `artifacts/`:

- `process-observer.athanor`, `process-observer.report.json`, `process-run.log`.
- `process-tests.log`: **15 focused tests pass**, including exact continuation of the
  next shared process-training update with persisted retention anchors.
- `goal006-regression.log`: **181 tests pass** in the full regression, including all
  three longer legacy training acceptances. This run began before the final retention
  persistence test was added; the subsequent current-source run below covers that test
  and the final process/checkpoint changes.
- `goal006-current-contracts.log`: 179 current-source correctness/acceptance tests,
  excluding only the three longer legacy training acceptances run in the full regression.
- `goal006-wheel-tests.log`: **94 dependency-free tests pass**, 88 neural tests skipped.
- `goal006-installed-continuation.log` and `process-installed-continued.athanor`.
- `goal006-headless.json`: restored version 0.6.0, paused world at tick 5.
- `goal006-build.log`: successful wheel and source distribution build.
- `process-viewer.jpg` and `process-viewer-ready.jpg`: actual restored viewer and result.

Ruff lint/format, strict mypy over 42 source files, and offline lock consistency pass.
The local viewer remains paused at `http://127.0.0.1:8766/`; browser application of the
restored recursive process displays exact **1/4** and adds a native application receipt.
The viewer exposes candidates, qualified alternatives, dependencies, proposals, mappings,
measurements, actual application results, capability counts and shared learning updates.

The initial inductive family is bounded typed exact quantity call trees. Taught compatible
conditional/accumulated procedures can participate, and acquired procedures compose again.
The acceptance establishes direct body-feedback discovery and structured digital transfer;
it does not establish nonlinear discovery from uncertain camera depth, general inverse
discovery, unrestricted synthesis, automatic arbitrary-unit conversion, or unbounded scale.


## Goal 007: native concept formation and structural induction, 0.7.0

The full preservation regression completes **213 tests successfully** in **2,231.184
seconds**, including all three earlier longer neural, semantic and grounded training
acceptances, Goal 005 autonomous discovery, Goal 006 recursive process acceptance and
the complete Goal 007 cross-domain session. The final unknown-whole-outcome edge is
subsequently covered by both current-source and installed-wheel 30-test contract suites.
Ruff lint/format, strict mypy over **46 source files**, offline lock consistency and
wheel/source-distribution builds pass. The local editable installation reports 0.7.0.

The continuous session restores the actual 0.6.0 `process-observer.athanor`, including
its learned model, optimizer and recursive processes. Independently generated anonymous
relational observations and raw symbolic rasters supply examples only; the observer has
no generator reference, hidden program, supplied concept name or correct grouping.
No official ARC data, published puzzles, solution material or benchmark campaign is used.

The session forms **69 operational concepts** (**40 supported, 29 provisional**), including a recurring unnamed directed
participation distinction used by the acquired relational transformation. The relational
inquiry constructs **765 candidates**, qualifying **763** inconsistent explanations.
Its fresh symbolic instance is exact. The separate spatial inquiry evaluates **1,024
candidates** and predicts a fresh raw raster exactly. A raster representation then reuses
the same learned relational concept by pinned reference through generic native adapters;
that inquiry evaluates **2,048 candidates** and its held-out output is also exact.
A single symmetric demonstration retains **four fitting alternatives** without adoption.

A goal requires the learned eligible initial participants, their cardinality and ordering
by an observed anonymous property. Its acquired native plan contains **two transformations**.
Three distinct later inputs satisfy the original native goal conditions and support the
plan within that schema. A composite concept calls earlier concepts by reference. Saving
and restoring retains native concepts/rules/tasks/history, shared parameters, optimizer,
recurrent state and capability records. After restoration another fresh goal is solved
without replaying the original demonstration examples.

There are still **107,980 shared parameters**. Structural consolidation changes only the
four existing scorer tensors, allocates no additional network, and performs **960** new
optimizer updates, continuing from **5,485 to 6,445**. Earlier structural/semantic proposal
measurements remain **1.0 before and after** on the retained sets. These are proposal
measurements, not universal execution success: the structural set's executable/exact-result
rate remains **19/48** because it includes unsupported and incompatible situations.
The previously acquired Goal 006 recursive process still executes exact **1/4**.

**30 ordinary structural and induction tests pass**, covering typed/native binding,
directed and noncommutative relationships, generic grouping/raster conversion, exact
JSON boundaries, unknown membership, ambiguity, bounded goals, provisional plans,
contradictions and dependency qualification. A later measured example can remove the
preferred rule, select a surviving explanation with the shared scorer, revoke the
rejected procedure and restore the revised state. Repeating one observation cannot
establish concept support, and repeating one application cannot establish generalization.
An unsupported neural first choice abstains rather than receiving a symbolic substitution.
Unknown whole outcomes remain unresolved hypotheses, with no positive reliability signal
or support for adoption; absent cells within a known raster remain valid observations.
Active-query checkpoint restoration and the exact next shared training update are tested.
Malformed inquiry metadata and invalid native/resource executions are rejected.
The final unknown-whole-outcome edge is covered by subsequent **30-test current-source
and installed-wheel contract runs**; the longer full regression began before that edge
was tightened. The change does not alter the known-outcome acceptance paths.

A clean installed 0.7 wheel runs **108 dependency-free tests successfully**, with the
**105 neural tests skipped** because that environment deliberately lacks PyTorch. A
separate installed-wheel continuation reuses the local neural extra while resolving
Athanor from the wheel: an unseen visual input, restored goal procedure, newly solved
goal, exact **1/4**, another save/restore and optimizer update **6,525** all pass.

The actual viewer is paused at `http://127.0.0.1:8766/`. Its problem editor applies a
restored learned rule to a new raw raster and displays the actual result. The headless
problem file path also produces an exact fresh measured visual result, records its
native application and saves the continuation. Structural inspection is read-only.

Compact native JSON preserves every entry, revision, receipt and semantic digest while
removing indentation. The completed checkpoint is **3.56 MiB** compressed and its native
history member is **86.05 MiB**, within the unchanged **256 MiB** import budget. Compact
and readable snapshots restore identical native state; evidence is not discarded.
Checkpoint format 5 retains all earlier format readers. Native execution authority,
atomicity, provenance and graph/history activation budgets remain in force.

Evidence artifacts under `artifacts/` include:

- `concept-observer.athanor`, `concept-observer.report.json`, `concept-run.log`.
- `goal007-contracts.log`, `goal007-regression.log`.
- `goal007-wheel-tests.log`, `goal007-installed-contracts.log`,
  `goal007-installed-continuation.log`,
  `concept-installed-continued.athanor`, `goal007-build.log`.
- `goal007-headless-problem.json`, `goal007-headless.json`,
  `concept-headless-continued.athanor`.
- `concept-viewer.athanor`, `concept-viewer.png`, `concept-viewer-saved.png`,
  `goal007-viewer.log`.

The demonstrated grammar is finite: observed field predicates, ordered peer relations,
directed participation, connectivity, exact collection/geometry operations and bounded
native compositions. Raw symbolic raster learning does not establish new camera-object
semantics. Supported evidence is scoped; a failed bounded search remains unknown.
This acceptance establishes operational representation/rule acquisition and reuse within
those limits, not unrestricted synthesis, autonomous new primitives or general intelligence.

## Goal 008 — representation invention and adaptive abstraction

The continuous session restores the real 0.7 learned checkpoint, constructs native
interpretations from anonymous collections and raw RGB, transfers acquired fragments and
relationships across interfaces, retains and tests ambiguity, revises a representation
for a composed goal, and continues after exact full-state restoration. See
[REPRESENTATION_VALIDATION](REPRESENTATION_VALIDATION.md) for the requirement/evidence map.

Measured validation on Python 3.14.6:

- Full regression: **232 tests passed**, 2,270.226 seconds.
- Final Goal 008 contracts: **21 tests passed**, 142.635 seconds, including current
  record-projection/camera-measurement additions and real-checkpoint continuous acceptance.
- Installed wheel with no neural dependencies: **113 passed, 121 skipped** of 234 tests.
- Installed wheel with the existing neural runtime: **20 contracts passed**, plus fresh
  perceptual transfer, prior numeric 1/4 execution, continuation without examples and a
  second exact restoration. Athanor imports were asserted to come from the installed wheel.
- Existing structural/semantic retention cases have identical before/after results.
  Proposal/semantic scores remain 1.0; structural executable/exact outcome is 19/48 because
  the set includes unsupported/incompatible cases. This is not 100% execution coverage.
- Headless commands execute the same Application and produce `[9, 15, 21]` for a fresh
  symbolic observation, then save it. The viewer captures RGB through the actual encoder,
  executes an acquired interpretation yielding `[9, 12, 15]`, and saves complete state.
- Ruff lint and 62-file format checks, strict typing on 49 source files, offline lock
  validation, wheel/source build and diff-whitespace checks pass. Package version is 0.8.0.

Evidence artifacts:

- `representation-observer.athanor`, `representation-observer.json`, `goal008-acceptance.log`.
- `goal008-regression.log`, `goal008-final-contracts.log`, `goal008-build.log`.
- `goal008-wheel-tests.log`, `goal008-installed-continuation.log`,
  `representation-installed-continued.athanor`.
- `goal008-retention.json`, `goal008-retention.log`.
- `goal008-headless-problem.json`, `goal008-headless.json`, `representation-headless.athanor`.
- `representation-viewer.athanor`, `representation-viewer.jpg`, `goal008-viewer.log`.

Outer checkpoint format 6 preserves earlier formats 1–5. Native imports, namespaces,
revision pins, exact computation, atomic rollback, evidence and resource limits remain
in force. No external AI teacher, new representation-specific network, ARC material,
privileged generator solution or comparative architecture benchmark is used.
