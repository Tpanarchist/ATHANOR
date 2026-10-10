# ATHANOR Goals 001 and 002 — verified implementation

Date: 2026-10-10. Platform exercised: Windows, CPython 3.14.6.
Development tools exercised: uv 0.11.28, Ruff 0.17.0, mypy 2.4.0.
The development dependency resolution is recorded in `uv.lock`.

## Goal 002 learning and integration evidence

The neural implementation uses PyTorch 2.14.1 on the exercised CPU path with Python
3.14.6. The kernel implementation and its original 61 tests are preserved; its only
source change is the package version. The neural dependency is optional. A separate
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
