# ATHANOR neural API — Goal 002

Install `athanor[neural]` with Python 3.14. The core package still has no third-party
runtime requirements. `athanor.neural` imports PyTorch; `athanor` alone does not.

## The small observer interface

```python
from athanor.neural import ModelConfig, NeuralObserverModel, Observation, Observer

model = NeuralObserverModel(ModelConfig(width=64, refinement_steps=4, seed=1729))
observer = Observer(knowledge, model)  # knowledge is an existing Athanor instance
decision = observer.observe(Observation(request_ref, offered_argument_refs))
decision = observer.think(steps=4)  # More processing, no new observation required.
feedback = observer.act()  # Actual exact execution, or learned abstention.
observer.save("artifacts/observer.athanor")
restored, optional_trainer = Observer.restore("artifacts/observer.athanor")
```

This constructs a random, untrained model. Restore trained parameters or train first
before expecting the curriculum competence. `Observation.candidates` is an unordered,
unique tuple of exact input references. Context is an ordinary structural object;
no natural-language parser is provided. The demonstrated context uses `kind`, `task`
and ordered `slots` reference fields. This schema belongs to teaching, not the model's
selector: graph construction handles arbitrary record paths and reference edges.

`Decision` contains a `Proposal(operation: Ref | None, arguments: tuple[Ref, ...])`,
joint pointer confidence, and `Expectation(success, preference, outcome)`. `None`
operation means abstention, with no executable arguments. Confidence is the product
of selected pointer probabilities, not a calibrated epistemic certainty.

An operation's declared arity determines how many argument pointers are decoded.
Every installed operation is offered; no type/applicability rule chooses or repairs
the action. If no input candidates exist but the model proposes an operation requiring
inputs, the proposal reaches the kernel with missing arguments and is rejected.
Exact namespaced revisions are retained. Old revisions do not silently become latest.

`act(decision=None, evaluate=None, correction=None)` optionally accepts an explicit
proposal to probe validation. Such a probe is caller supplied, not a learned decision.
`evaluate(result_ref) -> bool` provides preference evidence only after successful
execution. An absent evaluator means unknown satisfaction. It should be a pure,
total callback; evaluation happens **after** kernel commit, so evaluator failure cannot
undo a successful execution. The evaluator/teacher is not part of ordinary inference.
An optional `correction: Proposal` is explicit structured instruction.

`Feedback` distinguishes `executed`, `rejected`, and `abstained`; it carries actual
result/receipt, errors, observed outcome, preference evidence, and typed discrepancies:

| Channel | Measured quantity | Unknown when |
| --- | --- | --- |
| Predictive | Actual structural signature minus predicted signature | No successful result |
| Preference | Explicit satisfaction minus predicted satisfaction | No evaluator/result |
| Capability | Execution acceptance (1/0) minus predicted success | Abstention |

The outcome signature is: data role, has members, has numeric fields, field count
clipped/scaled to eight, identity-preserving update, fresh-result execution. It does
not predict the arithmetic answer. Arithmetic values remain exact kernel outputs.
Claims about external effects require future environment observation contracts.

## Activation and the self-map

`observer.state` holds detached per-reference field tensors, global context and
internal-step count. The observer retains no autograd history between inference calls.
`think()` preserves temporary state; constructing a new observer starts a new episode.
There is no automatic reset merely because a new observation arrives.

`self_ref` is an ordinary cornerstone structure. Each `act()` records the observation,
proposal, expectations, actual status/result/receipt, separate discrepancies,
correction, capability measurements and activation history as another ordinary
structure. These active references enter the same graph and recurrent model.
`capabilities[operation_ref] = (attempts, successes)` records measured execution,
including rejections; it is not an independently trained self-network. Learned success
and preference predictions come from shared processing and separate small readouts.

`history_limit` bounds the active self-map and sampled activation history; previous
durable self-observations remain in knowledge. Their sampled neural coordinates are
attributed seven-decimal rational measurements, whereas checkpoint tensors retain
full float32 precision. Foreign/missing references in rejection probes are represented
as unresolved (`None`) in self evidence, with the actual error retained.

Inference temporarily uses `eval()`/`no_grad()` and restores the caller's model mode.
It does not update learned parameters. `max_nodes` explicitly bounds the active graph.
Exceeding it raises `ValidationError`; it never silently truncates candidate knowledge.
The current observer/trainer is single-owner; concurrent training, teaching, saving
or execution across threads is not a supported neural contract.

## Teaching versus supervised training

Install knowledge using unchanged `teach_category`, `teach_relation`, `bind`, and
`teach_operation`. This changes the cornerstone, not neural weights. New definitions
allocate structural nodes and activations, not new output heads or embedding rows.

```python
from athanor.neural.curriculum import lessons
from athanor.neural.training import Trainer, TrainingConfig

trainer = Trainer(model, TrainingConfig(learning_rate=0.002, batch_size=16, seed=321))
material = lessons(256, model.config, seed=1000)
losses = trainer.train(material, epochs=40)
observer.think()  # Refresh an active decision after explicit parameter changes.
observer.save("artifacts/trained.athanor", trainer=trainer)
```

`Lesson` contains a pre-execution graph frame, desired pointer proposal, execution
probe, measured feedback, and optional prior graph. Supervised labels never enter
ordinary input features. Correction episodes deliberately include explicit corrective
evidence as input; this is visible instruction, not hidden label leakage. Graphs are
packed into sparse batches. One AdamW optimizer updates the shared encoder, recurrent
process, pointer scorer and readouts. Losses separately report selection, arguments,
applicability, capability, preference and structural prediction. The scalar optimization
loss combines those tasks; it does not replace typed inference discrepancies with a
single reward. Carried fields are detached between episodes/steps.

The initial curriculum covers exact add/subtract/divide/recursive total, category
specializations, nested composites, relationship-guided operation selection, shuffled
input pools, missing/incompatible/unsupported requests and explicit preference changes.
Numeric values, namespaces, definition names and allocation order vary. New-definition
evaluation adds multiplication with the same encoding and output pointers. Neither the
trainer nor teacher executes instead of the cornerstone at inference time.

To provide another curriculum, construct `Lesson` instances from public teaching,
`GraphBuilder.build`, explicit target proposals, and actual `measure` feedback.
This version does not discover its own training labels or orchestrate autonomous
continual updates. Replay/retention/conflict policies remain future work.

## Checkpoints and continuation

One `.athanor` ZIP contains `metadata.json`, canonical `knowledge.json` and restricted
tensor/primitive `tensors.pt`. Model configuration, parameters, exact field/context,
current observation and pending decision, active self-map/history/capability counts,
and optional optimizer/local shuffle RNG are saved together. Atomic replacement
preserves an older checkpoint if writing fails. Refresh with `think()` before saving
after changing knowledge or weights; a stale cached graph is rejected before saving
and on restore.

Restore validates format members, digests, model dimensions/names/finite float32
tensors, exact reference resolution, reconstructed graph alignment, proposal pointers,
and optimizer tensor shapes. It uses `torch.load(..., weights_only=True)` rather than
loading Python model objects or executable source. Checkpoint digests detect corruption,
not authenticity. Each member has a 256 MiB import bound; configurable runtime/model
resources are separate from this format's defensive import budget.

Restoration returns `(Observer, Trainer | None)` on CPU. Saved model seed and optimizer
shuffle state permit deterministic continuation on the exercised CPU/software path.
Bitwise equality is not promised across hardware, PyTorch versions or acceleration
backends. CUDA portability/performance was not validated in Goal 002. The base package
can restore `knowledge.json` independently via `Athanor.loads` without PyTorch.

Goal 004 adds an optional `sensory` sequence to the same model's forward call and
an optional compact `Frame.intent_features` view. Existing calls/parameters and the
no-sensory path retain their contracts. `PixelObserver` injects learned RGB/proprioceptive
activation before the same relational refinement and operation pointers. Physical
prediction is an additional readout of that shared state. Its complete outer checkpoint
is distinct from the native/Goal 002/003 formats above. See GROUNDED_OBSERVER and
GROUNDED_API for territory isolation, training, uncertainty and live operation.

```powershell
.venv\Scripts\athanor-neural-demo.exe --save artifacts\neural-observer.athanor
.venv\Scripts\athanor-neural-demo.exe --resume artifacts\neural-observer.athanor --epochs 5 --save artifacts\continued.athanor
```
