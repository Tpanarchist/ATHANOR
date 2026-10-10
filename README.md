# ATHANOR

**The furnace of transformation.** Recursive Intelligence Architecture.

**Foundation + Knowledge + Experience**  
**Activation → Transformation → Refinement**

Athanor is a new AI architecture built around reusable processes and representations.
The long-term model will acquire, activate, combine, and refine knowledge, eventually
applying controlled refinement to its own processes. The name describes the mechanism
by which something develops, rather than presuming the finished intelligence.

Goal 001 supplies a working local computational cornerstone: immutable versioned
structures, typed relationship claims, inherited schemas, structured teaching,
reusable exact procedures, recursive composition/execution, production receipts,
and persistence. Goal 002 integrates a trainable recurrent neural observer: shared
structural encoding, learned operation/argument pointers, persistent activation,
measured self-observations, supervised training, and resumable checkpoints. Inference
uses local learned parameters and exact installed procedures. No external AI is in
the execution path.

## Install and run (Python 3.14)

From `E:\ATHANOR` in PowerShell, with Python 3.14 installed:

```powershell
uv sync --locked --extra dev --extra neural --python 3.14 --cache-dir .tools\uv-cache
.venv\Scripts\athanor-demo.exe --save artifacts\cornerstone.json
.venv\Scripts\athanor-neural-demo.exe --save artifacts\neural-observer.athanor
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\ruff.exe check .
.venv\Scripts\ruff.exe format --check .
.venv\Scripts\mypy.exe
uv build --cache-dir .tools\uv-cache
```

Alternatively, use `py -3.14 -m venv .venv` and
`.venv\Scripts\python.exe -m pip install -e ".[dev,neural]"`. Runtime installation can omit
`[dev]`; the foundation itself has **zero third-party runtime dependencies**. PyTorch
is an optional `[neural]` dependency; importing `athanor` does not import it.
`python -m athanor.demo` is also supported. The demo creates the requested save
directory; the core `save()` API requires an existing parent directory.

The neural demo trains from scratch on CPU using 256 structured lessons for 40 epochs
(about 95 seconds on the exercised development machine). It checks held-out learning,
actual execution, self-feedback correction, new teaching after training, and exact
checkpoint continuation. It saves knowledge, weights, runtime activations, self-map,
optimizer, and an adjacent JSON report. `python -m athanor.neural.demo` also works.

Continue training an existing checkpoint:

```powershell
.venv\Scripts\athanor-neural-demo.exe --resume artifacts\neural-observer.athanor --epochs 5 --save artifacts\continued.athanor
```

Inference from a checkpoint is local and does not require the curriculum or teacher:

```python
from athanor.neural import Observer

observer, trainer = Observer.restore("artifacts/neural-observer.athanor")
decision = observer.think()  # Refine the restored observation without a new input.
print(decision.proposal, decision.expectation)
# observer.act() executes that proposal and records its measured feedback.
```

For a new request, create ordinary cornerstone data containing its structured context
and call `observer.observe(Observation(request_ref, offered_argument_refs))`. The
adapter supplies context and candidates; learned pointers select the operation and
ordered arguments. The exact kernel validates them. See the neural API for contracts.

## What the demonstration establishes

The runnable sequence is **Create → Teach → Bind → Transform → Compose → Execute →
Inspect → Save → Restore → Reuse**. A general measurable category gains a taught
recursive total procedure and an attributed relationship. A specialized category
inherits its contracts without copied edges. A declared update preserves one leaf's
identity, composites nest, and the taught procedure calculates an exact recursive
total. A restored store runs both the original procedure and a newly taught procedure.
The demo also checks unknown knowledge, contradictory evidence, unsupported execution,
and atomic rejection. All behavior goes through the same general public foundation.

## Small teaching example

```python
from fractions import Fraction
from athanor import Athanor, Endpoint, FieldSpec, Procedure, Role
from athanor.programs import argument, binary, field, literal

a = Athanor()
quantity = a.teach_category("Quantity", {"value": FieldSpec("number", minimum=0)})
half = a.teach_operation(
    "half",
    Procedure(
        inputs=(Endpoint(Role.DATA, quantity),),
        result=quantity,
        fields={"value": binary("div", field(argument(0), "value"), literal(2))},
    ),
)
input_ref = a.create(quantity, {"value": Fraction(3, 2)})
output_ref = a.execute(half, [input_ref])
assert a.get(output_ref).fields["value"] == Fraction(3, 4)
assert Athanor.loads(a.dumps()).get(output_ref) == a.get(output_ref)
```

Teaching installs schemas or procedures after initialization. Procedure definitions
are validated immutable structures. Execution checks the declared inputs, evaluates
portable expression data, checks complete results, and commits all nested changes
atomically. Relation teaching declares endpoint types; binding adds attributed
positive/negative/uncertain evidence. Assessments describe installed support, not
proof of external truth.

Use `get(ref)` for an exact revision, `get(ref.id)` to explicitly request latest
state, `inspect(ref)` for its production receipt, and `provenance(ref)` for the
dependency closure. Members and stored references are exact revisions. Revisions
never alter previously created composites. References include their store namespace.
Fresh stores have independent namespaces; specify a namespace to reproduce identical
IDs and snapshot bytes from the same actions. Foreign references are rejected;
import/unification between lineages needs a future explicit protocol.

## Documents and package

- [Cornerstone definitions, primitive review, invariants and counterexamples](docs/CORNERSTONE.md)
- [Full intended architecture and precise current boundary](docs/ARCHITECTURE.md)
- [Decisions, alternatives and unresolved issues](docs/DECISIONS.md)
- [Roadmap to the neural model and environments](docs/ROADMAP.md)
- [Public API and expression reference](docs/API.md)
- [Neural integration design and critical review](docs/NEURAL_OBSERVER.md)
- [Training, inference, feedback and checkpoint API](docs/NEURAL_API.md)
- [Verification evidence and requirement coverage](docs/VALIDATION.md)

`values` and `contracts` define representation; `store` enforces invariants;
`programs` evaluates taught procedure data; `kernel` exposes the small public API;
`persistence` saves validated snapshots; `demo` exercises the foundation. Tests live
outside the installable package. The optional `neural` package contains the shared
encoder/model, observer boundary, curriculum/trainer, checkpoint format, and learning demo.

## Limits that matter

The neural observer has learned bounded structural competence. In the verified run,
exact proposal accuracy improved from 0% to 95.31% on 128 held-out situations; operation
selection and insufficiency abstention reached 100%. Six proposals still bound an
argument incorrectly. A separate 48-case new-definition set reached 100% exact
proposals. These are curriculum results, not evidence of open-domain intelligence.
Natural-language teaching, perception, grounded semantics, calibrated open-world
uncertainty, unrestricted continual learning, and autonomous self-improvement remain
unimplemented. Preference correction is explicit structured instruction.

Schemas and definitions are immutable; revised definitions get new identities.
Inheritance rejects conflicting field contracts rather than implementing override
or constraint refinement. The language has a fixed trusted opcode set. Execution
limits bound expression work/call depth, not wall time or memory for hostile input.
Histories grow and transactions copy indexes; larger workloads need measured indexing,
retention, and storage work. Snapshot digests detect corruption and are not signatures.
Historical receipt validation does not authenticate teachers or independently prove
that an imported computation occurred. See the architecture and decisions for the
implemented neural interfaces and open contracts that must be addressed next.
