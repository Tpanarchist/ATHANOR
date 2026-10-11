"""One continuous discovery, transfer, further learning and checkpoint continuation."""

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import torch

from athanor.neural.curriculum import situation
from athanor.neural.training import evaluate as evaluate_structural
from athanor.semantics.curriculum import case
from athanor.semantics.evaluation import evaluate as evaluate_semantic
from athanor.world import World

from .checkpoint import restore, save
from .discovery import Estimate
from .runtime import Runtime
from .training import train_inquiry


def require(condition: bool, explanation: str) -> None:
    if not condition:
        raise AssertionError(explanation)


def run_demo(
    path: Path,
    *,
    base_checkpoint: Path = Path("artifacts/grounded-observer.athanor"),
    training_steps: int = 1600,
    response: float = 0.12,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    previous, _ = restore(base_checkpoint)
    # Enter a new territory with the same learned model. The observer's existing
    # knowledge, self-map, activation and capabilities remain installed.
    observer = previous.observer
    model = observer.observer.model
    native_modes = ("valid", "unsupported", "missing", "incompatible", "correction")
    meaning_modes = ("valid", "alias", "unknown", "missing", "ambiguous", "context")
    native_cases = [
        situation(51000 + i, mode=native_modes[i % len(native_modes)]) for i in range(48)
    ]
    meaning_cases = [case(52000 + i, mode=meaning_modes[i % len(meaning_modes)]) for i in range(48)]
    retained_before = {
        "structural": evaluate_structural(model, native_cases),
        "semantic": evaluate_semantic(model, meaning_cases),
    }
    print("Teaching general inquiry priorities without discovery-world answers…", flush=True)
    inquiry_training = train_inquiry(observer, steps=training_steps)
    retained_after = {
        "structural": evaluate_structural(model, native_cases),
        "semantic": evaluate_semantic(model, meaning_cases),
    }
    for family in retained_before:
        require(
            retained_after[family]["proposal"] >= retained_before[family]["proposal"] - 0.05,
            f"Inquiry training retains established {family} proposals",
        )
    world = World(namespace="discovery-territory", interaction_distance=response)
    world.add_object((0, 0, 0.75), size=(0.15,) * 3, movable=True, color=(230, 50, 40))
    observer.goal, observer.target = "observe", None
    runtime = Runtime(world, observer, trainer=previous.discovery_trainer)
    initial_native = len(observer.discovery.estimates)
    require(initial_native == 0, "The new environmental relationship is initially untaught")
    # A contact inquiry has explicit action constraints. The neural scorer chooses
    # their ordering, repetition or abstention; no action sequence is supplied.
    runtime.start_discovery(18, focus="object", allowed=("interact", "wait"))
    trace: list[dict[str, object]] = []
    learned: Estimate | None = None
    for _ in range(18):
        step = runtime.autonomous_step()
        trace.append(step)
        learned = next(
            (
                v
                for v in observer.discovery.estimates.values()
                if v.channel == "object"
                and v.action == "interact"
                and v.status == "supported"
                and v.operation is not None
            ),
            None,
        )
        if learned is not None:
            break
        if observer.discovery.status != "running":
            break
    require(
        learned is not None, "Autonomous interventions acquire an operational object relationship"
    )
    assert learned is not None
    require(
        all(not step.get("manual", False) for step in trace), "Every discovery action is neural"
    )
    require(any(step.get("intended") == "wait" for step in trace), "Passive comparison is observed")
    require(len(learned.evidence) >= 3, "Single correlation cannot consolidate a relationship")
    require(
        abs(learned.mean[2] - response) < 0.09, "The fitted camera effect reflects untaught physics"
    )
    operation, knowledge = learned.operation, learned.relationship
    assert operation is not None and knowledge is not None
    acquired = {
        "operation": str(operation),
        "relationship": str(knowledge),
        "effect": learned.mean,
        "samples": len(learned.samples),
        "dispersion": learned.dispersion,
        "scope": learned.scope,
        "hypothesis": str(learned.hypothesis),
        "evidence": [str(ref) for ref in learned.evidence],
    }
    # Change the actual arrangement, with no teacher, belief update, or policy answer.
    world.relocate(world.objects[0], (0.08, 0, 0.82))
    observer.observe(world.sense())
    target = next(p for p in observer.last_percepts if p.status == "visible" and p.relative)
    prediction_ref = observer.discovery.predict(
        learned, tuple(target.relative or ()), appearance=target.appearance
    )
    require(prediction_ref is not None, "Acquired native procedure executes on a fresh arrangement")
    assert prediction_ref is not None
    prediction = observer.knowledge.get(prediction_ref).fields
    require(
        prediction["status"] == "supported", "Fresh measured input satisfies learned conditions"
    )
    automatic_reuse = [
        ref
        for ref in observer.discovery.options
        if observer.knowledge.get(ref).fields.get("acquired_prediction") is not None
    ]
    require(bool(automatic_reuse), "Observer itself activates acquired procedures for experiments")
    transfer_trace = []
    transfer_error = None
    for _ in range(6):
        before_percept = next(
            (p for p in observer.last_percepts if p.status == "visible" and p.relative), None
        )
        step = runtime.autonomous_step()
        transfer_trace.append(step)
        if step.get("intended") == "interact" and before_percept is not None:
            after_percept = next(
                (p for p in observer.last_percepts if p.status == "visible" and p.relative), None
            )
            if after_percept is not None:
                transfer_error = abs(
                    float(cast(Fraction, prediction["effect_2"]))
                    - (
                        (after_percept.relative or (0, 0, 0))[2]
                        - (before_percept.relative or (0, 0, 0))[2]
                    )
                )
            break
    require(
        transfer_error is not None and transfer_error < 0.12,
        "A model-selected transfer interaction corroborates the acquired prediction",
    )
    # A new directed inquiry changes relevance, without resetting knowledge or weights.
    runtime.start_discovery(12, focus="self")
    parameters = {name: value.clone() for name, value in model.state_dict().items()}
    continued_trace = [runtime.autonomous_step() for _ in range(4)]
    require(any(step.get("acted") for step in continued_trace), "Observer continues investigating")
    shared_changed = [
        name
        for name, value in model.state_dict().items()
        if not torch.equal(value, parameters[name])
    ]
    require(bool(shared_changed), "Slow consolidation updates actual shared learned parameters")
    require(
        observer.discovery.estimates[learned.key].operation == operation,
        "Continued learning preserves acquired native procedure identity",
    )
    save(runtime, path)
    restored, trainer = restore(path)
    require(trainer is not None, "Continual optimizer state restores")
    require(
        runtime.observer.discovery.metadata() == restored.observer.discovery.metadata(),
        "Entire inquiry, hypotheses and resource state restore",
    )
    require(
        runtime.observer.knowledge.dumps() == restored.observer.knowledge.dumps(),
        "Native knowledge and provenance restore exactly",
    )
    left, right = runtime.autonomous_step(), restored.autonomous_step()
    require(left == right, "Restored next model decision and execution match")
    require(runtime.world.dumps() == restored.world.dumps(), "Territory continuation matches")
    require(
        all(
            torch.equal(value, restored.observer.observer.model.state_dict()[name])
            for name, value in model.state_dict().items()
        ),
        "Continued shared parameters match",
    )
    # Continue through the next consolidation boundary after restoring.
    for _ in range(3):
        require(
            runtime.autonomous_step() == restored.autonomous_step(),
            "Optimizer continuation preserves subsequent learning decisions",
        )
    require(
        all(
            torch.equal(value, restored.observer.observer.model.state_dict()[name])
            for name, value in model.state_dict().items()
        ),
        "Resumed optimizer updates match",
    )
    continued_path = path.with_name(path.stem + "-continued" + path.suffix)
    save(restored, continued_path)
    retained_continued = {
        "structural": evaluate_structural(model, native_cases),
        "semantic": evaluate_semantic(model, meaning_cases),
    }
    for family in retained_before:
        require(
            retained_continued[family]["proposal"] >= retained_before[family]["proposal"] - 0.05,
            f"Continual neural updates retain established {family} proposals",
        )
    report = {
        "version": "0.5.0",
        "untaught_territory_response": response,
        "teacher_answer_supplied": False,
        "hidden_world_inputs": False,
        "inquiry_training": inquiry_training,
        "retained_before": retained_before,
        "retained_after": retained_after,
        "retained_continued": retained_continued,
        "discovery": acquired,
        "trace": trace,
        "transfer_prediction": str(prediction_ref),
        "transfer_effect": [float(cast(Fraction, prediction[f"effect_{i}"])) for i in range(3)],
        "automatic_native_reuse": [str(ref) for ref in automatic_reuse],
        "transfer_trace": transfer_trace,
        "transfer_error": transfer_error,
        "continued_trace": continued_trace,
        "shared_parameters_changed": shared_changed,
        "neural_consolidations": observer.discovery.neural_updates,
        "checkpoint_next_step_identical": left == right,
        "optimizer_continuation_identical": True,
        "checkpoint": str(path),
        "continued_checkpoint": str(continued_path),
        "contact_inquiry_constraints": ["interact", "wait"],
        "continued_inquiry_actions": list(observer.discovery.allowed),
        "limits": [
            "conditional constant-effect hypothesis family",
            "camera-derived approximate effect; not universal causal law",
            "general inquiry ranking is learned and can abstain",
            "appearance alone cannot distinguish identical objects",
        ],
    }
    path.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/discovery-observer.athanor")
    )
    parser.add_argument(
        "--base-checkpoint", type=Path, default=Path("artifacts/grounded-observer.athanor")
    )
    parser.add_argument("--training-steps", type=int, default=1600)
    parser.add_argument("--response", type=float, default=0.12)
    args = parser.parse_args()
    print(
        json.dumps(
            run_demo(
                args.checkpoint,
                base_checkpoint=args.base_checkpoint,
                training_steps=args.training_steps,
                response=args.response,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
