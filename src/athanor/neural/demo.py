"""Train, execute, adapt through self-feedback, teach new knowledge, save and resume."""

import argparse
import json
import time
from fractions import Fraction
from pathlib import Path

import torch

from athanor import Endpoint, Evidence, Procedure, Role
from athanor.programs import argument, binary, field

from .contracts import Observation, Proposal
from .curriculum import Situation, lessons, situation
from .model import NeuralObserverModel
from .observer import Observer
from .training import Trainer, evaluate


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"Neural demonstration contract failed: {message}")


def run_demo(path: Path, *, epochs: int = 40, count: int = 256) -> dict[str, object]:
    started = time.perf_counter()
    # Execution settings are owned by the runnable demonstration, not import-time globals.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = NeuralObserverModel()
    trainer = Trainer(model)
    material = lessons(count, model.config)

    def held_out() -> list[Situation]:
        modes = (
            "valid",
            "valid",
            "valid",
            "valid",
            "unsupported",
            "missing",
            "incompatible",
            "correction",
        )
        return [situation(100000 + i, mode=modes[i % len(modes)]) for i in range(128)]

    before = evaluate(model, held_out())
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    loss: dict[str, float] = {}
    completed = 0
    while completed < epochs:
        block = min(5, epochs - completed)
        loss = trainer.train(material, epochs=block)
        completed += block
        print(
            f"Trained {completed}/{epochs} epochs; "
            f"pointer loss {loss['selection'] + loss['arguments']:.4f}",
            flush=True,
        )
    after = evaluate(model, held_out())
    change = (
        sum(
            float((model.state_dict()[key] - value).square().sum())
            for key, value in initial.items()
        )
        ** 0.5
    )
    _require(change > 0, "training changes learned parameters")
    _require(
        after["proposal"] >= 0.90 and after["proposal"] > before["proposal"] + 0.40,
        "held-out learned proposal accuracy",
    )
    _require(after["abstention"] >= 0.90, "learned insufficiency recognition")
    novel = evaluate(model, [situation(200000 + i, novel=True) for i in range(48)])
    _require(
        novel["proposal"] >= 0.90, "new definitions/specializations use existing neural machinery"
    )

    # No prior self correction is initially activated. Its first proposal is learned.
    case = situation(18001, mode="correction")
    observer = Observer(case.knowledge, model, self_ref=case.self_ref)
    first = observer.observe(case.observation)
    _require(
        first.proposal.operation is not None and first.proposal != case.target,
        "initial learned choice precedes changed preference",
    )
    feedback = observer.act(evaluate=lambda _: False, correction=case.target)
    corrected = observer.propose()
    _require(
        feedback.succeeded is True and feedback.satisfied is False,
        "preference failure differs from capability success",
    )
    _require(
        corrected.proposal == case.target,
        "explicit correction changes the learned recurrent proposal",
    )
    _require(
        feedback.discrepancies.predictive is not None
        and feedback.discrepancies.preference is not None
        and feedback.discrepancies.capability is not None,
        "three discrepancy channels remain separate",
    )
    learned_result = observer.act(
        evaluate=lambda ref: observer.knowledge.get(ref).fields.get("value") == case.expected_value
    )
    _require(learned_result.satisfied is True, "corrected learned proposal executes exactly")

    # Remove the self evidence and recurrent field together to expose its causal role.
    baseline_observer = Observer(observer.knowledge, model, self_ref=observer.self_ref)
    baseline_decision = baseline_observer.observe(case.observation)
    _require(baseline_decision.proposal != case.target, "correction is not a fixed task rewrite")
    activation_before = observer.state.context.clone()
    steps_before = observer.state.steps
    observer.think()
    _require(
        observer.state.steps > steps_before
        and not torch.equal(activation_before, observer.state.context),
        "further internal processing without a new observation",
    )

    # Install multiplication AFTER training, directly in this live cornerstone.
    parameter_count = sum(p.numel() for p in model.parameters())
    a = observer.knowledge
    number_category = a.get(case.target.arguments[0]).category
    _require(number_category is not None, "numeric argument category")
    assert number_category is not None
    # Use the common declared scalar interface, including newly taught specializations.
    second_category = a.get(case.target.arguments[1]).category
    assert second_category is not None
    second_interfaces = (second_category, *a.ancestors(second_category))
    number_category = next(
        ref
        for ref in (number_category, *a.ancestors(number_category))
        if ref in second_interfaces and "value" in a.schema(ref)
    )
    multiply = a.teach_operation(
        "newly taught multiplication",
        Procedure(
            inputs=(Endpoint(Role.DATA, number_category), Endpoint(Role.DATA, number_category)),
            result=number_category,
            fields={
                "value": binary("mul", field(argument(0), "value"), field(argument(1), "value"))
            },
        ),
    )
    intent = a.create(fields={"kind": "intent", "arity": 2})
    relation = next(node.ref for node in a.structures if node.role == Role.RELATION)
    a.bind(relation, intent, multiply, Evidence("new structured instruction after neural training"))
    new_observation = Observation(
        a.create(fields={"kind": "request", "task": intent, "slots": case.target.arguments}),
        tuple(reversed(case.observation.candidates)),
    )
    # Reset temporary field for a new episode; durable knowledge and parameters remain.
    new_observer = Observer(a, model)
    new_decision = new_observer.observe(new_observation)
    _require(
        new_decision.proposal == Proposal(multiply, case.target.arguments),
        "post-training installed procedure selected/bound by neural pointers",
    )
    left = a.get(case.target.arguments[0]).fields["value"]
    right = a.get(case.target.arguments[1]).fields["value"]
    assert isinstance(left, Fraction) and isinstance(right, Fraction)
    expected_product = left * right
    new_feedback = new_observer.act(
        evaluate=lambda ref: a.get(ref).fields["value"] == expected_product
    )
    _require(new_feedback.satisfied is True, "new operation executes through the unchanged kernel")
    _require(
        sum(p.numel() for p in model.parameters()) == parameter_count,
        "teaching adds no symbol-specific neural parameters",
    )

    # Preserve all three stores, the optimizer and its shuffle RNG.
    observer.think()  # Refresh the inventory after installing additional knowledge.
    observer.save(path, trainer=trainer)
    restored, continued = Observer.restore(path)
    _require(continued is not None, "optimizer restored")
    _require(
        restored.knowledge.dumps() == observer.knowledge.dumps()
        and torch.equal(restored.state.field, observer.state.field)
        and restored.propose() == observer.propose(),
        "exact knowledge/activation/decision restoration",
    )
    next_original, next_restored = observer.think(), restored.think()
    _require(
        next_original == next_restored
        and torch.equal(restored.state.context, observer.state.context),
        "same next recurrent computation after restore",
    )
    assert continued is not None
    old_parameters = {key: value.clone() for key, value in restored.model.state_dict().items()}
    updates_before = continued.updates
    continued.train(material[:32], epochs=1)
    _require(
        continued.updates > updates_before
        and any(
            not torch.equal(value, restored.model.state_dict()[key])
            for key, value in old_parameters.items()
        ),
        "continued training actually updates restored weights",
    )
    restored.think()
    reuse_path = path.with_name(path.stem + "-continued" + path.suffix)
    restored.save(reuse_path, trainer=continued)

    report: dict[str, object] = {
        "torch_version": torch.__version__,
        "device": str(model.device),
        "model_config": {
            "width": model.config.width,
            "refinement_steps": model.config.refinement_steps,
        },
        "parameters": parameter_count,
        "lessons": count,
        "epochs": epochs,
        "optimizer_updates": trainer.updates,
        "parameter_change_l2": change,
        "initial_held_out": before,
        "trained_held_out": after,
        "held_out_cases": 128,
        "new_definition_cases": 48,
        "new_definition_results": novel,
        "final_losses": loss,
        "feedback": {
            "first_operation": str(first.proposal.operation),
            "corrected_operation": str(corrected.proposal.operation),
            "initial_execution_success": feedback.succeeded,
            "initial_preference_satisfied": feedback.satisfied,
            "corrected_result": str(a.get(learned_result.result).fields["value"])
            if learned_result.result
            else None,
            "self_map_changes_decision": baseline_decision.proposal != case.target,
        },
        "newly_installed_multiplication": {
            "result": str(a.get(new_feedback.result).fields["value"])
            if new_feedback.result
            else None,
            "additional_parameters": 0,
        },
        "checkpoint": str(path.resolve()),
        "continued_checkpoint": str(reuse_path.resolve()),
        "restored_next_decision_identical": True,
        "continued_optimizer_updates": continued.updates,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "bounded supervised structural curriculum",
            "explicit structured preference correction",
            "no calibrated open-world uncertainty",
            "no unrestricted continual learning",
            "no sensory grounding or natural-language understanding",
        ],
    }
    report_path = path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=Path, default=Path("artifacts/neural-observer.athanor"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lessons", type=int, default=256)
    parser.add_argument(
        "--resume", type=Path, help="Restore a checkpoint and continue supervised training"
    )
    args = parser.parse_args()
    if args.resume is not None:
        torch.set_num_threads(1)
        observer, trainer = Observer.restore(args.resume)
        if trainer is None:
            trainer = Trainer(observer.model)
        losses = trainer.train(lessons(args.lessons, observer.model.config), epochs=args.epochs)
        if observer.observation is not None:
            observer.think()
        observer.save(args.save, trainer=trainer)
        print(
            json.dumps(
                {
                    "checkpoint": str(args.save.resolve()),
                    "optimizer_updates": trainer.updates,
                    "losses": losses,
                },
                indent=2,
            )
        )
    else:
        print(json.dumps(run_demo(args.save, epochs=args.epochs, count=args.lessons), indent=2))


if __name__ == "__main__":
    main()
