"""Train bounded meaning, teach new native knowledge, execute and resume locally."""

import argparse
import json
import time
from fractions import Fraction
from pathlib import Path

import torch

from athanor import Assessment, Endpoint, Evidence, FieldSpec, Record, Ref, Value, record
from athanor.neural import NeuralObserverModel, Observer
from athanor.neural.curriculum import Lesson, situation
from athanor.neural.curriculum import lessons as structural_lessons
from athanor.neural.training import Trainer
from athanor.neural.training import evaluate as evaluate_structural
from athanor.programs import argument, binary, call, choose, field, literal

from .curriculum import SemanticCase, case, install, lessons
from .evaluation import evaluate
from .lessons import import_lesson
from .observer import SemanticObserver
from .teaching import Teacher

MODES = (
    "pronoun",
    "valid",
    "valid",
    "correction",
    "alias",
    "context",
    "unknown",
    "missing",
    "ambiguous",
    "valid",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError("Meaning demonstration contract failed: " + message)


def _portable(value: Value) -> object:
    if isinstance(value, Ref):
        return {"ref": [value.id, value.revision, value.namespace]}
    if isinstance(value, Fraction):
        return {"fraction": [value.numerator, value.denominator]}
    if isinstance(value, Record):
        return {key: _portable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_portable(item) for item in value]
    return value


def demonstrate(
    model: NeuralObserverModel, trainer: Trainer, path: Path, material: list[Lesson]
) -> dict[str, object]:
    """Every accepted action comes from learned pointers; expected values only check it."""
    a, refs = install(88888)
    teacher = Teacher(a)
    semantic = SemanticObserver(Observer(a, model, self_ref=refs["self"]), refs["number"])
    interactions: list[dict[str, object]] = []

    def execute(text: str, expected: Fraction | bool, *, context: Record | None = None) -> Ref:
        interpreted = semantic.interpret(text, context=context)
        refinements = 0
        while interpreted.status in {"unresolved", "insufficient"} and refinements < 3:
            interpreted = semantic.refine()
            refinements += 1
        feedback = semantic.act(evaluate=lambda ref: a.get(ref).fields["value"] == expected)
        _require(
            interpreted.status == "accepted" and feedback.satisfied is True,
            f"{text}: {interpreted.status}, expected {expected}, "
            f"actual {None if feedback.result is None else a.get(feedback.result).fields}, "
            f"proposal {interpreted.decision.proposal}",
        )
        assert feedback.result is not None
        interactions.append(
            {
                "expression": text,
                "status": interpreted.status,
                "result": str(a.get(feedback.result).fields["value"]),
                "operation": str(interpreted.decision.proposal.operation),
                "confidence": interpreted.decision.confidence,
                "additional_refinements": refinements,
            }
        )
        _require(
            feedback.discrepancies.predictive is not None
            and feedback.discrepancies.preference is not None
            and feedback.discrepancies.capability is not None,
            "separate measured feedback channels",
        )
        return feedback.result

    execute("subtract 3 from 17", Fraction(14))
    execute("does bird include robin", True)
    execute("is turtle a bird", False)
    unknown = semantic.interpret("gap 19 and 7")
    _require(
        unknown.status == "unknown" and semantic.act().status == "abstained",
        "unknown concept is not invented",
    )
    frozen = {key: value.clone() for key, value in model.state_dict().items()}
    count = sum(parameter.numel() for parameter in model.parameters())
    examples = (a.create(refs["number"], {"value": 7}), a.create(refs["number"], {"value": 19}))
    difference = field(call(refs["op_sub"], argument(0), argument(1)), "value")
    body = choose(
        binary("lt", difference, literal(0)), binary("sub", literal(0), difference), difference
    )
    package = {
        "format": "athanor.lesson.v1",
        "source": "offline structured mathematics teacher",
        "procedures": [
            {
                "name": "separation",
                "definition": "The nonnegative distance between two exact quantities, "
                "built from ordered difference",
                "expressions": ["gap", "separation"],
                "roles": ["first", "second"],
                "procedure": {
                    "inputs": [{"role": "data", "category": _portable(refs["number"])}] * 2,
                    "result": _portable(refs["number"]),
                    "fields": {"value": _portable(body)},
                },
                "examples": [
                    {"arguments": _portable(examples), "expected": {"value": 12}},
                    {"arguments": _portable(tuple(reversed(examples))), "expected": {"value": 12}},
                ],
            }
        ],
    }
    lesson_path = path.with_suffix(".lesson.json")
    lesson_path.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    lesson_input = path.with_suffix(".before-teaching.json")
    a.save(lesson_input)
    installed = import_lesson(a, json.dumps(package))
    execute("gap 19 and 7", Fraction(12))
    execute("separation 5 and 24", Fraction(19))
    execute("add it and 2", Fraction(21))

    finch = teacher.specialize(
        "Finch", refs["bird"], {"color": FieldSpec("string")}, expressions=("finch",)
    )
    wren, witness = teacher.observe(
        finch,
        record(tag="new specimen", wings=2, color="red"),
        expression="wren",
        source="new local specimen record",
    )
    execute("is wren an animal", True)
    execute("does finch include wren", True)
    _require(
        refs["animal"] in a.ancestors(finch),
        "new specialization retains general category without copied relationships",
    )

    teacher.alias("balance", refs["sub"], context=record(domain="difference"))
    teacher.alias("balance", refs["div"], context=record(domain="ratio"))
    ambiguous = semantic.interpret("balance 19 and 7")
    _require(
        ambiguous.status == "ambiguous" and len(ambiguous.meanings) == 2,
        "competing meanings remain explicit",
    )
    field_before = semantic.observer.state.context.clone()
    _require(semantic.act().status == "abstained", "ambiguous expression cannot execute")
    execute("balance 19 and 7", Fraction(12), context=record(domain="difference"))
    _require(
        not torch.equal(field_before, semantic.observer.state.context),
        "additional context changes activation",
    )

    wrong = teacher.alias("delta", refs["add"])
    execute("delta 19 and 7", Fraction(26))
    correction = teacher.correct(
        "delta", (wrong,), refs["sub"], reason="Delta here is the ordered difference, not the sum"
    )
    execute("delta 19 and 7", Fraction(12))
    _require(
        a.get(wrong).fields["referent"] == refs["add"],
        "correction preserves superseded convention and provenance",
    )
    contested_binding = teacher.alias("contested", refs["sub"])
    contested_symbol = teacher.lexicon.find("contested")
    assert contested_symbol is not None
    a.bind(
        teacher.lexicon.relation,
        contested_symbol,
        contested_binding,
        Evidence("independent disagreement", "negative"),
    )
    contested = semantic.interpret("contested 19 and 7")
    _require(
        contested.status == "conflict" and semantic.act().status == "abstained",
        "contested interpretation remains unexecuted",
    )

    can_fly = a.teach_relation("can fly", Endpoint(), Endpoint())
    flight = a.create(fields={"kind": "observed behavior", "name": "flight"})
    teacher.claim(can_fly, refs["bird"], flight, source="teacher assertion")
    _require(
        a.assess(refs["bird"], can_fly, flight) == Assessment.UNKNOWN,
        "teacher assertion is not empirical proof",
    )
    flight_observation = teacher.observe_relation(
        can_fly, wren, flight, source="local flight observation"
    )
    teacher.claim(
        can_fly, wren, flight, source="measured local behavior", observation=flight_observation
    )
    _require(
        a.assess(wren, can_fly, flight) == Assessment.SUPPORTED
        and a.assess(refs["bird"], can_fly, flight) == Assessment.UNKNOWN,
        "observation scope is not generalized without evidence",
    )
    _require(
        all(torch.equal(value, model.state_dict()[key]) for key, value in frozen.items())
        and sum(parameter.numel() for parameter in model.parameters()) == count,
        "new procedure, aliases, corrections and specialization use unchanged weights",
    )

    # A real composition across clauses: its output participates in another proposal.
    composed = semantic.sequence("difference 17 and 3; add it and 2")
    composition_results = [
        (feedback.status, None if feedback.result is None else a.get(feedback.result).fields)
        for feedback in composed
    ]
    _require(
        len(composed) == 2
        and all(feedback.succeeded for feedback in composed)
        and composed[-1].result is not None
        and a.get(composed[-1].result).fields["value"] == 16,
        f"semantic composition reuses the prior result: {composition_results}",
    )
    semantic.save(path, trainer=trainer)
    restored, continued = SemanticObserver.restore(path)
    _require(
        continued is not None
        and restored.inspect() == semantic.inspect()
        and restored.observer.knowledge.dumps() == a.dumps()
        and torch.equal(restored.observer.state.field, semantic.observer.state.field),
        "knowledge, meanings, pending interpretation and state restore exactly",
    )
    _require(
        semantic.observer.think() == restored.observer.think()
        and torch.equal(semantic.observer.state.context, restored.observer.state.context),
        "identical next recurrent step",
    )
    reused = restored.interpret("gap 31 and 9")
    reuse_feedback = restored.act(
        evaluate=lambda ref: restored.observer.knowledge.get(ref).fields["value"] == 22
    )
    _require(
        reused.status == "accepted" and reuse_feedback.satisfied is True,
        "acquired concept independently reused after restore",
    )
    assert continued is not None
    updates_before = continued.updates
    weights_before = {
        key: value.clone() for key, value in restored.observer.model.state_dict().items()
    }
    continued.train(material[:32], epochs=1)
    _require(
        continued.updates > updates_before
        and any(
            not torch.equal(value, restored.observer.model.state_dict()[key])
            for key, value in weights_before.items()
        ),
        "continued training changes restored parameters",
    )
    continued_path = path.with_name(path.stem + "-continued" + path.suffix)
    restored.save(continued_path, trainer=continued)
    a.save(path.with_suffix(".knowledge.json"))
    return {
        "interactions": interactions,
        "new_concept": str(installed["separation"]),
        "new_operation": str(installed["separation.operation"]),
        "new_specimen": str(wren),
        "grounding_witness": str(witness),
        "correction": str(correction),
        "unknown_status": unknown.status,
        "ambiguous_status": ambiguous.status,
        "conflict_status": contested.status,
        "weights_unchanged_during_teaching": True,
        "additional_parameters": 0,
        "self_capability_measurements": [
            {"operation": str(ref), "attempts": attempts, "successes": successes}
            for ref, (attempts, successes) in semantic.observer.capabilities.items()
        ],
        "restored_next_decision_identical": True,
        "restored_new_concept_result": "22",
        "continued_optimizer_updates": continued.updates,
        "checkpoint": str(path.resolve()),
        "continued_checkpoint": str(continued_path.resolve()),
        "offline_lesson": str(lesson_path.resolve()),
        "lesson_input_knowledge": str(lesson_input.resolve()),
    }


def run_demo(
    path: Path, *, epochs: int = 40, count: int = 160, base_checkpoint: Path | None = None
) -> dict[str, object]:
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if base_checkpoint is None:
        # Reproduce the completed Goal 002 prerequisite with the same architecture.
        model = NeuralObserverModel()
        trainer = Trainer(model)
        base_material = structural_lessons(256, model.config)
        for block in range(8):
            trainer.train(base_material, epochs=5)
            print(f"Established cornerstone competence {(block + 1) * 5}/40 passes", flush=True)
    else:
        base_observer, base_trainer = Observer.restore(base_checkpoint)
        model = base_observer.model
        trainer = Trainer(model) if base_trainer is None else base_trainer
    prerequisite_updates = trainer.updates
    material = lessons(count, model.config) + structural_lessons(80, model.config)

    def held_out() -> list[SemanticCase]:
        return [case(90000 + i, mode=MODES[i % len(MODES)]) for i in range(100)]

    before = evaluate(model, held_out())
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    completed = 0
    losses = {}
    while completed < epochs:
        block = min(5, epochs - completed)
        losses = trainer.train(material, epochs=block)
        completed += block
        print(
            f"Taught {completed}/{epochs} passes; "
            f"learned pointer loss {losses['selection'] + losses['arguments']:.4f}",
            flush=True,
        )
    after = evaluate(model, held_out())
    legacy_modes = (
        "valid",
        "valid",
        "valid",
        "valid",
        "unsupported",
        "missing",
        "incompatible",
        "correction",
    )
    retention = evaluate_structural(
        model, [situation(100000 + i, mode=legacy_modes[i % len(legacy_modes)]) for i in range(128)]
    )
    _require(
        retention["proposal"] >= 0.90 and retention["abstention"] >= 0.90,
        "retain Goal 002 structural competence",
    )
    training_knowledge, training_refs = install(88887)
    trained_observer = Observer(training_knowledge, model, self_ref=training_refs["self"])
    trained_observer.save(path.with_name(path.stem + "-trained" + path.suffix), trainer=trainer)
    change = (
        sum(
            float((model.state_dict()[key] - value).square().sum())
            for key, value in initial.items()
        )
        ** 0.5
    )
    _require(
        change > 0
        and after["proposal"] >= 0.90
        and after["proposal"] > before["proposal"] + 0.40
        and after["meaning"] >= 0.95
        and after["abstention"] >= 0.95,
        "actual learned meaning and ordered proposal transfer",
    )
    integrated = demonstrate(model, trainer, path, material)
    report = {
        "version": "0.3.0",
        "torch_version": torch.__version__,
        "device": str(model.device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "semantic_cases": count,
        "fresh_and_active_lessons": count * 2,
        "structural_replay_lessons": 80,
        "epochs": epochs,
        "optimizer_updates": trainer.updates,
        "prerequisite_optimizer_updates": prerequisite_updates,
        "semantic_optimizer_updates": trainer.updates - prerequisite_updates,
        "base_checkpoint": None if base_checkpoint is None else str(base_checkpoint.resolve()),
        "parameter_change_l2": change,
        "held_out_cases": 100,
        "initial_held_out": before,
        "trained_held_out": after,
        "retained_structural_held_out": retention,
        "final_losses": losses,
        **integrated,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "bounded supervised language and grammar",
            "explicit structured definitions and attributed observations",
            "uncalibrated neural confidence",
            "exact examples are checks, not universal mathematical proofs",
            "no learned sensory recognition or open-domain semantics",
            "no unrestricted continual learning",
        ],
    }
    path.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=Path, default=Path("artifacts/meaning-observer.athanor"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lessons", type=int, default=160)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        help="Extend an existing Goal 002 checkpoint instead of reproducing prerequisite training",
    )
    args = parser.parse_args()
    if args.resume is None:
        print(
            json.dumps(
                run_demo(
                    args.save,
                    epochs=args.epochs,
                    count=args.lessons,
                    base_checkpoint=args.base_checkpoint,
                ),
                indent=2,
            )
        )
    else:
        torch.set_num_threads(1)
        observer, trainer = SemanticObserver.restore(args.resume)
        trainer = Trainer(observer.observer.model) if trainer is None else trainer
        material = lessons(args.lessons, trainer.model.config) + structural_lessons(
            80, trainer.model.config
        )
        losses = trainer.train(material, epochs=args.epochs)
        args.save.parent.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    main()
