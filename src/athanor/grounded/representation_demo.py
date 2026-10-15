"""Continuous, independently generated Goal 008 learning and restart evidence."""

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch

from athanor import record
from athanor.values import as_ref, as_tuple
from athanor.world.interpretations import Correspondences

from .checkpoint import restore, save


def numbers(*values: int) -> tuple[Fraction, ...]:
    return tuple(Fraction(v) for v in values)


def run_demo(
    path: Path, *, base_checkpoint: Path = Path("artifacts/concept-observer.athanor")
) -> dict[str, Any]:
    torch.set_num_threads(1)
    runtime, trainer = restore(base_checkpoint)
    assert trainer is not None
    observer = runtime.observer
    p = observer.representations
    original = {
        name: value.detach().clone() for name, value in observer.observer.model.named_parameters()
    }
    before_updates = trainer.updates
    old_concepts, old_rules = tuple(observer.induction.concepts), tuple(observer.induction.rules)
    old_numeric = observer.processes.adopted[-1]
    assert observer.processes.evaluate(old_numeric, Fraction(1, 2))[0] == Fraction(1, 4)
    territory = runtime.world.dumps()
    stream = Correspondences(811)
    pairs = tuple(
        (stream.collection((2, 3, 4)), stream.collection((4, 2, 3)), True) for _ in range(3)
    ) + tuple((stream.collection((2, 3, 4)), stream.collection((3, 3, 3)), False) for _ in range(3))
    print("Constructing a representation from anonymous structural correspondences", flush=True)
    relational = p.invent(pairs, trainer, max_candidates=128, beam=16)
    assert relational is not None
    relational_effort = p.search.copy()
    relational_fields = p.knowledge.get(relational).fields
    relational_fragments = tuple(p.fragments)
    fresh = tuple(stream.collection((2, 3, 4)) for _ in range(3))
    concept = p.form_concept(relational, fresh)
    assert p.recognize(concept, stream.collection((2, 3, 4)), expected=True) is True
    assert p.recognize(concept, stream.collection((3, 3, 3)), expected=False) is False
    interpreted = tuple(p.interpret(relational, v) for v in fresh)
    assert all(r is not None for r in interpreted)
    executions = p.search["executed"]
    reused = p.interpret(relational, fresh[0])
    assert reused is not None and p.knowledge.get(reused).fields["reused"] is True
    assert p.search["executed"] == executions

    print("Constructing a pixel interpretation and reusing native procedure fragments", flush=True)
    camera_pairs = tuple(
        (
            p.camera(stream.camera((3, 4, 5), offset=(index, 0))),
            p.camera(stream.camera((5, 3, 4), offset=(index, 2))),
            True,
        )
        for index in range(3)
    ) + tuple(
        (
            p.camera(stream.camera((3, 4, 5), offset=(index, 0))),
            p.camera(stream.camera((4, 4, 4), offset=(index, 1))),
            False,
        )
        for index in range(3)
    )
    perceptual = p.invent(camera_pairs, trainer, max_candidates=96, beam=12)
    assert perceptual is not None
    perceptual_effort = p.search.copy()
    fields = p.knowledge.get(perceptual).fields
    fragment_operations = {p.knowledge.get(f).fields["operation"] for f in relational_fragments}
    shared = fragment_operations.intersection(as_tuple(fields["constituents"]))
    assert shared, "The camera process must actually call the acquired structural fragment"
    camera_observation = p.camera(stream.camera((3, 4, 5), offset=(2, 3), color=(60, 90, 210)))
    camera_interpretation = p.interpret(perceptual, camera_observation)
    assert camera_interpretation is not None
    assert p.relationship(camera_pairs[0][0], camera_observation) is True

    print(
        "Learning a higher-level relationship, retaining ambiguity and selecting a test", flush=True
    )
    normalized = p.invent(
        (
            (numbers(2, 3, 4), numbers(6, 9, 12), True),
            (numbers(3, 5, 7), numbers(6, 10, 14), True),
            (numbers(3, 6, 9), numbers(4, 8, 12), True),
            (numbers(2, 3, 4), numbers(2, 3, 5), False),
            (numbers(3, 5, 7), numbers(3, 5, 8), False),
            (numbers(3, 6, 9), numbers(3, 4, 9), False),
        ),
        trainer,
        max_candidates=32,
        beam=8,
    )
    assert normalized is not None
    assert len(p.alternatives) >= 2
    assert p.relationship(numbers(2, 3, 4), numbers(3, 4, 5)) is None
    alternatives_before = tuple(p.alternatives)
    proposal = p.choose_test(
        ((numbers(2, 3, 4), numbers(3, 4, 5)), (numbers(2, 3, 4), numbers(4, 6, 8)))
    )
    assert proposal is not None
    # The correspondence oracle returns a new observable relation outcome only after
    # the observer has selected the query. No procedure or representation is supplied.
    assert p.raw(proposal[0]) == (Fraction(2), Fraction(3), Fraction(4))
    assert p.raw(proposal[1]) == (Fraction(3), Fraction(4), Fraction(5))
    p.revise(proposal[0], proposal[1], True)
    normalized = p.selected
    assert normalized is not None and p.relationship(numbers(2, 3, 4), numbers(3, 4, 5)) is True
    qualified = [
        p.current(r)
        for r in alternatives_before
        if p.knowledge.get(p.current(r)).fields["status"] == "qualified"
    ]
    assert qualified
    higher_concept = p.form_concept(normalized, tuple(r for r in interpreted if r is not None))
    assert p.recognize(higher_concept, camera_interpretation, expected=True) is True
    composite = p.compose(relational, normalized, fresh)
    assert p.knowledge.get(composite).fields["dependencies"] == (relational, normalized)

    print("Revising an inadequate active interpretation to solve a composed objective", flush=True)
    conditions = (record(kind="equals", value=3),)
    count = next(
        op
        for op in observer.induction.library
        if observer.knowledge.get(op).fields["name"] == "count"
    )
    goal = p.solve(
        stream.collection((2, 3, 4)),
        conditions,
        trainer,
        max_nodes=16,
        max_depth=1,
        operations=(count,),
    )
    assert goal is not None
    task = p.knowledge.get(p.tasks[-1]).fields
    assert p.knowledge.get(as_ref(task["initial_task"])).fields["status"] == "unknown"
    assert task["attempts"] and task["status"] == "supported"
    for _ in range(3):
        value = stream.collection((2, 3, 4))
        assert observer.induction.apply(goal, value, expected=Fraction(3), measured=True) == 3
        goal = observer.induction.rules[-1]
    assert p.knowledge.get(goal).fields["status"] == "supported"
    incomplete = p.camera(stream.camera((3, 4, 5), offset=(-3, 0)))
    assert not p.sufficient(incomplete)
    assert p.relationship(incomplete, incomplete) is None
    assert tuple(observer.induction.concepts) == old_concepts
    assert tuple(observer.induction.rules[: len(old_rules)]) == old_rules
    assert observer.processes.evaluate(old_numeric, Fraction(1, 2))[0] == Fraction(1, 4)
    assert runtime.world.dumps() == territory

    changed = [
        name
        for name, value in observer.observer.model.named_parameters()
        if not torch.equal(original[name], value)
    ]
    assert changed and all(name.startswith("scorer.") for name in changed)
    p.induction.activate(p.raw(incomplete), source="completed continuous representation session")
    parameters = sum(v.numel() for v in observer.observer.model.parameters())
    before_meta = p.metadata()
    optimizer_updates = trainer.updates
    state = observer.observer.state.detached()
    print("Saving full native, neural, optimizer, interpretation and inquiry state", flush=True)
    save(runtime, path, trainer=trainer)
    restored, continued = restore(path)
    assert continued is not None and continued.updates == optimizer_updates
    assert restored.observer.representations.metadata() == before_meta
    assert restored.world.dumps() == territory
    assert torch.equal(restored.observer.observer.state.context, state.context)
    assert torch.equal(restored.observer.observer.state.field, state.field)
    assert all(
        torch.equal(value, dict(restored.observer.observer.model.named_parameters())[name])
        for name, value in observer.observer.model.named_parameters()
    )
    p = restored.observer.representations
    print("Continuing after restoration without replaying discovery examples", flush=True)
    observations_before = len(p.evidence)
    fresh_camera = p.camera(stream.camera((5, 3, 4), offset=(1, 3), color=(210, 150, 55)))
    fresh_interpretation = p.interpret(perceptual, fresh_camera)
    assert fresh_interpretation is not None
    assert p.recognize(higher_concept, fresh_interpretation, expected=True) is True
    assert restored.observer.induction.apply(goal, stream.collection((2, 3, 4))) == 3
    continued_goal = p.solve(
        stream.collection((4, 6, 8)),
        conditions,
        continued,
        max_nodes=16,
        max_depth=1,
        operations=(count,),
    )
    assert continued_goal is not None
    assert len(p.evidence) == observations_before
    assert restored.observer.processes.evaluate(old_numeric, Fraction(1, 2))[0] == Fraction(1, 4)
    p.induction.activate(p.raw(fresh_camera), source="continued native and neural session")
    save(restored, path, trainer=continued)
    report = {
        "version": "0.8.0",
        "base_checkpoint": str(base_checkpoint),
        "restored_prior": {
            "concepts": len(old_concepts),
            "rules": len(old_rules),
            "numeric_result": "1/4",
        },
        "relational_representation": str(relational),
        "relational_constituents": [str(v) for v in as_tuple(relational_fields["constituents"])],
        "relational_effort": relational_effort,
        "perceptual_representation": str(perceptual),
        "perceptual_effort": perceptual_effort,
        "shared_fragment_operations": [str(v) for v in shared],
        "concept": str(concept),
        "higher_concept": str(higher_concept),
        "composite": str(composite),
        "goal_rule": str(goal),
        "continued_goal": str(continued_goal),
        "ambiguity_retained": True,
        "informative_test_selected_before_outcome": True,
        "qualified_alternatives": [str(v) for v in qualified],
        "incomplete_observation_unknown": True,
        "perceptual_transfer": True,
        "appearance_transfer": True,
        "initial_representation_failed": True,
        "revised_representation_solved": True,
        "cache_reuse_without_execution": True,
        "restoration_exact": True,
        "continued_without_replay": True,
        "parameter_count": parameters,
        "changed_parameters": changed,
        "optimizer_before": before_updates,
        "optimizer_after": continued.updates,
        "neural_updates": continued.updates - before_updates,
        "checkpoint": str(path),
        "checkpoint_bytes": path.stat().st_size,
        "native_representations": len(p.representations),
        "concepts": len(p.concepts),
        "fragments": len(p.fragments),
        "supported_fragments": sum(
            p.knowledge.get(f).fields["status"] == "supported" for f in p.fragments
        ),
        "observations": len(p.observations),
        "capability_history": len(restored.observer.induction.capabilities),
        "arc_accessed": False,
        "external_teacher": False,
        "hidden_generator_rule_access": False,
    }
    report_path = path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/representation-observer.athanor")
    )
    parser.add_argument(
        "--base-checkpoint", type=Path, default=Path("artifacts/concept-observer.athanor")
    )
    args = parser.parse_args()
    print(json.dumps(run_demo(args.checkpoint, base_checkpoint=args.base_checkpoint), indent=2))


if __name__ == "__main__":
    main()
