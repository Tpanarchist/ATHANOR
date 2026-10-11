"""One continuous independent relational/raster learning and goal-solving session."""

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch

from athanor import record
from athanor.values import as_ref, as_tuple, exact_equal
from athanor.world.situations import ProblemStream

from .checkpoint import restore, save


def run_demo(
    path: Path, *, base_checkpoint: Path = Path("artifacts/process-observer.athanor")
) -> dict[str, Any]:
    torch.set_num_threads(1)
    runtime, trainer = restore(base_checkpoint)
    assert trainer is not None, "Continue the real trained 0.6 observer and optimizer"
    observer = runtime.observer
    model = observer.observer.model
    inquiry = observer.induction
    old_process = observer.processes.adopted[-1]
    old_result = observer.processes.evaluate(old_process, Fraction(1, 2))[0]
    assert old_result == Fraction(1, 4)
    before_weights = {n: p.detach().clone() for n, p in model.named_parameters()}
    parameter_count = sum(p.numel() for p in model.parameters())
    from athanor.neural.curriculum import situation
    from athanor.neural.training import evaluate as native_evaluate
    from athanor.semantics.curriculum import case
    from athanor.semantics.evaluation import evaluate as semantic_evaluate

    native_cases = [
        situation(
            51000 + i, mode=("valid", "unsupported", "missing", "incompatible", "correction")[i % 5]
        )
        for i in range(48)
    ]
    semantic_cases = [
        case(
            52000 + i, mode=("valid", "alias", "unknown", "missing", "ambiguous", "context")[i % 6]
        )
        for i in range(48)
    ]
    retained_before = {
        "structural": native_evaluate(model, native_cases),
        "semantic": semantic_evaluate(model, semantic_cases),
    }
    relation = ProblemStream(7123, "relational")
    observations = tuple(relation.observe() for _ in range(3))
    print("Inducing concepts and a relational rule from anonymous observations", flush=True)
    relational_rule = inquiry.infer(
        tuple((v, relation.demonstrate(v)) for v in observations),
        trainer,
        max_candidates=1024,
        beam=16,
    )
    assert relational_rule is not None, "Neural proposal needs consistent multi-example support"
    relational_candidates = len(inquiry.candidates)
    relational_qualified = sum(c.status == "qualified" for c in inquiry.candidates)
    rule_fields = inquiry.knowledge.get(relational_rule).fields
    dependencies = set(as_tuple(rule_fields["dependencies"]))
    used_concepts = [
        c for c in inquiry.concepts if inquiry.knowledge.get(c).fields["operation"] in dependencies
    ]
    assert used_concepts, "The rule must use a newly formed operational concept"
    concept = used_concepts[0]
    fresh_domain = tuple(relation.observe() for _ in range(3))
    fresh = inquiry.choose_test(fresh_domain)
    predicted = inquiry.apply(
        relational_rule, fresh, expected=relation.demonstrate(fresh), measured=True
    )
    assert exact_equal(predicted, relation.demonstrate(fresh))
    relational_rule = inquiry.rules[-1]
    recognized = inquiry.recognize(concept, fresh)
    assert recognized is not None
    print("Relational transfer succeeded; acquiring a composite concept", flush=True)
    grouping = next(
        c
        for c in inquiry.concepts
        if inquiry.knowledge.get(c).fields["roles"]
        == (
            record(name="root", type="graph"),
            record(name="group", type="collection_of_bound_indices"),
        )
    )
    composite_concept = inquiry.combine_concepts(concept, grouping, observations)
    assert inquiry.knowledge.get(composite_concept).fields["dependencies"] == (concept, grouping)

    raster = ProblemStream(7341, "raster")
    rasters = tuple(raster.observe() for _ in range(3))
    print("Inferring an unfamiliar transformation from raw symbolic rasters", flush=True)
    spatial_rule = inquiry.infer(
        tuple((v, raster.demonstrate(v)) for v in rasters), trainer, max_candidates=1024, beam=16
    )
    assert spatial_rule is not None
    spatial_candidates = len(inquiry.candidates)
    spatial_fresh = raster.observe()
    spatial_output = inquiry.apply(
        spatial_rule, spatial_fresh, expected=raster.demonstrate(spatial_fresh), measured=True
    )
    assert exact_equal(spatial_output, raster.demonstrate(spatial_fresh))
    spatial_rule = inquiry.rules[-1]

    print("Learning reuse of the relational distinction through a raster interface", flush=True)
    transfer_rule = inquiry.infer(
        tuple((v, relation.demonstrate_raster(v)) for v in rasters),
        trainer,
        max_calls=3,
        max_candidates=2048,
        beam=16,
    )
    assert transfer_rule is not None
    transfer_candidates = len(inquiry.candidates)
    transfer_deps = as_tuple(inquiry.knowledge.get(transfer_rule).fields["dependencies"])
    # At least one pinned learned constituent must occur; the constructor receives
    # examples only, never the authority's hidden transformation or its role names.
    acquired = {inquiry.knowledge.get(c).fields["operation"] for c in [concept, relational_rule]}
    assert any(d in acquired for d in transfer_deps)
    transfer_input = raster.observe()
    transfer_output = inquiry.apply(
        transfer_rule,
        transfer_input,
        expected=relation.demonstrate_raster(transfer_input),
        measured=True,
    )
    assert exact_equal(transfer_output, relation.demonstrate_raster(transfer_input))
    transfer_rule = inquiry.rules[-1]

    symmetric = (("a", None, "a"), (None, None, None), ("a", None, "a"))
    print("Retaining uncertainty for insufficient symmetric demonstrations", flush=True)
    uncertain = inquiry.infer(((symmetric, symmetric),), trainer, max_calls=1, max_candidates=256)
    assert uncertain is None
    fitting_alternatives = sum(max(c.errors) == 0 for c in inquiry.candidates)
    assert fitting_alternatives >= 2

    observer.observe(runtime.world.sense())
    save(runtime, path.with_suffix(".preplan.athanor"), trainer=trainer)
    print("Solving a structural objective with multiple native transformations", flush=True)
    initial = relation.observe()
    constraints = (
        record(kind="concept", reference=concept),
        record(kind="count_concept", reference=concept),
        record(kind="ordered", field=relation.property),
    )
    plan = inquiry.solve(initial, constraints, trainer, max_depth=3, max_nodes=1024)
    assert plan is not None
    plan_spec = inquiry.knowledge.get(plan).fields
    assert len(as_tuple(plan_spec["dependencies"])) >= 2
    plan_output = inquiry.apply(plan, initial)
    assert inquiry.goal_test(as_ref(plan_spec["goal"]), plan_output, initial=initial) is True
    plan_depth = len(as_tuple(plan_spec["dependencies"]))
    plan_dependencies = [str(v) for v in as_tuple(plan_spec["dependencies"])]
    # Generalization is tested on distinct instances against native goal conditions.
    for _ in range(3):
        new_input = relation.observe()
        expected, _ = inquiry.evaluate(as_ref(plan_spec["operation"]), new_input)
        assert inquiry.goal_test(as_ref(plan_spec["goal"]), expected, initial=new_input) is True
        assert exact_equal(
            inquiry.apply(plan, new_input, expected=expected, measured=True), expected
        )
        plan = inquiry.rules[-1]
    assert inquiry.knowledge.get(plan).fields["status"] == "supported"

    retained_after = {
        "structural": native_evaluate(model, native_cases),
        "semantic": semantic_evaluate(model, semantic_cases),
    }
    for key in retained_before:
        assert retained_after[key]["proposal"] >= retained_before[key]["proposal"] - 0.05
    assert observer.processes.evaluate(old_process, Fraction(1, 2))[0] == old_result
    assert sum(p.numel() for p in model.parameters()) == parameter_count
    changed = [n for n, p in model.named_parameters() if not torch.equal(before_weights[n], p)]
    assert changed and all(n.startswith("scorer.") for n in changed)
    runtime.paused = True
    observer.observe(runtime.world.sense())
    save(runtime, path, trainer=trainer)
    continued, continued_trainer = restore(path)
    assert continued_trainer is not None
    assert continued.observer.induction.metadata() == inquiry.metadata()
    assert continued_trainer.updates == trainer.updates
    new = continued.observer.induction
    assert exact_equal(new.apply(spatial_rule, spatial_fresh), spatial_output)
    assert exact_equal(new.apply(transfer_rule, transfer_input), transfer_output)
    restored_input = relation.observe()
    restored_plan = new.apply(plan, restored_input)
    assert (
        new.goal_test(
            as_ref(new.knowledge.get(plan).fields["goal"]), restored_plan, initial=restored_input
        )
        is True
    )
    # A new objective after restoration exercises continued planning, not only replay.
    solved_after_restore = new.solve(
        restored_input, constraints, continued_trainer, max_depth=3, max_nodes=1024
    )
    assert solved_after_restore is not None
    continued.observer.observe(continued.world.sense())
    save(continued, path, trainer=continued_trainer)
    report: dict[str, Any] = {
        "version": "0.7.0",
        "starting_checkpoint": str(base_checkpoint),
        "hidden_rule_supplied": False,
        "external_teacher": False,
        "arc_access": False,
        "concept_count": len(new.concepts),
        "used_concept": str(concept),
        "composite_concept": str(composite_concept),
        "relational_rule": str(relational_rule),
        "relational_candidates": relational_candidates,
        "relational_qualified": relational_qualified,
        "held_out_relational_exact": True,
        "spatial_rule": str(spatial_rule),
        "spatial_candidates": spatial_candidates,
        "held_out_spatial_exact": True,
        "cross_context_rule": str(transfer_rule),
        "cross_context_candidates": transfer_candidates,
        "cross_context_exact": True,
        "cross_context_learned_dependencies": [str(v) for v in transfer_deps],
        "uncertain_alternatives": fitting_alternatives,
        "plan_depth": plan_depth,
        "plan_dependencies": plan_dependencies,
        "goal_achieved": True,
        "generalized_plan": str(plan),
        "solving_after_restore": True,
        "restoration_exact": True,
        "parameter_count": parameter_count,
        "changed_parameters": changed,
        "neural_updates": new.neural_updates,
        "optimizer_updates": continued_trainer.updates,
        "goal006_process_preserved": str(old_result),
        "retained_before": retained_before,
        "retained_after": retained_after,
        "capabilities": new.capabilities,
        "checkpoint": str(path),
    }
    path.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-checkpoint", type=Path, default=Path("artifacts/process-observer.athanor")
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/concept-observer.athanor")
    )
    args = parser.parse_args()
    run_demo(args.checkpoint, base_checkpoint=args.base_checkpoint)


if __name__ == "__main__":
    main()
