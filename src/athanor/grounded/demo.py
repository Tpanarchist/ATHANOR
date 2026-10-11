"""Train and operate one grounded observer through a continuous real-world scenario."""

import argparse
import json
import time
from fractions import Fraction
from pathlib import Path
from typing import cast

import torch
from torch import nn

from athanor import ValidationError
from athanor.neural import Observer
from athanor.neural.curriculum import lessons as structural_lessons
from athanor.neural.curriculum import situation
from athanor.neural.training import evaluate as evaluate_structural
from athanor.semantics.curriculum import case
from athanor.semantics.curriculum import lessons as semantic_lessons
from athanor.semantics.evaluation import evaluate as evaluate_semantics
from athanor.values import as_ref, as_tuple
from athanor.world import World
from athanor.world.geometry import distance

from .checkpoint import restore, save
from .runtime import GroundedObserver, Runtime
from .training import (
    GroundedTrainer,
    continuous_lessons,
    embodied_lessons,
    evaluate_actions,
    evaluate_vision,
    train_vision,
    visual_lessons,
)


def require(condition: bool, description: str) -> None:
    if not condition:
        raise AssertionError(description)


def demonstrate(
    observer: GroundedObserver, trainer: GroundedTrainer, path: Path
) -> dict[str, object]:
    world = World(namespace="acceptance-territory")
    beacon = world.add_object((0.35, 0, 3), color=(225, 45, 35), movable=True)
    world.add_object((-1.5, 0, 3.6), color=(40, 195, 75), velocity=(0.5, 0, 0))
    runtime = Runtime(world, observer)
    trace: list[dict[str, object]] = []
    require(
        len(observer.last_percepts) >= 2,
        "raw stereo distinguishes two unfamiliar scene participants",
    )
    identities = {t.ref.id for t in observer.tracks}
    visual_index = max(
        range(len(observer.last_percepts)),
        key=lambda i: (
            observer.last_percepts[i].appearance[0] - observer.last_percepts[i].appearance[1]
        ),
    )
    weights = {k: v.clone() for k, v in observer.observer.model.state_dict().items()}
    concept = observer.teach_visible("beacon", visual_index)
    require(
        all(torch.equal(v, observer.observer.model.state_dict()[k]) for k, v in weights.items()),
        "a new grounded expression allocates no neural training or network",
    )
    observer.instruct("approach beacon")
    observer.observe(world.sense())
    old_distance = distance(world.position, (0.35, 0, 3))
    for _ in range(3):
        trace.append(runtime.step())
    initial_progress = old_distance - distance(world.position, (0.35, 0, 3))
    require(
        distance(world.position, (0.35, 0, 3)) < old_distance - 0.2,
        "learned policy uses newly taught concept to approach",
    )
    learning_transition = runtime.last_transition
    assert observer.last_feedback is not None
    measured_preference = cast(
        Fraction | None,
        observer.knowledge.get(observer.last_feedback).fields.get("preference_discrepancy"),
    )
    require(
        bool(identities & {t.ref.id for t in observer.tracks}),
        "perceptual identity hypothesis survives viewpoint change",
    )
    target = observer._target_percept(observer.last_percepts)
    require(target is not None, "taught appearance is recognized after moving")
    visible_tracks = [
        t for t in observer.tracks if t.relative is not None and t.status == "visible"
    ]
    require(bool(visible_tracks), "grounded relation has measured inputs")
    near_operation = as_ref(observer.knowledge.get(observer.near).fields["operation"])
    relation = observer.knowledge.execute(near_operation, (visible_tracks[0].ref,))
    require(
        type(observer.knowledge.get(relation).fields["value"]) is bool,
        "exact taught relationship operates on learned map measurements",
    )
    observer.instruct("observe")
    observer.observe(world.sense())
    before_motion = world.knowledge.get(world.objects[1]).fields["position"]
    for _ in range(4):
        trace.append(runtime.step())
    require(
        world.knowledge.get(world.objects[1]).fields["position"] != before_motion,
        "independent object movement has actual consequences",
    )
    require(
        bool(observer.motion_evidence), "world-map compares observed object changes and odometry"
    )
    # Environmental events are scenario setup, not a policy or interpretation oracle.
    body = world.position
    screen = world.add_object(
        ((body[0] + 0.35) / 2, 0, (body[2] + 3) / 2),
        shape="box",
        size=(0.34, 0.7, 0.1),
        color=(70, 70, 90),
    )
    observer.observe(world.sense())
    retained = [t for t in observer.tracks if t.status == "occluded_hypothesis"]
    require(bool(retained), "occlusion retains uncertain last-supported identity")
    require(
        any(t.confidence < 1 for t in retained),
        "unseen existence is a hypothesis, not privileged observation",
    )
    observer.instruct("search")
    observer.observe(world.sense())
    for _ in range(4):
        trace.append(runtime.step())
    observer.instruct("approach beacon")
    observer.observe(world.sense())
    revealed = observer._target_percept(observer.last_percepts)
    require(revealed is not None, "learned evidence-seeking action exposes new camera evidence")
    require(
        bool(identities & {t.ref.id for t in observer.tracks}),
        "reappearance reuses existing identity hypothesis",
    )
    # Actuator fault is hidden behind the limited interface.
    world.motor = False
    observer.instruct("forward")
    observer.observe(world.sense())
    trace.append(runtime.step())
    assert observer.last_feedback is not None
    failed = observer.knowledge.get(observer.last_feedback)
    require(
        failed.fields["capability_success"] is False,
        "self-map measures physical failure separately from kernel execution",
    )
    require(
        failed.fields["uncertain"] is True and len(as_tuple(failed.fields["alternatives"])) == 3,
        "failed translation preserves competing world/self/interface explanations",
    )
    observer.instruct("turn_right")
    observer.observe(world.sense())
    trace.append(runtime.step())
    world.motor = True
    # Transfer to a different arrangement of the same taught kind, without replay.
    current = next(ref for ref in world.objects if ref.id == beacon.id)
    new_position = (world.position[0] - 0.35, 0, min(5.2, world.position[2] + 2.8))
    world.relocate(current, new_position)
    world.relocate(next(ref for ref in world.objects if ref.id == screen.id), (4.5, 0, 4.5))
    observer.instruct("approach beacon")
    observer.observe(world.sense())
    transfer_before = distance(world.position, new_position)
    transfer_start = len(trace)
    for _ in range(8):
        trace.append(runtime.step())
    transfer_progress = transfer_before - distance(world.position, new_position)
    require(
        distance(world.position, new_position) < transfer_before
        and observer._target_percept(observer.last_percepts) is not None
        and any(step["intended"] == "forward" for step in trace[transfer_start:]),
        "acquired visual meaning transfers to another valid arrangement",
    )
    # Learning from actual outcome, with preference kept separate from prediction.
    transition = learning_transition
    require(transition is not None, "last actual transition available for continued learning")
    assert transition is not None
    before, after, action, frame, prior = transition
    physical_layer = cast(nn.Linear, observer.observer.sensory.physical[0])
    updated = physical_layer.weight.detach().clone()
    trainer.learn_transition(
        before,
        after,
        action,
        frame=frame,
        prior=prior,
        preference=float(measured_preference or 0),
    )
    require(
        not torch.equal(updated, physical_layer.weight),
        "measured feedback changes trained neural parameters",
    )
    observer.observer.think()
    save(runtime, path, trainer=trainer)
    restored, continued = restore(path)
    require(
        continued is not None and continued.updates == trainer.updates,
        "optimizer and learned state restore",
    )
    require(
        runtime.world.dumps() == restored.world.dumps()
        and observer.knowledge.dumps() == restored.observer.knowledge.dumps(),
        "territory and acquired map restore exactly",
    )
    require(
        observer.observer.think() == restored.observer.observer.think(),
        "same next shared recurrent computation after restore",
    )
    a, b = runtime.step(), restored.step()
    require(
        a == b and runtime.world.sense() == restored.world.sense(),
        "restored learned behavior and physical continuation match",
    )
    save(restored, path.with_name(path.stem + "-continued" + path.suffix), trainer=continued)
    # Namespaced evidence proves the inference map contains no territory references.
    from athanor.values import references

    require(
        all(
            ref.namespace != world.knowledge.namespace
            for node in observer.knowledge.structures
            for ref in references(node.fields)
        ),
        "no privileged territory handles enter observer knowledge",
    )
    return {
        "concept": str(concept),
        "grounded_relation_result": str(relation),
        "initial_approach_progress": initial_progress,
        "occluded_hypotheses": len(retained),
        "movement_failure_alternatives": list(as_tuple(failed.fields["alternatives"])),
        "transfer_progress": transfer_progress,
        "identity_hypotheses": len(observer.tracks),
        "checkpoint_next_step_identical": True,
        "teacher_required_during_inference": False,
        "continued_updates": trainer.updates,
        "online_preference": float(measured_preference or 0),
        "feedback_learning_tick": after.tick,
        "trace": trace,
        "checkpoint": str(path.resolve()),
    }


def run_demo(
    path: Path,
    *,
    base_checkpoint: Path = Path("artifacts/meaning-observer.athanor"),
    epochs: int = 80,
) -> dict[str, object]:
    torch.set_num_threads(1)
    started = time.perf_counter()
    if not base_checkpoint.is_file():
        raise ValidationError(
            "Goal 003 trained checkpoint required; run athanor-meaning-demo "
            "or supply --base-checkpoint"
        )
    base, _ = Observer.restore(base_checkpoint)
    observer = GroundedObserver(model=base.model)
    visual = visual_lessons(240)
    unseen_visual = visual_lessons(40, seed=9404)
    before_vision = evaluate_vision(observer.observer.sensory, unseen_visual)
    print("Training raw stereo perception…", flush=True)
    vision_losses = train_vision(observer.observer.sensory, visual, epochs=80)
    after_vision = evaluate_vision(observer.observer.sensory, unseen_visual)
    require(after_vision["foreground_iou"] > 0.95, "learned visual grounding on held-out scenes")
    teacher = GroundedObserver(
        model=observer.observer.model, sensory=observer.observer.sensory, namespace="curriculum-map"
    )
    isolated = embodied_lessons(teacher, 400)
    continuous = continuous_lessons(teacher, 160)
    material = isolated + continuous
    held = embodied_lessons(teacher, 100, seed=9406)
    initial_actions = evaluate_actions(observer, held)
    original = {k: v.clone() for k, v in observer.observer.model.state_dict().items()}
    trainer = GroundedTrainer(observer)
    replay = semantic_lessons(160, observer.observer.model.config, seed=31000) + structural_lessons(
        80, observer.observer.model.config
    )
    print("Learning grounded actions with complete structural/semantic replay…", flush=True)
    losses = trainer.train(material, epochs=epochs, replay=replay)
    final_actions = evaluate_actions(observer, held)
    retained = evaluate_semantics(
        observer.observer.model,
        [case(9934 + i, mode=("valid", "alias", "unknown", "missing")[i % 4]) for i in range(40)],
    )
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
    retained_structural = evaluate_structural(
        observer.observer.model,
        [situation(100000 + i, mode=modes[i % len(modes)]) for i in range(128)],
    )
    require(
        final_actions["operation_accuracy"] >= 0.85
        and final_actions["operation_accuracy"] > initial_actions["operation_accuracy"] + 0.5,
        "actual learned action transfer",
    )
    require(
        final_actions["consequence_mse"] < initial_actions["consequence_mse"] * 0.3,
        "consequence predictions improve from execution",
    )
    require(retained["proposal"] >= 0.9, "prior semantic competence is retained")
    require(retained_structural["proposal"] >= 0.9, "prior structured competence is retained")
    require(
        any(
            not torch.equal(v, observer.observer.model.state_dict()[k]) for k, v in original.items()
        ),
        "shared neural parameters really train",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": observer.observer.model.state_dict(),
            "sensory": observer.observer.sensory.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "updates": trainer.updates,
            "rng": trainer.rng.getstate(),
        },
        path.with_name(path.stem + "-parameters.pt"),
    )
    path.with_suffix(".training.json").write_text(
        json.dumps(
            {
                "initial": initial_actions,
                "trained": final_actions,
                "retained": retained,
                "retained_structural": retained_structural,
            },
            indent=2,
        )
    )
    report = {
        "version": "0.4.0",
        "vision_before": before_vision,
        "vision_after": after_vision,
        "initial_actions": initial_actions,
        "trained_actions": final_actions,
        "retained_meaning": retained,
        "retained_structural": retained_structural,
        "vision_losses": vision_losses,
        "losses": losses,
        "optimizer_updates": trainer.updates,
        "curriculum": {
            "visual_scenes": len(visual),
            "isolated_situations": len(isolated),
            "continuous_frames": len(continuous),
            "continuous_frame_budget": 160,
            "epochs": epochs,
            "replay_lessons": len(replay),
        },
        "base_checkpoint": str(base_checkpoint.resolve()),
        **demonstrate(observer, trainer, path),
        "elapsed_seconds": time.perf_counter() - started,
    }
    path.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=Path, default=Path("artifacts/grounded-observer.athanor"))
    parser.add_argument(
        "--base-checkpoint", type=Path, default=Path("artifacts/meaning-observer.athanor")
    )
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.resume is None:
        print(json.dumps(run_demo(args.save, base_checkpoint=args.base_checkpoint), indent=2))
    else:
        runtime, trainer = restore(args.resume)
        trainer = GroundedTrainer(runtime.observer) if trainer is None else trainer
        teacher = GroundedObserver(
            model=runtime.observer.observer.model,
            sensory=runtime.observer.observer.sensory,
            namespace="continued-curriculum",
        )
        trainer.train(embodied_lessons(teacher, 32, seed=trainer.updates + 404), epochs=1)
        runtime.observer.observer.think()
        save(runtime, args.save, trainer=trainer)
        print(
            json.dumps(
                {"optimizer_updates": trainer.updates, "checkpoint": str(args.save.resolve())}
            )
        )


if __name__ == "__main__":
    main()
