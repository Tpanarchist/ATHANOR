"""Continuous physical discovery, distinct interface transfer and recursive reuse."""

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import torch

from athanor import Endpoint, FieldSpec, Procedure, Record, Role, record
from athanor.programs import argument, binary, call, field, literal
from athanor.values import as_record, as_ref, as_tuple
from athanor.world import World

from .checkpoint import restore, save
from .processes import project, stored_path
from .runtime import Runtime


def territory() -> World:
    """Set up private native physics; this authority is not given to Processes."""
    world = World(namespace="compound-territory", resolution=64)
    a = world.knowledge
    scalar = a.teach_category("Territory distance", {"value": FieldSpec("number")})
    add, div = [
        a.teach_operation(
            op,
            Procedure(
                (Endpoint(Role.DATA, scalar),) * 2,
                scalar,
                {"value": binary(op, field(argument(0), "value"), field(argument(1), "value"))},
            ),
        )
        for op in ("add", "div")
    ]
    one = a.create(scalar, {"value": 1})
    law = a.teach_operation(
        "Private nonlinear territory response",
        Procedure(
            (Endpoint(Role.DATA, scalar),),
            scalar,
            {"value": field(call(div, argument(0), call(add, argument(0), literal(one))), "value")},
        ),
    )
    world.set_motion_process(law)
    world.add_object((1, 0, 2), size=(0.15,) * 3, shape="box", movable=True)
    return world


class NumericPort:
    """A digital quantity port: ordered packets, distinct from camera depth estimation."""

    def __init__(self, world: World) -> None:
        self._world = world
        self.tick = 0

    def sample(self, value: Fraction, *, repetitions: int = 1) -> tuple[Record, Record]:
        self.tick += 1
        a = self._world.knowledge
        assert self._world.motion_process is not None
        law = self._world.motion_process
        scalar = as_ref(as_record(a.get(law).fields["procedure"])["result"])
        ref = a.create(scalar, {"value": value})
        for _ in range(repetitions):
            ref = a.execute(law, (ref,))
        # Unrelated channels vary independently; selector names provide no answer.
        before = record(channel=(self.tick + 10, value), valid=True, passes=repetitions)
        after = record(readings=(a.get(ref).fields["value"], 100 - self.tick), valid=True)
        return before, after


def run_demo(
    path: Path, *, base_checkpoint: Path = Path("artifacts/discovery-observer.athanor")
) -> dict[str, Any]:
    torch.set_num_threads(1)
    previous, trainer = restore(base_checkpoint)
    parameters = sum(p.numel() for p in previous.observer.observer.model.parameters())
    from athanor.neural.curriculum import situation
    from athanor.neural.training import evaluate as evaluate_structural
    from athanor.semantics.curriculum import case
    from athanor.semantics.evaluation import evaluate as evaluate_semantic

    native_cases = [
        situation(
            51000 + i, mode=("valid", "unsupported", "missing", "incompatible", "correction")[i % 5]
        )
        for i in range(48)
    ]
    meaning_cases = [
        case(
            52000 + i, mode=("valid", "alias", "unknown", "missing", "ambiguous", "context")[i % 6]
        )
        for i in range(48)
    ]
    model = previous.observer.observer.model
    retained_before = {
        "structural": evaluate_structural(model, native_cases),
        "semantic": evaluate_semantic(model, meaning_cases),
    }
    world = territory()
    runtime = Runtime(world, previous.observer, trainer=trainer)
    runtime.observer.goal, runtime.observer.target = "observe", None
    p = runtime.observer.processes
    runtime.start_process_discovery(18)
    trace = []
    operation = None
    for _ in range(18):
        step = runtime.autonomous_step()
        trace.append(step)
        print(
            "Inquiry",
            len(trace),
            step.get("intended"),
            "samples",
            len(p.inputs),
            "excluded",
            p.excluded,
            flush=True,
        )
        operation = p.adopted[-1] if p.adopted else None
        if operation is not None:
            break
        if not step.get("acted"):
            break
    if operation is None:
        print(
            "Best",
            [(c.status, c.errors, c.score, str(c.expression)) for c in p.rank()[:5]],
            flush=True,
        )
        raise AssertionError("Autonomous inquiry did not adopt a supported compound process")
    candidate = next(c for c in p.candidates if c.operation == operation)
    assert candidate.calls >= 2
    assert any(c.status == "qualified" for c in p.candidates)
    acquired_expression = candidate.expression
    original_inputs, original_outputs = list(p.inputs), list(p.outputs)
    initial_parameters = {
        name: value.clone() for name, value in previous.observer.observer.model.state_dict().items()
    }
    # Fresh physical prediction before the next permitted action.
    assert p.last_body_value is not None
    expected, prediction_ref = p.evaluate(operation, p.last_body_value)
    physical_step = runtime.autonomous_step()
    assert physical_step["acted"] and physical_step["intended"] == "forward"
    physical_error = abs(float(expected) - runtime.world.sense().proprio[2])
    assert physical_error < 0.001
    p.apply(operation, p.inputs[-1], expected=p.outputs[-1])
    port = NumericPort(world)
    domain = tuple(Fraction(i, 10) for i in range(1, 10) if i != 5)
    packets = [port.sample(p.choose_probe(domain)) for _ in range(3)]
    mapping = p.correspond(operation, packets, source="ordered digital quantity packets")
    assert mapping is not None
    bound = p.knowledge.get(mapping).fields
    low, high = cast(Fraction, bound["minimum"]), cast(Fraction, bound["maximum"])
    tested = {p.knowledge.get(r).fields["value"] for r in p.probes}
    fresh_value = next(
        low + (high - low) / d for d in range(3, 12) if low + (high - low) / d not in tested
    )
    assert isinstance(fresh_value, Fraction)
    fresh_before, fresh_after = port.sample(fresh_value)
    transferred = p.apply_mapping(mapping, fresh_before, observed=fresh_after)
    assert transferred is not None
    reconstructed = p.materialize_mapping(
        mapping,
        fresh_before,
        record(readings=(0, as_tuple(fresh_after["readings"])[1]), valid=True),
    )
    assert reconstructed is not None and p.knowledge.get(reconstructed).fields == fresh_after
    false_packets = [(before, record(readings=(9, -3), valid=True)) for before, _ in packets]
    assert p.correspond(operation, false_packets, source="unsupported correspondence") is None
    assert (
        p.apply_mapping(mapping, record(channel=(99, Fraction(9, 5)), valid=True, passes=1)) is None
    )
    assert p.apply_mapping(mapping, port.sample(fresh_value, repetitions=2)[0]) is None
    # New task: the digital interface composes its private law twice. No answer enters construction.
    p.start(6, interface="numeric")
    assert trainer is not None
    for _ in range(4):
        v = p.choose_probe(domain)
        before, after = port.sample(v, repetitions=2)
        p.observe(
            project(before, stored_path(bound["input_selector"])),
            project(after, stored_path(bound["output_selector"])),
            source="two-stage digital consequences through learned selectors",
        )
        p.learn(trainer, steps=40)
    larger = p.adopt()
    assert larger is not None
    larger_candidate = next(c for c in p.candidates if c.operation == larger)
    assert operation in larger_candidate.dependencies
    before, after = port.sample(Fraction(1, 2), repetitions=2)
    larger_result = p.apply(
        larger, Fraction(1, 2), expected=project(after, stored_path(bound["output_selector"]))
    )
    assert larger_result is not None and p.knowledge.get(larger_result).fields["value"] == Fraction(
        1, 4
    )
    changed = [
        name
        for name, value in previous.observer.observer.model.state_dict().items()
        if not torch.equal(initial_parameters[name], value)
    ]
    assert (
        changed
        and sum(x.numel() for x in previous.observer.observer.model.parameters()) == parameters
    )
    p.status = "paused"
    runtime.paused = True
    runtime.observer.observe(world.sense())
    save(runtime, path, trainer=trainer)
    restored, resumed_trainer = restore(path)
    assert resumed_trainer is not None
    assert restored.observer.processes.metadata() == p.metadata()
    assert restored.world.dumps() == world.dumps()
    resumed = restored.observer.processes
    # Reuse from persisted structure alone, without replay or the teaching/observation sequence.
    replay_free = resumed.apply(larger, Fraction(1, 2))
    assert replay_free is not None and resumed.knowledge.get(replay_free).fields[
        "value"
    ] == Fraction(1, 4)
    assert resumed.apply_mapping(mapping, fresh_before) is not None
    retained_after = {
        "structural": evaluate_structural(model, native_cases),
        "semantic": evaluate_semantic(model, meaning_cases),
    }
    for family in retained_before:
        assert retained_after[family]["proposal"] >= retained_before[family]["proposal"] - 0.05
    report = {
        "version": "0.6.0",
        "retained_before": retained_before,
        "retained_after": retained_after,
        "hidden_world_inputs": False,
        "teacher_answer_supplied": False,
        "operation": str(operation),
        "expression": str(acquired_expression),
        "trace": trace,
        "inputs": list(map(str, original_inputs)),
        "outputs": list(map(str, original_outputs)),
        "parameter_count": parameters,
        "parameter_count_unchanged": True,
        "shared_parameters_changed": changed,
        "physical_transfer_error": physical_error,
        "physical_prediction": str(prediction_ref),
        "physical_transfer_step": physical_step,
        "correspondence": str(mapping),
        "second_interface": "ordered digital quantity packets",
        "numeric_transfer_exact": p.knowledge.get(reconstructed).fields == fresh_after,
        "numeric_transfer_input": str(fresh_value),
        "numeric_transfer_input_unobserved": fresh_value not in tested,
        "unsupported_correspondence_rejected": True,
        "larger_process": str(larger),
        "larger_expression": str(larger_candidate.expression),
        "recursive_result": str(p.knowledge.get(larger_result).fields["value"]),
        "capabilities": {str(op): values for op, values in p.capabilities.items()},
        "neural_updates": p.neural_updates,
        "checkpoint_restoration_exact": True,
        "restored_reuse_without_replay": True,
        "checkpoint": str(path),
    }
    path.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/process-observer.athanor")
    )
    parser.add_argument(
        "--base-checkpoint", type=Path, default=Path("artifacts/discovery-observer.athanor")
    )
    args = parser.parse_args()
    print(json.dumps(run_demo(args.checkpoint, base_checkpoint=args.base_checkpoint), indent=2))


if __name__ == "__main__":
    main()
