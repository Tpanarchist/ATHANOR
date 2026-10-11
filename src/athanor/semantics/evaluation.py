"""Held-out structural language checks of learned pointers before evidence vetoes."""

from collections.abc import Sequence

import torch

from athanor.neural import NeuralObserverModel, Observation, Proposal
from athanor.neural.encoding import GraphBuilder

from .curriculum import SemanticCase


def evaluate(model: NeuralObserverModel, cases: Sequence[SemanticCase]) -> dict[str, float]:
    if not cases:
        raise ValueError("Evaluation requires held-out cases")
    counts = dict.fromkeys(("operation", "proposal", "meaning", "abstention"), 0)
    abstentions = 0
    noncommutative = 0
    ordered_correct = 0
    families = {"mathematical": [0, 0], "categorical": [0, 0]}
    model.eval()
    with torch.no_grad():
        for case in cases:
            frame = GraphBuilder(model.config).build(
                case.knowledge,
                Observation(case.utterance.ref, case.utterance.candidates),
                observer=case.refs["self"],
            )
            output = model([frame])[0]
            operation = (
                frame.operations[output.operation_index]
                if output.operation_index < len(frame.operations)
                else None
            )
            proposal = Proposal(
                operation, tuple(frame.candidates[i] for i in output.argument_indices)
            )
            index = (
                int(output.meaning_logits.argmax().item())
                if output.meaning_logits is not None
                else len(frame.meanings)
            )
            meaning = frame.meanings[index] if index < len(frame.meanings) else None
            counts["operation"] += operation == case.target.operation
            counts["proposal"] += proposal == case.target
            counts["meaning"] += meaning == case.meaning
            if case.target.operation is not None:
                families[case.family][0] += int(proposal == case.target)
                families[case.family][1] += 1
                if case.target.operation in (case.refs["op_sub"], case.refs["op_div"]):
                    noncommutative += 1
                    ordered_correct += int(proposal == case.target)
            if case.target.operation is None:
                abstentions += 1
                counts["abstention"] += operation is None
    result = {
        key: value / (abstentions if key == "abstention" and abstentions else len(cases))
        for key, value in counts.items()
    }
    result["ordered_noncommutative"] = ordered_correct / max(noncommutative, 1)
    for family, (correct, total) in families.items():
        result[family + "_proposal"] = correct / max(total, 1)
    return result
