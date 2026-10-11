"""Supervised pointer learning with distinct, measured execution feedback losses."""

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor
from torch.nn import functional as F

from athanor import ValidationError

from .contracts import Decision, Expectation, ModelOutput, Proposal
from .curriculum import Lesson, Situation
from .model import NeuralObserverModel
from .observer import measure


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    learning_rate: float = 0.002
    batch_size: int = 16
    seed: int = 321

    def __post_init__(self) -> None:
        if (
            not 0 < self.learning_rate < 1
            or type(self.batch_size) is not int
            or self.batch_size < 1
            or type(self.seed) is not int
            or not 0 <= self.seed < 2**63
        ):
            raise ValidationError("Invalid optimizer configuration")


class Trainer:
    def __init__(self, model: NeuralObserverModel, config: TrainingConfig | None = None) -> None:
        self.model = model
        self.config = TrainingConfig() if config is None else config
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.config.learning_rate, weight_decay=0.0001
        )
        self.rng = random.Random(self.config.seed)
        self.updates = 0

    def train(self, lessons: Sequence[Lesson], *, epochs: int = 1) -> dict[str, float]:
        if not lessons or type(epochs) is not int or epochs < 1:
            raise ValidationError("Training requires lessons and positive epochs")
        self.model.train()
        totals = {
            key: 0.0
            for key in (
                "selection",
                "arguments",
                "applicability",
                "capability",
                "preference",
                "prediction",
                "meaning",
                "loss",
            )
        }
        batches = 0
        for _ in range(epochs):
            order = list(range(len(lessons)))
            self.rng.shuffle(order)
            for start in range(0, len(order), self.config.batch_size):
                batch = [lessons[index] for index in order[start : start + self.config.batch_size]]
                frames = [lesson.frame for lesson in batch]
                with torch.no_grad():
                    previous = cast(
                        tuple[ModelOutput, ...],
                        self.model([lesson.prior or lesson.frame for lesson in batch]),
                    )
                states = [
                    output.activation.detached() if lesson.prior is not None else None
                    for lesson, output in zip(batch, previous, strict=True)
                ]
                outputs = cast(
                    tuple[ModelOutput, ...],
                    self.model(frames, states, forced=[lesson.target for lesson in batch]),
                )
                probes = cast(
                    tuple[ModelOutput, ...],
                    self.model(frames, states, forced=[lesson.probe for lesson in batch]),
                )
                losses: dict[str, list[Tensor]] = {key: [] for key in totals if key != "loss"}
                for lesson, output, probe in zip(batch, outputs, probes, strict=True):
                    frame, target = lesson.frame, lesson.target
                    if output.meaning_logits is not None:
                        meaning_index = (
                            len(frame.meanings)
                            if lesson.meaning is None
                            else frame.meanings.index(lesson.meaning)
                        )
                        losses["meaning"].append(
                            F.cross_entropy(
                                output.meaning_logits.unsqueeze(0),
                                torch.tensor([meaning_index], device=self.model.device),
                            )
                        )
                    op_index = (
                        len(frame.operations)
                        if target.operation is None
                        else frame.operations.index(target.operation)
                    )
                    losses["selection"].append(
                        F.cross_entropy(
                            output.operation_logits.unsqueeze(0),
                            torch.tensor([op_index], device=self.model.device),
                        )
                    )
                    for logits, ref in zip(output.argument_logits, target.arguments, strict=True):
                        losses["arguments"].append(
                            F.cross_entropy(
                                logits.unsqueeze(0),
                                torch.tensor(
                                    [frame.candidates.index(ref)], device=self.model.device
                                ),
                            )
                        )
                    applicability = torch.zeros(len(frame.operations), device=self.model.device)
                    if target.operation is not None:
                        applicability[op_index] = 1
                    losses["applicability"].append(
                        F.binary_cross_entropy_with_logits(
                            output.applicability_logits, applicability
                        )
                    )
                    feedback = lesson.feedback
                    if feedback.succeeded is not None:
                        losses["capability"].append(
                            F.binary_cross_entropy_with_logits(
                                probe.success_logit,
                                torch.tensor(float(feedback.succeeded), device=self.model.device),
                            )
                        )
                    if feedback.satisfied is not None:
                        losses["preference"].append(
                            F.binary_cross_entropy_with_logits(
                                probe.preference_logit,
                                torch.tensor(float(feedback.satisfied), device=self.model.device),
                            )
                        )
                    if feedback.observed_outcome is not None:
                        losses["prediction"].append(
                            F.mse_loss(
                                probe.outcome,
                                torch.tensor(feedback.observed_outcome, device=self.model.device),
                            )
                        )
                means = {
                    key: torch.stack(values).mean()
                    if values
                    else torch.zeros((), device=self.model.device)
                    for key, values in losses.items()
                }
                loss = (
                    means["selection"]
                    + means["arguments"]
                    + 0.3 * means["applicability"]
                    + 0.5 * means["meaning"]
                    + 0.25 * (means["capability"] + means["preference"] + means["prediction"])
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()  # type: ignore[no-untyped-call]  # PyTorch's backward stub is untyped.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                self.optimizer.step()
                self.updates += 1
                batches += 1
                for key, value in means.items():
                    totals[key] += float(value.detach().item())
                totals["loss"] += float(loss.detach().item())
        return {key: value / batches for key, value in totals.items()}


def evaluate(model: NeuralObserverModel, cases: Sequence[Situation]) -> dict[str, float]:
    if not cases:
        raise ValidationError("Evaluation requires held-out situations")
    model.eval()
    counts = {
        key: 0 for key in ("operation", "proposal", "abstention", "executable", "exact_result")
    }
    abstentions = 0
    with torch.no_grad():
        for case in cases:
            frame = case.frame(model.config)
            output = cast(tuple[ModelOutput, ...], model([frame]))[0]
            operation = (
                None
                if output.operation_index == len(frame.operations)
                else frame.operations[output.operation_index]
            )
            proposal = Proposal(
                operation, tuple(frame.candidates[i] for i in output.argument_indices)
            )
            counts["operation"] += int(operation == case.target.operation)
            counts["proposal"] += int(proposal == case.target)
            if case.target.operation is None:
                abstentions += 1
                counts["abstention"] += int(operation is None)
            feedback = measure(
                case.knowledge, Decision(proposal, 0.5, Expectation(0.5, 0.5, (0.5,) * 6))
            )
            counts["executable"] += int(feedback.succeeded is True)
            counts["exact_result"] += int(
                feedback.result is not None
                and case.expected_value is not None
                and case.knowledge.get(feedback.result).fields.get("value") == case.expected_value
            )
    return {
        key: value / (abstentions if key == "abstention" and abstentions else len(cases))
        for key, value in counts.items()
    }
