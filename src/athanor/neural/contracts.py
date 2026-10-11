"""Neural boundary contracts: knowledge handles, learned expectations, and separate errors."""

from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor

from athanor import Ref, ValidationError

OUTCOME_SIZE = 6


@dataclass(frozen=True, slots=True)
class ModelConfig:
    width: int = 64
    refinement_steps: int = 4
    max_nodes: int = 512
    history_limit: int = 16
    seed: int = 1729

    def __post_init__(self) -> None:
        for value in (self.width, self.refinement_steps, self.max_nodes, self.history_limit):
            if type(value) is not int or value < 1:
                raise ValidationError(
                    "Neural dimensions and resource limits must be positive integers"
                )
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValidationError("Model seed must be a nonnegative 63-bit integer")


@dataclass(frozen=True, slots=True)
class Observation:
    ref: Ref
    candidates: tuple[Ref, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.ref, Ref) or type(self.candidates) is not tuple:
            raise ValidationError("An observation needs an exact reference and candidate tuple")
        if any(not isinstance(ref, Ref) for ref in self.candidates):
            raise ValidationError("Observation candidates must be exact references")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValidationError("Duplicate observation candidates are not allowed")


@dataclass(frozen=True, slots=True)
class Expectation:
    success: float
    preference: float
    outcome: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.outcome) != OUTCOME_SIZE:
            raise ValidationError("Unexpected structural outcome signature size")
        if any(
            not isfinite(v) or not 0 <= v <= 1
            for v in (self.success, self.preference, *self.outcome)
        ):
            raise ValidationError(
                "Neural expectations must be finite probabilities/signatures in [0,1]"
            )


@dataclass(frozen=True, slots=True)
class Proposal:
    operation: Ref | None
    arguments: tuple[Ref, ...] = ()

    def __post_init__(self) -> None:
        if self.operation is not None and not isinstance(self.operation, Ref):
            raise ValidationError("A proposal operation must be an exact reference")
        if type(self.arguments) is not tuple or any(
            not isinstance(ref, Ref) for ref in self.arguments
        ):
            raise ValidationError("Proposal arguments must be an ordered tuple of exact references")
        if self.operation is None and self.arguments:
            raise ValidationError("Abstention cannot carry executable arguments")


@dataclass(frozen=True, slots=True)
class Decision:
    proposal: Proposal
    confidence: float
    expectation: Expectation

    def __post_init__(self) -> None:
        if (
            not isinstance(self.proposal, Proposal)
            or not isinstance(self.expectation, Expectation)
            or not isfinite(self.confidence)
            or not 0 <= self.confidence <= 1
        ):
            raise ValidationError("Invalid neural decision/confidence")


@dataclass(frozen=True, slots=True)
class Discrepancies:
    predictive: tuple[float, ...] | None
    preference: float | None
    capability: float | None


@dataclass(frozen=True, slots=True)
class Feedback:
    decision: Decision
    status: str
    result: Ref | None
    receipt_sequence: int | None
    error_type: str | None
    error_message: str | None
    observed_outcome: tuple[float, ...] | None
    satisfied: bool | None
    discrepancies: Discrepancies
    correction: Proposal | None = None

    @property
    def succeeded(self) -> bool | None:
        if self.status == "abstained":
            return None
        return self.status == "executed"


@dataclass(slots=True)
class Activation:
    refs: tuple[Ref, ...]
    field: Tensor
    context: Tensor
    steps: int = 0

    def detached(self) -> Activation:
        return Activation(
            self.refs, self.field.detach().clone(), self.context.detach().clone(), self.steps
        )


@dataclass(slots=True)
class ModelOutput:
    operation_logits: Tensor
    argument_logits: tuple[Tensor, ...]
    applicability_logits: Tensor
    success_logit: Tensor
    preference_logit: Tensor
    outcome: Tensor
    operation_index: int
    argument_indices: tuple[int, ...]
    activation: Activation
    meaning_logits: Tensor | None = None


def empty_activation(width: int) -> Activation:
    return Activation((), torch.empty((0, width)), torch.zeros(width))
