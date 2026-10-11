"""Learned proposals reach exact execution; measured self observations reenter the graph."""

from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from athanor import Athanor, AthanorError, Ref, Role, Structure, ValidationError, record

from .contracts import (
    Decision,
    Discrepancies,
    Expectation,
    Feedback,
    ModelOutput,
    Observation,
    Proposal,
    empty_activation,
)
from .encoding import Frame, GraphBuilder
from .model import NeuralObserverModel

if TYPE_CHECKING:
    from .training import Trainer


def outcome_signature(node: Structure, *, updated: bool) -> tuple[float, ...]:
    """Predict structural outcomes only. No target approximates an arithmetic answer."""
    return (
        float(node.role == Role.DATA),
        float(bool(node.members)),
        float(any(isinstance(value, Fraction) for value in node.fields.values())),
        min(len(node.fields), 8) / 8,
        float(updated),
        float(not updated),
    )


def measure(
    knowledge: Athanor,
    decision: Decision,
    *,
    evaluate: Callable[[Ref], bool] | None = None,
    correction: Proposal | None = None,
) -> Feedback:
    """Call the unchanged cornerstone exactly once; preserve distinct discrepancy types."""
    proposal = decision.proposal
    expected = decision.expectation
    result: Ref | None = None
    receipt: int | None = None
    outcome: tuple[float, ...] | None = None
    satisfied: bool | None = None
    error_type: str | None = None
    error_message: str | None = None
    if proposal.operation is None:
        status = "abstained"
    else:
        try:
            result = knowledge.execute(proposal.operation, proposal.arguments)
        except AthanorError as exc:
            status = "rejected"
            error_type, error_message = type(exc).__name__, str(exc)
        else:
            status = "executed"
            inspection = knowledge.inspect(result)
            receipt = inspection.receipt.sequence
            outcome = outcome_signature(
                inspection.structure, updated=inspection.receipt.action == "transform"
            )
            if evaluate is not None:
                satisfied = evaluate(result)
                if type(satisfied) is not bool:
                    raise ValidationError("Preference evaluators must return an explicit boolean")
    discrepancy = Discrepancies(
        None
        if outcome is None
        else tuple(
            actual - predicted for actual, predicted in zip(outcome, expected.outcome, strict=True)
        ),
        None if satisfied is None else float(satisfied) - expected.preference,
        None if status == "abstained" else float(status == "executed") - expected.success,
    )
    return Feedback(
        decision,
        status,
        result,
        receipt,
        error_type,
        error_message,
        outcome,
        satisfied,
        discrepancy,
        correction,
    )


def _exact(value: float) -> Fraction:
    """Attributed sampled neural measurements; checkpoint tensors retain full precision."""
    return Fraction(f"{value:.7f}")


def record_self_observation(
    knowledge: Athanor,
    observer: Ref,
    observation: Observation,
    feedback: Feedback,
    *,
    activation: tuple[float, ...] = (),
    history: tuple[tuple[float, ...], ...] = (),
    attempts: int = 0,
    successes: int = 0,
    steps: int = 0,
) -> Ref:
    def local(ref: Ref | None) -> Ref | None:
        if ref is None:
            return None
        try:
            knowledge.get(ref)
        except ValidationError:
            return None
        return ref

    proposal = feedback.decision.proposal
    correction = feedback.correction
    discrepancies = feedback.discrepancies
    capabilities = tuple(node.ref for node in knowledge.structures if node.role == Role.OPERATION)
    fields = record(
        kind="self_observation",
        observer=observer,
        observation=observation.ref,
        available_capabilities=capabilities,
        operation=local(proposal.operation),
        arguments=tuple(local(ref) for ref in proposal.arguments),
        status=feedback.status,
        success=feedback.succeeded,
        confidence=_exact(feedback.decision.confidence),
        expected_success=_exact(feedback.decision.expectation.success),
        expected_preference=_exact(feedback.decision.expectation.preference),
        expected_outcome=tuple(_exact(value) for value in feedback.decision.expectation.outcome),
        observed_outcome=None
        if feedback.observed_outcome is None
        else tuple(_exact(value) for value in feedback.observed_outcome),
        predictive_discrepancy=None
        if discrepancies.predictive is None
        else tuple(_exact(value) for value in discrepancies.predictive),
        preference_discrepancy=None
        if discrepancies.preference is None
        else _exact(discrepancies.preference),
        capability_discrepancy=None
        if discrepancies.capability is None
        else _exact(discrepancies.capability),
        satisfied=feedback.satisfied,
        correction=None
        if correction is None
        else record(
            operation=local(correction.operation),
            arguments=tuple(local(ref) for ref in correction.arguments),
        ),
        result=feedback.result,
        receipt_sequence=feedback.receipt_sequence,
        error_type=feedback.error_type,
        error_message=feedback.error_message,
        activation=tuple(_exact(value) for value in activation),
        activation_history=tuple(tuple(_exact(value) for value in row) for row in history),
        attempts=attempts,
        successes=successes,
        steps=steps,
    )
    return knowledge.create(
        fields=fields, source="observer: measured execution and explicit feedback"
    )


class Observer:
    def __init__(
        self, knowledge: Athanor, model: NeuralObserverModel, *, self_ref: Ref | None = None
    ) -> None:
        self.knowledge = knowledge
        self.model = model
        self.builder = GraphBuilder(model.config)
        self.active_operations: tuple[Ref, ...] | None = None
        self.self_ref = (
            self_ref
            if self_ref is not None
            else knowledge.create(
                fields={"kind": "observer", "width": model.config.width},
                source="observer initialization",
            )
        )
        knowledge.get(self.self_ref)
        self.state = empty_activation(model.config.width)
        self.observation: Observation | None = None
        self.self_map: list[Ref] = []
        self.activation_history: list[tuple[float, ...]] = []
        self.capabilities: dict[Ref, tuple[int, int]] = {}
        self._frame: Frame | None = None
        self._output: ModelOutput | None = None

    def observe(self, observation: Observation, *, steps: int | None = None) -> Decision:
        # Resolve everything before replacing an existing observation/state.
        self.knowledge.get(observation.ref)
        for ref in observation.candidates:
            self.knowledge.get(ref)
        previous = self.observation
        self.observation = observation
        try:
            return self.think(steps=steps)
        except AthanorError, RuntimeError:
            self.observation = previous
            raise

    def think(self, *, steps: int | None = None) -> Decision:
        if self.observation is None:
            raise ValidationError("Internal refinement needs an existing observation")
        frame = self.builder.build(
            self.knowledge,
            self.observation,
            self_refs=tuple(self.self_map),
            observer=self.self_ref,
            operations=self.active_operations,
        )
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                output = self.model([frame], [self.state], steps=steps)[0]
        finally:
            self.model.train(was_training)
        self._frame, self._output = frame, output
        self.state = output.activation.detached()
        self.activation_history.append(
            tuple(float(value) for value in self.state.context[:8].tolist())
        )
        self.activation_history = self.activation_history[-self.model.config.history_limit :]
        return self.propose()

    def propose(self) -> Decision:
        if self._frame is None or self._output is None:
            raise ValidationError("Process an observation before proposing execution")
        frame, output = self._frame, self._output
        if output.operation_index == len(frame.operations):
            proposal = Proposal(None)
        elif not frame.candidates and output.argument_indices:
            # Learned selection had no offered arguments; preserve insufficiency explicitly.
            # This is an unexecutable proposal rather than selecting an invented reference.
            proposal = Proposal(frame.operations[output.operation_index], ())
        else:
            proposal = Proposal(
                frame.operations[output.operation_index],
                tuple(frame.candidates[index] for index in output.argument_indices),
            )
        confidence = float(output.operation_logits.softmax(0)[output.operation_index].item())
        for logits, index in zip(output.argument_logits, output.argument_indices, strict=True):
            confidence *= float(logits.softmax(0)[index].item())
        return Decision(
            proposal,
            confidence,
            Expectation(
                float(output.success_logit.sigmoid().item()),
                float(output.preference_logit.sigmoid().item()),
                tuple(float(value) for value in output.outcome.tolist()),
            ),
        )

    def act(
        self,
        decision: Decision | None = None,
        *,
        evaluate: Callable[[Ref], bool] | None = None,
        correction: Proposal | None = None,
    ) -> Feedback:
        if self.observation is None:
            raise ValidationError("Execution needs the observation that prompted it")
        chosen = self.propose() if decision is None else decision
        feedback = measure(self.knowledge, chosen, evaluate=evaluate, correction=correction)
        operation = chosen.proposal.operation
        attempts, successes = (
            self.capabilities.get(operation, (0, 0)) if operation is not None else (0, 0)
        )
        if feedback.succeeded is not None and operation is not None:
            attempts += 1
            successes += int(feedback.succeeded)
            self.capabilities[operation] = (attempts, successes)
        summary = self.activation_history[-1] if self.activation_history else ()
        ref = record_self_observation(
            self.knowledge,
            self.self_ref,
            self.observation,
            feedback,
            activation=summary,
            history=tuple(self.activation_history),
            attempts=attempts,
            successes=successes,
            steps=self.state.steps,
        )
        self.self_map.append(ref)
        self.self_map = self.self_map[-self.model.config.history_limit :]
        # Incorporate feedback through the same learned process without a new input.
        self.think(steps=self.model.config.refinement_steps)
        return feedback

    def save(self, path: str | Path, *, trainer: Trainer | None = None) -> None:
        from .checkpoint import save_checkpoint

        save_checkpoint(self, Path(path), trainer=trainer)

    @classmethod
    def restore(cls, path: str | Path) -> tuple[Observer, Trainer | None]:
        from .checkpoint import restore_checkpoint

        return restore_checkpoint(Path(path))
