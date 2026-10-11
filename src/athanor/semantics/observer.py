"""Learned semantic activation with explicit evidence vetoes and exact execution."""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from athanor import Assessment, Record, Ref, ValidationError
from athanor.neural import Decision, Feedback, Observation, Observer, Proposal
from athanor.neural.training import Trainer
from athanor.values import as_ref, as_tuple

from .language import Language, Utterance, meaning_evidence


@dataclass(frozen=True, slots=True)
class Interpretation:
    status: str
    decision: Decision
    meanings: tuple[tuple[Ref, float, Assessment], ...]
    selected: Ref | None
    reason: str


class SemanticObserver:
    def __init__(self, observer: Observer, quantity: Ref) -> None:
        self.observer = observer
        self.language = Language(observer.knowledge, quantity)
        self.utterance: Utterance | None = None

    @property
    def previous_result(self) -> Ref | None:
        for ref in reversed(self.observer.self_map):
            node = self.observer.knowledge.get(ref)
            if isinstance(node.fields.get("result"), Ref):
                return as_ref(node.fields["result"])
        return None

    def interpret(
        self, text: str, *, context: Record | None = None, distractors: tuple[Ref, ...] = ()
    ) -> Interpretation:
        utterance = self.language.encode(
            text, context=context, previous_result=self.previous_result, distractors=distractors
        )
        self.utterance = utterance
        self.observer.observe(Observation(utterance.ref, utterance.candidates))
        return self.inspect()

    def inspect(self) -> Interpretation:
        if self.utterance is None or self.observer._frame is None or self.observer._output is None:
            raise ValidationError("Interpret an expression before inspecting its meaning")
        # Pinned utterances preserve what was heard; current evidence can still veto
        # acting on a now-contested or superseded convention.
        self.utterance = self.language.reconstruct(self.utterance.ref, self.utterance.candidates)
        frame, output = self.observer._frame, self.observer._output
        decision = self.observer.propose()
        evidence = meaning_evidence(self.observer.knowledge, self.utterance)
        probabilities = (
            [] if output.meaning_logits is None else output.meaning_logits.softmax(0).tolist()
        )
        alternatives = tuple(
            (
                ref,
                float(probabilities[frame.meanings.index(ref)]) if ref in frame.meanings else 0.0,
                status,
            )
            for ref, status in evidence.items()
        )
        selected = None
        if probabilities:
            index = int(output.meaning_logits.argmax().item())  # type: ignore[union-attr]
            selected = frame.meanings[index] if index < len(frame.meanings) else None
        status, reason = "accepted", "learned pointers agree with supported operational meaning"
        supported = [
            ref for ref, assessment in evidence.items() if assessment == Assessment.SUPPORTED
        ]
        entity_evidence = [
            binding.assessment
            for binding in self.utterance.bindings
            if self.observer.knowledge.get(binding.referent).fields.get("kind") != "concept"
        ]
        competing_entities = any(
            len(as_tuple(self.observer.knowledge.get(ref).fields["referents"])) > 1
            for ref in self.observer.knowledge.get(self.utterance.ref).members
        )
        if self.utterance.unknown:
            status, reason = "unknown", "uninstalled content expression"
        elif any(
            assessment == Assessment.CONFLICT
            for assessment in (*evidence.values(), *entity_evidence)
        ):
            status, reason = "conflict", "contradictory interpretation evidence"
        elif len(supported) > 1 or competing_entities:
            status, reason = "ambiguous", "several context-compatible meanings remain"
        elif any(
            any(key not in self.utterance.context for key in binding.context)
            for binding in self.utterance.bindings
        ):
            status, reason = "unknown", "required convention context has not been supplied"
        elif not supported or any(status != Assessment.SUPPORTED for status in entity_evidence):
            status, reason = "unknown", "no supported operational meaning"
        elif selected != supported[0] or decision.proposal.operation is None:
            status, reason = "unresolved", "neural activation has not selected supported knowledge"
        elif (
            decision.proposal.operation != self.observer.knowledge.get(selected).fields["operation"]
        ):
            status, reason = "unresolved", "concept and executable pointers disagree"
        elif Counter(decision.proposal.arguments) - Counter(self.utterance.mentions) or any(
            ref not in self.utterance.mentions for ref in decision.proposal.arguments
        ):
            status, reason = "insufficient", "proposed inputs are not grounded in this expression"
        if status != "accepted":
            decision = Decision(Proposal(None), decision.confidence, decision.expectation)
        return Interpretation(status, decision, alternatives, selected, reason)

    def act(
        self, *, evaluate: Callable[[Ref], bool] | None = None, correction: Proposal | None = None
    ) -> Feedback:
        interpretation = self.inspect()
        feedback = self.observer.act(
            interpretation.decision, evaluate=evaluate, correction=correction
        )
        self_record = self.observer.knowledge.create(
            fields={
                "kind": "semantic_capability_use",
                "concept": interpretation.selected,
                "recognized": interpretation.status == "accepted",
                "operation": interpretation.decision.proposal.operation,
                "execution_success": feedback.succeeded,
                "confidence": Fraction(f"{interpretation.decision.confidence:.7f}"),
                "observation": self.utterance.ref if self.utterance else None,
            },
            source="measured semantic observer performance",
        )
        self.observer.self_map.append(self_record)
        self.observer.self_map = self.observer.self_map[-self.observer.model.config.history_limit :]
        self.observer.think()
        return feedback

    def refine(self, *, steps: int | None = None) -> Interpretation:
        """Process the same observation further; evidence vetoes remain in force."""
        self.observer.think(steps=steps)
        return self.inspect()

    def sequence(
        self, text: str, *, context: Record | None = None, max_refinements: int = 3
    ) -> tuple[Feedback, ...]:
        clauses = [clause.strip() for clause in text.split(";")]
        if (
            not 1 <= len(clauses) <= 4
            or any(not clause for clause in clauses)
            or type(max_refinements) is not int
            or not 0 <= max_refinements <= 8
        ):
            raise ValidationError("An expression sequence needs 1..4 nonempty clauses")
        results = []
        for clause in clauses:
            interpretation = self.interpret(clause, context=context)
            for _ in range(max_refinements):
                if interpretation.status not in {"unresolved", "insufficient"}:
                    break
                interpretation = self.refine()
            feedback = self.act()
            results.append(feedback)
            if not feedback.succeeded:
                break
        return tuple(results)

    def save(self, path: str | Path, *, trainer: Trainer | None = None) -> None:
        # Context/lexicon/interaction observations are native knowledge; current fields
        # and pending semantic logits use the same versioned neural checkpoint.
        if self.observer.observation is not None:
            self.observer.think()
        self.observer.save(path, trainer=trainer)

    @classmethod
    def restore(
        cls, path: str | Path, quantity: Ref | None = None
    ) -> tuple[SemanticObserver, Trainer | None]:
        observer, trainer = Observer.restore(path)
        if quantity is None:
            anchors = [
                node
                for node in observer.knowledge.structures
                if node.fields.get("kind") == "language_anchor"
            ]
            if len(anchors) != 1:
                raise ValidationError("Checkpoint needs one explicit language quantity anchor")
            quantity = as_ref(anchors[0].fields["quantity"])
        semantic = cls(observer, quantity)
        if (
            observer.observation is not None
            and observer.knowledge.get(observer.observation.ref).fields.get("kind") == "utterance"
        ):
            semantic.utterance = semantic.language.reconstruct(
                observer.observation.ref, observer.observation.candidates
            )
        return semantic, trainer
