"""Semantic evidence boundary, shared learned machinery and checkpoint contracts."""

import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from athanor import Assessment, Evidence, PersistenceError, record

AVAILABLE = importlib.util.find_spec("torch") is not None
if AVAILABLE:
    import torch

    from athanor.neural import ModelConfig, NeuralObserverModel, Observer
    from athanor.neural.curriculum import situation
    from athanor.neural.training import Trainer, TrainingConfig
    from athanor.semantics import Teacher
    from athanor.semantics.curriculum import install, lessons
    from athanor.semantics.demo import run_demo
    from athanor.semantics.observer import SemanticObserver


@unittest.skipUnless(AVAILABLE, "Install the neural extra for semantic activation")
class SemanticNeuralContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.a, self.refs = install(5412)
        self.model = NeuralObserverModel(ModelConfig(width=24))
        self.semantic = SemanticObserver(Observer(self.a, self.model), self.refs["number"])
        self.teacher = Teacher(self.a)

    def test_new_aliases_allocate_no_parameters_and_concept_scoring_is_shared(self):
        parameters = {key: value.clone() for key, value in self.model.state_dict().items()}
        self.teacher.alias("dax", self.refs["sub"])
        self.semantic.interpret("dax 19 and 7")
        frame, output = self.semantic.observer._frame, self.semantic.observer._output
        self.assertEqual(output.meaning_logits.shape, (len(frame.meanings) + 1,))
        self.assertTrue(
            all(
                torch.equal(value, self.model.state_dict()[key])
                for key, value in parameters.items()
            )
        )
        self.assertFalse(any("meaning" in key or "semantic" in key for key in parameters))

    def test_unknown_ambiguous_and_conflicting_meanings_veto_execution(self):
        self.teacher.alias("balance", self.refs["sub"], context=record(domain="difference"))
        self.teacher.alias("balance", self.refs["div"], context=record(domain="ratio"))
        ambiguous = self.semantic.interpret("balance 19 and 7")
        self.assertEqual(ambiguous.status, "ambiguous")
        self.assertEqual(len(ambiguous.meanings), 2)
        self.assertIsNone(ambiguous.decision.proposal.operation)
        self.assertEqual(self.semantic.act().status, "abstained")
        contextual = self.semantic.interpret("balance 19 and 7", context=record(domain="ratio"))
        self.assertEqual(len(contextual.meanings), 1)
        self.assertNotEqual(contextual.status, "ambiguous")
        self.assertEqual(self.semantic.interpret("unsupported 19 and 7").status, "unknown")
        binding = self.teacher.alias("contested", self.refs["sub"])
        self.a.bind(
            self.teacher.lexicon.relation,
            self.teacher.lexicon.find("contested"),
            binding,
            Evidence("negative instructor", "negative"),
        )
        contested = self.semantic.interpret("contested 19 and 7")
        self.assertEqual(contested.status, "conflict")
        self.assertEqual(contested.meanings[0][2], Assessment.CONFLICT)

    def test_competing_or_uncertain_entity_referents_do_not_become_certain_arguments(self):
        self.teacher.alias("specimen", self.refs["robin"])
        self.teacher.alias("specimen", self.refs["turtle"])
        self.assertEqual(self.semantic.interpret("is specimen a bird").status, "ambiguous")
        self.teacher.alias(
            "unverified", self.refs["robin"], evidence=Evidence("teacher guess", "uncertain")
        )
        self.assertEqual(self.semantic.interpret("is unverified a bird").status, "unknown")

    def test_single_scoped_meaning_requires_its_context_before_execution(self):
        self.teacher.alias("local", self.refs["sub"], context=record(domain="local mathematics"))
        interpreted = self.semantic.interpret("local 19 and 7")
        self.assertEqual(interpreted.status, "unknown")
        self.assertIsNone(interpreted.decision.proposal.operation)

    def test_new_conflict_or_supersession_vetoes_a_pending_interpretation(self):
        binding = self.teacher.alias("pending", self.refs["sub"])
        self.semantic.interpret("pending 19 and 7")
        symbol = self.teacher.lexicon.find("pending")
        self.a.bind(
            self.teacher.lexicon.relation,
            symbol,
            binding,
            Evidence("later disagreement", "negative"),
        )
        self.assertEqual(self.semantic.inspect().status, "conflict")
        self.assertEqual(self.semantic.act().status, "abstained")
        self.teacher.correct(
            "pending", (binding,), self.refs["add"], reason="adopt this replacement"
        )
        self.assertEqual(self.semantic.inspect().status, "unknown")
        self.assertIsNone(self.semantic.inspect().decision.proposal.operation)

    def test_grounded_input_multiplicity_vetoes_reusing_one_occurrence(self):
        self.semantic.interpret("difference 19 and 7")
        frame, output = self.semantic.observer._frame, self.semantic.observer._output
        operation_index = frame.operations.index(self.refs["op_sub"])
        output.operation_index = operation_index
        output.operation_logits.fill_(-10)
        output.operation_logits[operation_index] = 10
        output.meaning_logits.fill_(-10)
        output.meaning_logits[frame.meanings.index(self.refs["sub"])] = 10
        index = frame.candidates.index(self.semantic.utterance.mentions[0])
        output.argument_indices = (index, index)
        output.argument_logits = tuple(torch.zeros(len(frame.candidates)) for _ in range(2))
        # The boundary checks the explicit occurrences, not merely valid types.
        self.assertEqual(self.semantic.inspect().status, "insufficient")
        self.assertIsNone(self.semantic.inspect().decision.proposal.operation)

    def test_further_refinement_keeps_evidence_vetoes_and_changes_activation(self):
        self.semantic.interpret("unsupported 19 and 7")
        before = self.semantic.observer.state.context.clone()
        steps = self.semantic.observer.state.steps
        result = self.semantic.refine()
        self.assertEqual(result.status, "unknown")
        self.assertGreater(self.semantic.observer.state.steps, steps)
        self.assertFalse(torch.equal(before, self.semantic.observer.state.context))

    def test_installation_and_observer_recognition_are_different_self_records(self):
        installations = [
            node
            for node in self.a.structures
            if node.fields.get("kind") == "capability_installation"
        ]
        self.assertTrue(
            all(
                node.fields["installed"] and not node.fields["observer_success_demonstrated"]
                for node in installations
            )
        )
        self.semantic.interpret("difference 19 and 7")
        feedback = self.semantic.act()
        self.assertTrue(
            any(
                self.a.get(ref).fields.get("kind") == "semantic_capability_use"
                for ref in self.semantic.observer.self_map
            )
        )
        measured = self.a.get(self.semantic.observer.self_map[-1])
        self.assertEqual(measured.fields["execution_success"], feedback.succeeded)
        self.assertIn(measured.ref, self.semantic.observer.state.refs)

    def test_semantic_checkpoint_preserves_meanings_pending_interpretation_and_next_step(self):
        self.semantic.interpret("subtract 7 from 19")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meaning.athanor"
            self.semantic.save(path)
            restored, trainer = SemanticObserver.restore(path)
            self.assertIsNone(trainer)
            self.assertEqual(self.semantic.inspect(), restored.inspect())
            self.assertEqual(self.a.dumps(), restored.observer.knowledge.dumps())
            self.assertEqual(self.semantic.observer.think(), restored.observer.think())
            self.assertTrue(
                torch.equal(self.semantic.observer.state.context, restored.observer.state.context)
            )

    def test_fresh_semantic_observer_checkpoint_can_restore_before_first_instruction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fresh.athanor"
            self.semantic.save(path)
            restored, trainer = SemanticObserver.restore(path)
            self.assertIsNone(trainer)
            self.assertIsNone(restored.utterance)
            self.assertIsNone(restored.observer.observation)
            self.assertEqual(restored.observer.knowledge.dumps(), self.a.dumps())

    def test_semantic_training_changes_parameters_and_optimizer_continues_identically(self):
        material = lessons(8, self.model.config)
        self.assertTrue(
            any(
                lesson.prior is not None
                and lesson.prior.refs[lesson.prior.query_index]
                != lesson.frame.refs[lesson.frame.query_index]
                for lesson in material
            )
        )
        trainer = Trainer(self.model, TrainingConfig(batch_size=8))
        initial = {key: value.clone() for key, value in self.model.state_dict().items()}
        losses = trainer.train(material)
        self.assertGreater(losses["meaning"], 0)
        self.assertTrue(
            any(
                not torch.equal(value, self.model.state_dict()[key])
                for key, value in initial.items()
            )
        )
        self.semantic.interpret("difference 19 and 7")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meaning.athanor"
            self.semantic.save(path, trainer=trainer)
            restored, continued = SemanticObserver.restore(path)
            trainer.train(material)
            continued.train(material)
            self.assertEqual(trainer.updates, continued.updates)
            self.assertTrue(
                all(
                    torch.equal(value, restored.observer.model.state_dict()[key])
                    for key, value in self.model.state_dict().items()
                )
            )

    def rewrite(self, path, change):
        with zipfile.ZipFile(path) as archive:
            content = {name: archive.read(name) for name in archive.namelist()}
        metadata = json.loads(content["metadata.json"])
        payload = torch.load(io.BytesIO(content["tensors.pt"]), weights_only=True)
        change(metadata, payload)
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        content["tensors.pt"] = buffer.getvalue()
        metadata["hashes"]["tensors.pt"] = hashlib.sha256(content["tensors.pt"]).hexdigest()
        content["metadata.json"] = json.dumps(metadata).encode()
        with zipfile.ZipFile(path, "w") as archive:
            for name, value in content.items():
                archive.writestr(name, value)

    def test_version_one_neural_checkpoint_remains_restorable(self):
        case = situation(300)
        observer = Observer(case.knowledge, self.model, self_ref=case.self_ref)
        observer.observe(case.observation)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.athanor"
            observer.save(path)

            def downgrade(metadata, payload):
                metadata["version"] = 1
                del payload["output"]["meaning_logits"]

            self.rewrite(path, downgrade)
            restored, _ = Observer.restore(path)
            self.assertEqual(observer.propose(), restored.propose())
            self.assertEqual(observer.think(), restored.think())

    def test_missing_or_wrong_meaning_tensor_rejected_with_valid_digest(self):
        self.semantic.interpret("difference 19 and 7")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.athanor"
            self.semantic.save(path)

            def erase(metadata, payload):
                payload["output"]["meaning_logits"] = None

            self.rewrite(path, erase)
            with self.assertRaises(PersistenceError):
                SemanticObserver.restore(path)


@unittest.skipUnless(AVAILABLE, "Install the neural extra for complete meaning learning")
class SemanticLearningAcceptance(unittest.TestCase):
    def test_complete_teaching_grounding_and_learned_execution(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            report = run_demo(Path(directory) / "meaning.athanor")
        self.assertGreaterEqual(report["trained_held_out"]["proposal"], 0.90)
        self.assertGreaterEqual(report["trained_held_out"]["ordered_noncommutative"], 0.90)
        self.assertTrue(report["weights_unchanged_during_teaching"])
        self.assertTrue(report["restored_next_decision_identical"])
        self.assertEqual(report["restored_new_concept_result"], "22")
        self.assertEqual(report["additional_parameters"], 0)


if __name__ == "__main__":
    unittest.main()
