"""Focused integration contracts; the optional extra does not burden the cornerstone."""

import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path

from athanor import PersistenceError, Ref, ValidationError

AVAILABLE = importlib.util.find_spec("torch") is not None
if AVAILABLE:
    import torch

    from athanor.neural import ModelConfig, NeuralObserverModel, Observation, Observer, Proposal
    from athanor.neural.contracts import Activation, Decision, Expectation
    from athanor.neural.curriculum import lessons, situation
    from athanor.neural.demo import run_demo
    from athanor.neural.encoding import GraphBuilder, structural_features
    from athanor.neural.observer import measure
    from athanor.neural.training import Trainer, TrainingConfig


@unittest.skipUnless(AVAILABLE, "Install the neural extra for neural contracts")
class NeuralContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.model = NeuralObserverModel(ModelConfig(width=24))
        self.case = situation(441)

    def test_identity_and_display_names_are_not_learned_features(self):
        node = self.case.knowledge.get(self.case.target.operation)
        renamed = replace(node, ref=Ref("s99999999", 3, "other"))
        from athanor import record

        fields = dict(node.fields)
        fields["name"] = "an arbitrary new display name"
        renamed = replace(renamed, fields=record(fields))
        self.assertEqual(structural_features(node), structural_features(renamed))

    def test_model_initialization_preserves_caller_random_state(self):
        before = torch.random.get_rng_state().clone()
        other = NeuralObserverModel(self.model.config)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertTrue(
            all(
                torch.equal(value, other.state_dict()[key])
                for key, value in self.model.state_dict().items()
            )
        )

    def test_all_operations_and_incompatible_arguments_reach_neural_candidates(self):
        frame = self.case.frame(self.model.config)
        self.assertEqual(len(frame.operations), 4)
        self.assertEqual(frame.candidates, self.case.observation.candidates)
        self.assertTrue(
            any("text" in self.case.knowledge.get(ref).fields for ref in frame.candidates)
        )

    def test_new_operations_allocate_no_new_parameters(self):
        before = sum(p.numel() for p in self.model.parameters())
        frame = situation(997, novel=True).frame(self.model.config)
        output = self.model([frame])[0]
        self.assertEqual(len(output.operation_logits), 6)
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), before)

    def test_candidate_permutation_is_equivariant(self):
        frame = self.case.frame(self.model.config)
        reversed_case = replace(
            self.case,
            observation=Observation(self.case.observation.ref, tuple(reversed(frame.candidates))),
        )
        reordered = reversed_case.frame(self.model.config)
        with torch.no_grad():
            original = self.model([frame], forced=[self.case.target])[0]
            permuted = self.model([reordered], forced=[self.case.target])[0]
        torch.testing.assert_close(original.operation_logits, permuted.operation_logits)
        for left, right in zip(original.argument_logits, permuted.argument_logits, strict=True):
            torch.testing.assert_close(left, right.flip(0))

    def test_reference_revision_is_retained_at_boundary(self):
        other_namespace = Ref(self.case.observation.ref.id, 1, "foreign")
        observer = Observer(self.case.knowledge, self.model, self_ref=self.case.self_ref)
        with self.assertRaises(ValidationError):
            observer.observe(Observation(other_namespace, ()))
        self.assertIsNone(observer.observation)

    def test_internal_steps_persist_without_new_external_input(self):
        observer = Observer(self.case.knowledge, self.model, self_ref=self.case.self_ref)
        observer.observe(self.case.observation)
        before = observer.state.detached()
        knowledge_before = observer.knowledge.dumps()
        observer.think(steps=3)
        self.assertEqual(observer.state.steps, before.steps + 3)
        self.assertFalse(torch.equal(before.context, observer.state.context))
        self.assertEqual(observer.knowledge.dumps(), knowledge_before)
        self.assertFalse(observer.state.field.requires_grad)

    def test_invalid_activation_and_limits_fail_explicitly(self):
        frame = self.case.frame(self.model.config)
        with self.assertRaises(ValidationError):
            self.model([frame], [Activation((), torch.zeros(1, 24), torch.zeros(24))])
        with self.assertRaises(ValidationError):
            GraphBuilder(ModelConfig(max_nodes=2)).build(self.case.knowledge, self.case.observation)
        with self.assertRaises(ValidationError):
            self.model([frame], steps=0)

    def test_invalid_observation_processing_preserves_previous_state(self):
        observer = Observer(self.case.knowledge, self.model, self_ref=self.case.self_ref)
        observer.observe(self.case.observation)
        state = observer.state.detached()
        other = Observation(self.case.target.arguments[0], ())
        with self.assertRaises(ValidationError):
            observer.observe(other, steps=0)
        self.assertEqual(observer.observation, self.case.observation)
        self.assertTrue(torch.equal(observer.state.field, state.field))

    def test_rejection_enters_dynamic_self_capability_history(self):
        observer = Observer(self.case.knowledge, self.model, self_ref=self.case.self_ref)
        decision = observer.observe(self.case.observation)
        old_expectation = decision.expectation
        bad = next(
            ref
            for ref in self.case.observation.candidates
            if "text" in self.case.knowledge.get(ref).fields
        )
        probe = Decision(
            Proposal(self.case.target.operation, (bad,) * len(self.case.target.arguments)),
            0.5,
            old_expectation,
        )
        feedback = observer.act(probe)
        self.assertFalse(feedback.succeeded)
        self.assertEqual(observer.capabilities[self.case.target.operation], (1, 0))
        self.assertEqual(observer.knowledge.get(observer.self_map[-1]).fields["status"], "rejected")
        self.assertNotEqual(observer.propose().expectation, old_expectation)

    def test_rejected_execution_is_atomic_and_channel_specific(self):
        target = self.case.target
        bad = next(
            ref
            for ref in self.case.observation.candidates
            if "text" in self.case.knowledge.get(ref).fields
        )
        invalid = Proposal(target.operation, (bad,) * len(target.arguments))
        decision = Decision(invalid, 0.7, Expectation(0.8, 0.9, (0.5,) * 6))
        before = self.case.knowledge.dumps()
        feedback = measure(self.case.knowledge, decision)
        self.assertEqual(feedback.status, "rejected")
        self.assertIsNotNone(feedback.error_type)
        self.assertAlmostEqual(feedback.discrepancies.capability, -0.8)
        self.assertIsNone(feedback.discrepancies.preference)
        self.assertIsNone(feedback.discrepancies.predictive)
        self.assertEqual(before, self.case.knowledge.dumps())

    def test_success_does_not_imply_preference(self):
        decision = Decision(self.case.target, 0.7, Expectation(0.8, 0.9, (0.5,) * 6))
        feedback = measure(self.case.knowledge, decision)
        self.assertTrue(feedback.succeeded)
        self.assertIsNone(feedback.satisfied)
        self.assertIsNone(feedback.discrepancies.preference)
        wrong_preference = measure(self.case.knowledge, decision, evaluate=lambda _: False)
        self.assertTrue(wrong_preference.succeeded)
        self.assertFalse(wrong_preference.satisfied)
        self.assertAlmostEqual(wrong_preference.discrepancies.preference, -0.9)

    def test_abstention_is_not_failure(self):
        before = self.case.knowledge.dumps()
        feedback = measure(
            self.case.knowledge, Decision(Proposal(None), 0.7, Expectation(0.5, 0.5, (0.5,) * 6))
        )
        self.assertIsNone(feedback.succeeded)
        self.assertIsNone(feedback.discrepancies.capability)
        self.assertEqual(before, self.case.knowledge.dumps())

    def test_self_map_records_real_execution_and_changes_processing(self):
        observer = Observer(self.case.knowledge, self.model, self_ref=self.case.self_ref)
        observer.observe(self.case.observation)
        before = observer.state.context.clone()
        feedback = observer.act(
            Decision(self.case.target, 0.7, Expectation(0.8, 0.9, (0.5,) * 6)),
            evaluate=lambda _: True,
        )
        node = observer.knowledge.get(observer.self_map[-1])
        self.assertEqual(node.fields["result"], feedback.result)
        self.assertEqual(node.fields["observer"], observer.self_ref)
        self.assertTrue(node.fields["success"])
        self.assertEqual(observer.capabilities[self.case.target.operation], (1, 1))
        self.assertIn(node.ref, observer.state.refs)
        self.assertFalse(torch.equal(before, observer.state.context))

    def test_training_uses_measured_targets_and_changes_parameters(self):
        material = lessons(16, self.model.config)
        self.assertTrue(any(lesson.feedback.succeeded is False for lesson in material))
        self.assertTrue(any(lesson.feedback.satisfied is True for lesson in material))
        trainer = Trainer(self.model)
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        loss = trainer.train(material, epochs=2)
        self.assertEqual(trainer.updates, 2)
        self.assertGreater(loss["loss"], 0)
        self.assertTrue(
            any(
                not torch.equal(before[key], value)
                for key, value in self.model.state_dict().items()
            )
        )

    def test_fresh_and_active_checkpoint_roundtrip(self):
        observer = Observer(self.case.knowledge, self.model, self_ref=self.case.self_ref)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observer.athanor"
            observer.save(path)
            fresh, trainer = Observer.restore(path)
            self.assertIsNone(trainer)
            self.assertIsNone(fresh.observation)
            observer.observe(self.case.observation)
            observer.act(Decision(self.case.target, 0.7, Expectation(0.8, 0.9, (0.5,) * 6)))
            observer.save(path)
            restored, _ = Observer.restore(path)
            self.assertEqual(observer.knowledge.dumps(), restored.knowledge.dumps())
            self.assertEqual(observer.propose(), restored.propose())
            self.assertEqual(observer.think(), restored.think())
            self.assertTrue(torch.equal(observer.state.field, restored.state.field))

    def test_optimizer_and_shuffle_resume_exactly(self):
        material = lessons(16, self.model.config)
        trainer = Trainer(self.model, TrainingConfig(batch_size=8))
        trainer.train(material)
        observer = Observer(self.case.knowledge, self.model, self_ref=self.case.self_ref)
        observer.observe(self.case.observation)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observer.athanor"
            observer.save(path, trainer=trainer)
            restored, continued = Observer.restore(path)
            self.assertIsNotNone(continued)
            trainer.train(material)
            continued.train(material)
            self.assertEqual(trainer.updates, continued.updates)
            for key, value in self.model.state_dict().items():
                self.assertTrue(torch.equal(value, restored.model.state_dict()[key]))

    def test_checkpoint_corruption_and_unknown_fields_rejected(self):
        observer = Observer(self.case.knowledge, self.model, self_ref=self.case.self_ref)
        observer.observe(self.case.observation)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observer.athanor"
            observer.save(path)
            with zipfile.ZipFile(path) as archive:
                content = {name: archive.read(name) for name in archive.namelist()}
            metadata = json.loads(content["metadata.json"])
            metadata["surprise"] = True
            content["metadata.json"] = json.dumps(metadata).encode()
            with zipfile.ZipFile(path, "w") as archive:
                for name, value in content.items():
                    archive.writestr(name, value)
            with self.assertRaises(PersistenceError):
                Observer.restore(path)

    def test_stale_graph_save_preserves_existing_checkpoint(self):
        observer = Observer(self.case.knowledge, self.model, self_ref=self.case.self_ref)
        observer.observe(self.case.observation)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observer.athanor"
            observer.save(path)
            old = path.read_bytes()
            from athanor.programs import Procedure, literal

            self.case.knowledge.teach_operation(
                "new operation", Procedure((), None, {"value": literal(1)})
            )
            with self.assertRaises(PersistenceError):
                observer.save(path)
            self.assertEqual(path.read_bytes(), old)

    def test_nonfinite_tensor_rejected_even_with_matching_digest(self):
        observer = Observer(self.case.knowledge, self.model, self_ref=self.case.self_ref)
        observer.observe(self.case.observation)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observer.athanor"
            observer.save(path)
            with zipfile.ZipFile(path) as archive:
                content = {name: archive.read(name) for name in archive.namelist()}
            payload = torch.load(io.BytesIO(content["tensors.pt"]), weights_only=True)
            payload["context"][0] = float("nan")
            buffer = io.BytesIO()
            torch.save(payload, buffer)
            content["tensors.pt"] = buffer.getvalue()
            metadata = json.loads(content["metadata.json"])
            metadata["hashes"]["tensors.pt"] = hashlib.sha256(content["tensors.pt"]).hexdigest()
            content["metadata.json"] = json.dumps(metadata).encode()
            with zipfile.ZipFile(path, "w") as archive:
                for name, value in content.items():
                    archive.writestr(name, value)
            with self.assertRaises(PersistenceError):
                Observer.restore(path)


@unittest.skipUnless(AVAILABLE, "Install the neural extra for the learning acceptance test")
class NeuralLearningAcceptance(unittest.TestCase):
    def test_complete_learned_observer(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            report = run_demo(Path(directory) / "learned.athanor")
        self.assertGreaterEqual(report["trained_held_out"]["proposal"], 0.90)
        self.assertTrue(report["restored_next_decision_identical"])


if __name__ == "__main__":
    unittest.main()
