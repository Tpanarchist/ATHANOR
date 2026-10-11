"""Native composition, empirical scope, dependency revision and whole-session contracts."""

import contextlib
import copy
import importlib.util
import io
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from athanor import (
    Athanor,
    Endpoint,
    FieldSpec,
    Limits,
    Procedure,
    ResourceLimitError,
    Role,
    ValidationError,
    record,
)
from athanor.programs import argument, binary, call, choose, field, literal, record_expression
from athanor.values import as_record

AVAILABLE = importlib.util.find_spec("torch") is not None
if AVAILABLE:
    import torch

    from athanor.grounded.app import Application
    from athanor.grounded.checkpoint import restore, save
    from athanor.grounded.process_demo import run_demo, territory
    from athanor.grounded.processes import Processes, project, selectors, stored_path
    from athanor.grounded.runtime import Runtime
    from athanor.grounded.training import GroundedTrainer


class NativeRecordContracts(unittest.TestCase):
    def test_native_nested_record_exact_restore_and_bounded_atomic_failure(self):
        a = Athanor(namespace="record-contract")
        category = a.teach_category("packet", {"nested": FieldSpec("record")})
        operation = a.teach_operation(
            "construct",
            Procedure(
                (),
                category,
                {"nested": record_expression({"value": binary("div", literal(1), literal(3))})},
            ),
        )
        result = a.execute(operation)
        self.assertEqual(a.get(result).fields, record(nested=record(value=Fraction(1, 3))))
        restored = Athanor.loads(a.dumps())
        self.assertEqual(restored.get(restored.execute(operation)).fields, a.get(result).fields)
        before = a.dumps()
        with self.assertRaises(ResourceLimitError):
            a.execute(operation, limits=Limits(max_steps=1))
        self.assertEqual(before, a.dumps())
        with self.assertRaises(ValidationError):
            a.teach_operation(
                "malformed", Procedure((), None, {"x": record(op="record", fields=record(x=3))})
            )
        self.assertEqual(before, a.dumps())


@unittest.skipUnless(AVAILABLE, "Install the neural extra for recursive discovery")
class ProcessContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.runtime = Runtime()
        self.p = self.runtime.observer.processes
        self.p.start(6, interface="numeric")

    def acquire(self):
        p = self.p
        for value in (Fraction(1, 5), Fraction(2, 5), Fraction(4, 5)):
            p.observe(value, value / (1 + value), source="measured numeric fixture")
        add, _, _, div = p.library[:4]
        wanted = call(div, argument(0), call(add, argument(0), literal(p.constants[0])))
        candidate = next(c for c in p.candidates if c.expression == wanted)
        with patch.object(p, "rank", return_value=[candidate]):
            operation = p.adopt()
        self.assertIsNotNone(operation)
        return operation, candidate

    def packets(self):
        return [
            (
                record(channel=(10 + i, v), valid=True, passes=1),
                record(readings=(v / (1 + v), 90 - i), valid=True),
            )
            for i, v in enumerate((Fraction(1, 5), Fraction(2, 5), Fraction(4, 5)))
        ]

    def mapping(self, operation):
        # This test isolates binding from an untrained random priority scorer.
        model = self.runtime.observer.observer.model

        def binding_scores(v, c, **kw):
            scores = torch.zeros(v.shape[0])
            scores[2] = 1
            return scores

        with patch.object(model, "_scores", side_effect=binding_scores):
            result = self.p.correspond(operation, self.packets(), source="typed packet fixture")
        self.assertIsNotNone(result)
        return result

    def test_constructor_answer_independent_and_installed_is_not_mastery(self):
        self.assertEqual(len(self.p.candidates), 272)
        self.assertEqual(self.p.capabilities, {})
        self.assertTrue(all(c.status == "unknown" for c in self.p.candidates))
        other = Runtime().observer.processes
        other.start(6, interface="numeric")
        # Same relative structure, despite independent native identities/namespaces.
        self.assertEqual([c.calls for c in other.candidates], [c.calls for c in self.p.candidates])
        with self.assertRaises(ValidationError):
            self.p.construct(max_calls=999)
        state = self.p.metadata()
        with self.assertRaises(ValidationError):
            self.p.start(6, interface="hidden")
        self.assertEqual(state, self.p.metadata())

    def test_evidence_veto_does_not_substitute_supported_answer(self):
        operation, candidate = self.acquire()
        failed = next(c for c in self.p.candidates if c.status == "qualified")
        with patch.object(self.p, "rank", return_value=[failed, candidate]):
            self.assertIsNone(self.p.adopt())
        self.assertEqual(self.p.selected, failed.operation)
        self.assertEqual(self.p.adopted, [operation])

    def test_conditional_constituent_and_recursive_reference_are_generic(self):
        operation, _ = self.acquire()
        scalar = self.p.scalar
        conditional = self.p.knowledge.teach_operation(
            "positive",
            Procedure(
                (Endpoint(Role.DATA, scalar),),
                scalar,
                {
                    "value": choose(
                        binary("lt", literal(0), field(argument(0), "value")),
                        field(argument(0), "value"),
                        literal(0),
                    )
                },
            ),
        )
        self.p.offer(conditional)
        self.p.start(6, interface="numeric", max_candidates=768)
        expressions = [c.expression for c in self.p.candidates]
        self.assertIn(call(operation, call(operation, argument(0))), expressions)
        self.assertIn(call(conditional, argument(0)), expressions)
        self.assertEqual(self.p.evaluate(operation, Fraction(1, 2))[0], Fraction(1, 3))

    def test_selectors_are_typed_and_malformed_paths_reject(self):
        values = record(valid=True, items=(3, record(x=Fraction(1, 7))))
        self.assertEqual(selectors(values), [("items", 0), ("items", 1, "x")])
        self.assertEqual(project(values, ("items", 1, "x")), Fraction(1, 7))
        with self.assertRaises(ValidationError):
            project(values, ("valid",))
        with self.assertRaises(ValidationError):
            stored_path((Fraction(1, 2),))

    def test_typed_input_output_binding_scope_and_fresh_conflict(self):
        operation, _ = self.acquire()
        mapping = self.mapping(operation)
        before = record(channel=(99, Fraction(3, 5)), valid=True, passes=1)
        template = record(readings=(0, -9), valid=True)
        result = self.p.materialize_mapping(mapping, before, template)
        self.assertEqual(
            self.p.knowledge.get(result).fields, record(readings=(Fraction(3, 8), -9), valid=True)
        )
        self.assertIsNone(
            self.p.apply_mapping(mapping, record(channel=(99, 2), valid=True, passes=1))
        )
        self.assertIsNone(
            self.p.apply_mapping(
                mapping, record(channel=(99, Fraction(3, 5)), valid=True, passes=2)
            )
        )
        self.assertIsNone(
            self.p.apply_mapping(mapping, record(channel=(99, Fraction(3, 5)), valid=1, passes=1))
        )
        self.assertGreater(
            self.p.capabilities[operation]["binding_attempts"],
            self.p.capabilities[operation]["binding_successes"],
        )
        self.p.apply_mapping(mapping, before, observed=record(readings=(9, -9), valid=True))
        self.assertIsNone(self.p.apply_mapping(mapping, before))

    def test_mapping_adoption_is_atomic_on_late_contract_failure(self):
        operation, _ = self.acquire()
        original = self.p._bind

        def fail(staged, *args):
            original(staged, *args)
            raise ValidationError("simulated late output binding failure")

        with patch.object(self.p, "_bind", side_effect=fail):
            with self.assertRaises(ValidationError):
                self.mapping(operation)
        self.assertEqual(self.p.mappings, [])
        self.assertFalse(
            any(
                node.fields.get("name") == "Observed input packet contract"
                for node in self.p.knowledge.structures
            )
        )

    def test_ambiguous_correspondence_retains_uncertainty(self):
        operation, _ = self.acquire()
        pairs = [
            (record(x=v, y=v), record(z=v / (1 + v)))
            for v in (Fraction(1, 5), Fraction(2, 5), Fraction(4, 5))
        ]
        self.assertIsNone(self.p.correspond(operation, pairs, source="co-varying ambiguous leaves"))
        self.assertEqual(
            self.p.knowledge.get(self.p.correspondence_hypotheses[-1]).fields["status"], "unknown"
        )

    def test_contradiction_qualifies_dependents_and_keeps_pinned_support(self):
        operation, candidate = self.acquire()
        mapping = self.mapping(operation)
        pinned = candidate.hypothesis
        self.p.start(6, interface="numeric")
        for v in (Fraction(1, 5), Fraction(2, 5), Fraction(4, 5)):
            self.p.observe(v, v / (1 + 2 * v), source="measured repeated process fixture")
        larger = next(
            c
            for c in self.p.candidates
            if c.expression == call(operation, call(operation, argument(0)))
        )
        with patch.object(self.p, "rank", return_value=[larger]):
            self.p.adopt()
        self.p.qualify(operation, "fresh contradictory receipt")
        self.assertIsNone(self.p.apply(larger.operation, Fraction(1, 2)))
        self.assertIsNone(self.p.apply_mapping(mapping, self.packets()[0][0]))
        self.assertEqual(self.p.knowledge.get(pinned).fields["status"], "supported")
        self.assertEqual(
            self.p.knowledge.get(candidate.hypothesis).fields["support"],
            self.p.knowledge.get(pinned).fields["support"],
        )
        self.assertTrue(self.p.knowledge.get(larger.hypothesis).fields["contradictions"])

    def test_real_shared_update_without_parameter_growth(self):
        self.acquire()
        model = self.runtime.observer.observer.model
        weights = {n: v.clone() for n, v in model.state_dict().items()}
        count = sum(v.numel() for v in model.parameters())
        trainer = GroundedTrainer(self.runtime.observer)
        self.p.learn(trainer, steps=2)
        self.assertEqual(sum(v.numel() for v in model.parameters()), count)
        self.assertTrue(any(not torch.equal(weights[n], v) for n, v in model.state_dict().items()))
        self.assertEqual(trainer.updates, 2)

    def test_restore_rejects_native_disagreement_and_capability_forgery(self):
        self.acquire()
        metadata = self.p.metadata()
        bad = copy.deepcopy(metadata)
        bad["candidates"][0]["calls"] = 3
        with self.assertRaises(ValidationError):
            Processes.restore(self.runtime.observer, bad)
        bad = copy.deepcopy(metadata)
        bad["library"][0] = None
        with self.assertRaises(ValidationError):
            Processes.restore(self.runtime.observer, bad)
        wrong = self.p.knowledge.teach_category("Distinct quantity", {"value": FieldSpec("number")})
        bad = copy.deepcopy(metadata)
        bad["scalar"] = [wrong.id, wrong.revision, wrong.namespace]
        with self.assertRaises(ValidationError):
            Processes.restore(self.runtime.observer, bad)
        bad = copy.deepcopy(metadata)
        bad["capabilities"][0]["counts"]["recognized_successes"] = 999
        with self.assertRaises(ValidationError):
            Processes.restore(self.runtime.observer, bad)
        self.assertEqual(Processes.restore(self.runtime.observer, metadata).metadata(), metadata)

    def test_checkpoint_and_application_inspection_controls(self):
        operation, _ = self.acquire()
        self.p.choose_probe((Fraction(1, 5), Fraction(4, 5)))
        self.runtime.observer.observe(self.runtime.world.sense())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "process.athanor"
            app = Application(self.runtime, path)
            prior = self.p.metadata()
            app.state()
            app.state()
            self.assertEqual(prior, self.p.metadata())
            app.command({"op": "pause"})
            self.assertEqual(self.p.status, "paused")
            self.assertFalse(self.runtime.autonomous_step()["acted"])
            save(self.runtime, path, trainer=app.trainer)
            continued, _ = restore(path)
            self.assertEqual(continued.observer.processes.metadata(), self.p.metadata())
            continued.observer.processes.inputs.clear()
            continued.observer.processes.outputs.clear()
            continued.observer.processes.examples.clear()
            result = continued.observer.processes.apply(operation, Fraction(1, 2))
            self.assertEqual(
                continued.observer.knowledge.get(result).fields["value"], Fraction(1, 3)
            )
            app.command({"op": "stop_inquiry"})
            self.assertEqual(self.p.status, "stopped")

    def test_world_law_is_not_an_observer_feature_and_rolls_back(self):
        world = territory()
        sensor = world.sense()
        before = world.dumps()
        scalar = as_record(world.knowledge.get(world.motion_process).fields["procedure"])["result"]
        bad = world.knowledge.teach_operation(
            "unbounded", Procedure((Endpoint(Role.DATA, scalar),), scalar, {"value": literal(9)})
        )
        world.set_motion_process(bad)
        self.assertEqual(world.sense(), sensor)
        from athanor.world import Action

        before = world.dumps()
        with self.assertRaises(ValidationError):
            world.step(Action("forward"))
        self.assertEqual(world.dumps(), before)

    def test_checkpoint_resumes_identical_shared_process_update(self):
        self.acquire()
        trainer = GroundedTrainer(self.runtime.observer)
        self.p.learn(trainer, steps=2)
        self.runtime.observer.observe(self.runtime.world.sense())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.athanor"
            save(self.runtime, path, trainer=trainer)
            other, resumed = restore(path)
        for old, new in zip(self.p._retained, other.observer.processes._retained, strict=True):
            self.assertTrue(torch.equal(old, new))
        self.p.learn(trainer, steps=1)
        other.observer.processes.learn(resumed, steps=1)
        for name, value in self.runtime.observer.observer.model.state_dict().items():
            self.assertTrue(
                torch.equal(value, other.observer.observer.model.state_dict()[name]), name
            )
        self.assertEqual(trainer.updates, resumed.updates)


@unittest.skipUnless(AVAILABLE, "Install neural extra for process acceptance")
class ProcessAcceptance(unittest.TestCase):
    @unittest.skipUnless(
        Path("artifacts/discovery-observer.athanor").exists(),
        "Restore Goal 005 acceptance checkpoint first",
    )
    def test_continuous_autonomous_physical_transfer_recursive_restoration(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            report = run_demo(Path(directory) / "process.athanor")
        self.assertTrue(report["numeric_transfer_exact"])
        self.assertTrue(report["unsupported_correspondence_rejected"])
        self.assertTrue(report["restored_reuse_without_replay"])
        self.assertTrue(report["parameter_count_unchanged"])
        self.assertTrue(report["shared_parameters_changed"])
        self.assertLess(report["physical_transfer_error"], 0.001)


if __name__ == "__main__":
    unittest.main()
