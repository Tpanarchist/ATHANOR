"""Focused evidence, authority, native execution, controls and continuation contracts."""

import contextlib
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from athanor import Athanor, PersistenceError, Procedure, ValidationError, record
from athanor.world import World

AVAILABLE = importlib.util.find_spec("torch") is not None
if AVAILABLE:
    import torch

    from athanor.grounded.app import Application
    from athanor.grounded.checkpoint import restore, save
    from athanor.grounded.discovery import experiment_features
    from athanor.grounded.discovery_demo import run_demo
    from athanor.grounded.runtime import Runtime, Track
    from athanor.neural import Decision, ModelConfig, Observation, Proposal
    from athanor.neural.encoding import GraphBuilder
    from athanor.programs import literal


@unittest.skipUnless(AVAILABLE, "Install the neural extra for autonomous discovery")
class DiscoveryContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def setUp(self) -> None:
        self.runtime = Runtime()
        self.o = self.runtime.observer
        self.d = self.o.discovery

    def evidence(self, values=(0,) * 4):
        return self.o.knowledge.create(
            fields=record(kind="test_measured_evidence", observed=values),
            source="test measurement fixture",
        )

    def test_single_observation_is_provisional_and_alternatives_remain(self):
        value = self.d.estimate("self", "left")
        self.d._add(value, (-0.25, 0, 0, 0), self.evidence())
        self.assertEqual(value.status, "hypothesized")
        self.assertIsNone(value.operation)
        fields = self.o.knowledge.get(value.hypothesis).fields
        self.assertEqual(len(fields["alternatives"]), 3)
        self.assertEqual(fields["alternatives"][2]["prediction"], None)

    def test_repeated_effect_compiles_native_procedure_and_reuses_identity(self):
        value = self.d.estimate("self", "left")
        for _ in range(3):
            self.d._add(value, (-0.25, 0, 0, 0), self.evidence())
        self.assertEqual(value.status, "supported")
        op = value.operation
        prediction = self.d.predict(value, (4, 2, 3, 0))
        self.assertEqual(self.o.knowledge.get(prediction).fields["next_0"], 3.75)
        self.d._add(value, (-0.25, 0, 0, 0), self.evidence())
        self.assertEqual(value.operation, op)
        mismatch = self.d.predict(value, (4, 2, 3, 0), scope="untested")
        self.assertEqual(self.o.knowledge.get(mismatch).fields["status"], "unknown")
        self.assertIsNone(self.o.knowledge.get(mismatch).fields["next_0"])

    def test_contradiction_is_retained_and_current_reuse_is_withheld(self):
        value = self.d.estimate("self", "forward")
        for _ in range(3):
            self.d._add(value, (0, 0, 0.25, 0), self.evidence())
        pinned = value.hypothesis
        conflict = self.evidence()
        self.d._add(value, (0, 0, 0, 0), conflict)
        self.assertEqual(value.status, "qualified")
        self.assertIn(conflict, value.conflicts)
        self.assertIsNone(self.d.predict(value, (0, 0, 0, 0)))
        self.assertEqual(self.o.knowledge.get(pinned).fields["status"], "supported")
        self.assertEqual(value.hypothesis.id, pinned.id)

    def test_object_effect_needs_passive_comparison(self):
        appearance = (1.0, 0.2, 0.2)
        value = self.d.estimate("object", "interact", appearance, "near")
        for _ in range(3):
            self.d._add(value, (0, 0, 0.17), self.evidence(), context=(0, 0, 0.8))
        self.assertIsNone(value.operation)
        control = self.d.estimate("object", "wait", appearance, "near")
        for _ in range(2):
            self.d._add(control, (0, 0, 0), self.evidence(), context=(0, 0, 0.8))
        self.d._add(value, (0, 0, 0.17), self.evidence(), context=(0, 0, 0.8))
        self.assertIsNotNone(value.operation)
        out_of_range = self.d.predict(value, (0, 0, 1.3), appearance=appearance)
        self.assertEqual(self.o.knowledge.get(out_of_range).fields["status"], "unknown")
        alien = self.d.predict(value, (0, 0, 0.8), appearance=(0.2, 1, 0.2))
        self.assertEqual(self.o.knowledge.get(alien).fields["status"], "unknown")

    def test_unavailable_evidence_is_not_zero_effect(self):
        value = self.d.estimate("self", "forward")
        self.d._add(value, None, self.evidence())
        self.assertEqual(value.samples, [])
        self.assertEqual(value.excluded, 1)
        self.assertEqual(value.status, "unknown")

    def test_conflicted_passive_controls_do_not_establish_an_object_effect(self):
        appearance = (1.0, 0.2, 0.2)
        control = self.d.estimate("object", "wait", appearance, "near")
        for effect in (0.0, 0.0, 0.07):
            self.d._add(control, (0, 0, effect), self.evidence(), context=(0, 0, 0.8))
        self.assertEqual(control.status, "qualified")
        value = self.d.estimate("object", "interact", appearance, "near")
        for _ in range(3):
            self.d._add(value, (0, 0, 0.17), self.evidence(), context=(0, 0, 0.8))
        self.assertEqual(value.status, "hypothesized")
        self.assertIsNone(value.operation)

        # A later baseline contradiction also qualifies already acquired predictions.
        other_appearance = (0.2, 1.0, 0.2)
        stable = self.d.estimate("object", "wait", other_appearance, "near")
        for _ in range(2):
            self.d._add(stable, (0, 0, 0), self.evidence(), context=(0, 0, 0.8))
        acquired = self.d.estimate("object", "interact", other_appearance, "near")
        for _ in range(3):
            self.d._add(acquired, (0, 0, 0.17), self.evidence(), context=(0, 0, 0.8))
        self.assertIsNotNone(acquired.operation)
        self.d._add(stable, (0, 0, 0.07), self.evidence(), context=(0, 0, 0.8))
        self.assertEqual(acquired.status, "qualified")
        self.assertIsNone(self.d.predict(acquired, (0, 0, 0.8), appearance=other_appearance))
        fields = self.o.knowledge.get(acquired.hypothesis).fields
        self.assertEqual(fields["comparison"], stable.hypothesis)

    def test_controls_and_action_budget_bound_actual_execution(self):
        r = self.runtime
        r.start_discovery(1, focus="self", allowed=("wait",))
        base = self.o.observer.propose()
        decision = Decision(Proposal(self.o.operations["wait"]), 1, base.expectation)
        self.d.pause()
        self.assertFalse(r.autonomous_step()["acted"])
        self.assertEqual(r.world.tick, 0)
        self.d.resume()
        with patch.object(type(self.d), "choose", return_value=decision):
            self.assertTrue(r.autonomous_step()["acted"])
        self.assertEqual(self.d.status, "exhausted")
        self.assertEqual(self.d.spent, 1)
        self.assertFalse(r.autonomous_step()["acted"])
        self.assertEqual(r.world.tick, 1)
        with self.assertRaises(ValidationError):
            self.d.resume()

    def test_neural_abstention_has_no_world_effect(self):
        self.runtime.start_discovery(3)
        base = self.o.observer.propose()
        with patch.object(
            type(self.d), "choose", return_value=Decision(Proposal(None), 1, base.expectation)
        ):
            self.assertFalse(self.runtime.autonomous_step()["acted"])
        self.assertEqual(self.runtime.world.tick, 0)
        self.assertEqual(self.d.spent, 0)

    def test_unknown_physics_never_enters_inference(self):
        a, b = (
            World(namespace="hidden a", interaction_distance=0.08),
            World(namespace="hidden b", interaction_distance=0.27),
        )
        for world in (a, b):
            world.add_object((0, 0, 0.8), movable=True)
        x, y = Runtime(a), Runtime(b)
        self.assertEqual(a.sense(), b.sense())
        for runtime in (x, y):
            runtime.start_discovery(2, focus="object")
        self.assertEqual(x.observer.propose(), y.observer.propose())
        self.assertTrue(
            torch.equal(x.observer.observer.state.context, y.observer.observer.state.context)
        )
        self.assertFalse(hasattr(x.observer.discovery, "world"))

    def test_priority_encoding_cannot_memorize_action_names(self):
        values = record(
            uncertainty=1, relevance=1, benefit=1, reliability=1, cost=0, risk=0, action="interact"
        )
        other = record({**dict(values), "action": "turn_left"})
        self.assertEqual(experiment_features(values), experiment_features(other))

    def test_capability_feedback_changes_future_opportunities(self):
        self.runtime.start_discovery(3, focus="self", allowed=("forward", "wait"))
        before = self.o.knowledge.get(self.d.options[0]).fields["reliability"]
        self.o.observer.capabilities[self.o.operations["forward"]] = (12, 0)
        self.o.observe(self.runtime.world.sense())
        after = self.o.knowledge.get(self.d.options[0]).fields["reliability"]
        self.assertLess(after, before)

    def test_interface_focus_survives_a_visible_object(self):
        self.runtime.start_discovery(3, focus="interface", allowed=("forward", "wait"))
        self.o.tracks.append(
            Track(self.evidence(), (1.0, 0.2, 0.2), (0.0, 0.0, 0.8), 1.0, 0, "visible")
        )
        self.d.prepare(self.runtime.world.sense())
        for ref in self.d.options[:-1]:
            option = self.o.knowledge.get(ref).fields
            hypothesis = self.o.knowledge.get(option["hypothesis"]).fields
            self.assertEqual(hypothesis["channel"], "interface")
            self.assertEqual(option["relevance"], 1)

    def test_viewer_inspection_does_not_mutate_inquiry_or_knowledge(self):
        self.runtime.start_discovery(4)
        with tempfile.TemporaryDirectory() as directory:
            app = Application(self.runtime, Path(directory) / "state.athanor")
            native, controls = self.o.knowledge.dumps(), self.d.metadata()
            app.state()
            app.state()
            self.assertEqual(native, self.o.knowledge.dumps())
            self.assertEqual(controls, self.d.metadata())

    def test_paused_inquiry_restores_and_can_resume(self):
        self.runtime.start_discovery(3, focus="self")
        self.d.pause()
        self.runtime.paused = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.athanor"
            save(self.runtime, path)
            restored, _ = restore(path)
            self.assertEqual(self.d.metadata(), restored.observer.discovery.metadata())
            restored.observer.discovery.resume()
            self.assertEqual(restored.observer.discovery.status, "running")

    def test_rehashed_invalid_inquiry_state_rejects(self):
        self.runtime.start_discovery(3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.athanor"
            save(self.runtime, path)
            with zipfile.ZipFile(path) as archive:
                members = {name: archive.read(name) for name in archive.namelist()}
            metadata = json.loads(members["runtime.json"])
            metadata["discovery"]["spent"] = 5
            members["runtime.json"] = json.dumps(metadata).encode()
            members["manifest.json"] = json.dumps(
                {
                    name: hashlib.sha256(data).hexdigest()
                    for name, data in members.items()
                    if name != "manifest.json"
                }
            ).encode()
            with zipfile.ZipFile(path, "w") as archive:
                for name, data in members.items():
                    archive.writestr(name, data)
            with self.assertRaises(PersistenceError):
                restore(path)

    def test_incremental_activation_does_not_scan_installed_history(self):
        class IndexedOnly(Athanor):
            @property
            def structures(self):
                raise AssertionError("Full history scan is forbidden")

        a = IndexedOnly(namespace="indexed")
        query = a.create(fields={"kind": "question"})
        op = a.teach_operation("offered", Procedure((), None, {"value": literal(1)}))
        builder = GraphBuilder(ModelConfig(max_nodes=16))
        first = builder.build(a, Observation(query, ()), operations=(op,))
        baseline = builder.index.indexed
        for i in range(200):
            a.create(fields={"irrelevant": i})
        second = builder.build(a, Observation(query, ()), operations=(op,))
        self.assertEqual(first.refs, second.refs)
        self.assertEqual(builder.index.indexed, baseline + 200)
        builder.build(a, Observation(query, ()), operations=(op,))
        self.assertEqual(builder.index.indexed, baseline + 200)

    def test_legacy_world_physics_restores_default_law(self):
        raw = json.loads(World().dumps())
        raw["version"] = 1
        raw.pop("interaction_distance")
        restored = World.loads(json.dumps(raw))
        self.assertEqual(restored.interaction_distance, 0.2)


@unittest.skipUnless(AVAILABLE, "Install the neural extra for discovery acceptance")
class DiscoveryAcceptance(unittest.TestCase):
    def test_continuous_untaught_discovery_transfer_and_resume(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            base = Path("artifacts/grounded-observer.athanor")
            if not base.is_file():
                from athanor.grounded.demo import run_demo as prepare_grounding
                from athanor.semantics.demo import run_demo as prepare_meaning

                meaning = Path(directory) / "meaning.athanor"
                prepare_meaning(meaning)
                base = Path(directory) / "grounded.athanor"
                prepare_grounding(base, base_checkpoint=meaning)
            report = run_demo(Path(directory) / "discovery.athanor", base_checkpoint=base)
        self.assertFalse(report["teacher_answer_supplied"])
        self.assertFalse(report["hidden_world_inputs"])
        self.assertGreaterEqual(report["discovery"]["samples"], 3)
        self.assertAlmostEqual(report["discovery"]["effect"][2], 0.12, delta=0.09)
        self.assertLess(report["transfer_error"], 0.12)
        self.assertTrue(report["automatic_native_reuse"])
        self.assertTrue(report["shared_parameters_changed"])
        self.assertTrue(report["checkpoint_next_step_identical"])
        self.assertTrue(report["optimizer_continuation_identical"])
        self.assertTrue(all(step.get("manual") is False for step in report["trace"]))


if __name__ == "__main__":
    unittest.main()
