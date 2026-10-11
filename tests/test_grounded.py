import dataclasses
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from athanor import PersistenceError, ValidationError
from athanor.world import Action, World

AVAILABLE = importlib.util.find_spec("torch") is not None
if AVAILABLE:
    import torch

    from athanor.grounded.app import Application
    from athanor.grounded.checkpoint import restore, save
    from athanor.grounded.perception import SensoryEncoder, pixels, regions
    from athanor.grounded.runtime import GroundedObserver, Runtime
    from athanor.grounded.training import GroundedTrainer, train_vision, visual_lessons


@unittest.skipUnless(AVAILABLE, "Install the neural extra for grounded contracts")
class GroundedContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_sensory_input_changes_shared_field_and_receives_gradients(self) -> None:
        o = GroundedObserver()
        w = World()
        w.add_object((0, 0, 3))
        o.observe(w.sense())
        frame = o.observer._frame
        self.assertIsNotNone(frame)
        model = o.observer.model
        bare = model([frame])[0]
        embedding = o.observer.sensory.activation(pixels(w.sense()), torch.zeros(6))
        sensed = model([frame], sensory=[embedding])[0]
        self.assertFalse(torch.equal(bare.activation.context, sensed.activation.context))
        torch.autograd.backward(sensed.activation.context.square().sum())
        self.assertIsNotNone(o.observer.sensory.fusion.weight.grad)
        self.assertGreater(float(o.observer.sensory.fusion.weight.grad.abs().sum()), 0)
        self.assertTrue(torch.equal(model([frame])[0].operation_logits, bare.operation_logits))

    def test_visual_training_changes_real_parameters(self) -> None:
        encoder = SensoryEncoder()
        before = encoder.image[0].weight.detach().clone()
        result = train_vision(encoder, visual_lessons(12, seed=8414), epochs=3)
        self.assertFalse(torch.equal(before, encoder.image[0].weight))
        self.assertTrue(result["loss"] > 0)

    def test_equal_pixels_with_different_territory_ids_give_equal_inference(self) -> None:
        a, b = World(namespace="private a"), World(namespace="private b")
        a.add_object((0.5, 0, 3))
        b.add_object((0.5, 0, 3))
        x, y = GroundedObserver(namespace="same map"), GroundedObserver(namespace="same map")
        self.assertEqual(x.observe(a.sense()), y.observe(b.sense()))
        self.assertTrue(torch.equal(x.observer.state.context, y.observer.state.context))
        self.assertFalse(hasattr(x, "world"))

    def test_unknown_instruction_is_not_a_motor_proposal(self) -> None:
        o = GroundedObserver()
        o.observe(World().sense())
        self.assertIn("unknown", o.instruct("approach nonexistent"))
        self.assertEqual(o.propose()[0], Action())
        self.assertIsNone(o.command(o.propose()[1]))

    def test_failed_movement_preserves_competing_causes_and_actual_feedback(self) -> None:
        w = World()
        w.motor = False
        r = Runtime(w)
        r.step(Action("forward"))
        node = r.observer.knowledge.get(r.observer.last_feedback)
        self.assertEqual(node.fields["executed"], "wait")
        self.assertFalse(node.fields["capability_success"])
        self.assertTrue(node.fields["uncertain"])
        self.assertEqual(len(node.fields["alternatives"]), 3)
        self.assertIn(node.ref, r.observer.observer.self_map)
        self.assertEqual(r.observer.experience["forward"][0], 1)

    def test_camera_limitation_not_exposed_as_hidden_fault_flag(self) -> None:
        w = World()
        w.add_object((0, 0, 3))
        before = w.sense()
        w.camera_hold = True
        w.step(Action("forward"))
        after = w.sense()
        self.assertEqual(before.left, after.left)
        self.assertEqual(after.proprio[5], 1)
        self.assertNotIn("camera_hold", dataclasses.asdict(after))

    def test_checkpoint_preserves_next_recurrent_step_and_world(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            r = Runtime()
            r.step(Action("right"))
            t = GroundedTrainer(r.observer)
            path = Path(directory) / "state.athanor"
            save(r, path, trainer=t)
            other, continued = restore(path)
            self.assertIsNotNone(continued)
            self.assertEqual(r.world.dumps(), other.world.dumps())
            self.assertEqual(r.observer.knowledge.dumps(), other.observer.knowledge.dumps())
            self.assertEqual(r.observer.observer.think(), other.observer.observer.think())
            self.assertTrue(
                torch.equal(r.observer.observer.state.field, other.observer.observer.state.field)
            )
            self.assertEqual(r.step(Action("left")), other.step(Action("left")))

    def test_checkpoint_corruption_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.athanor"
            save(Runtime(), path)
            with zipfile.ZipFile(path) as archive:
                data = {n: archive.read(n) for n in archive.namelist()}
            data["runtime.json"] = json.dumps({"version": 99}).encode()
            with zipfile.ZipFile(path, "w") as archive:
                for n, v in data.items():
                    archive.writestr(n, v)
            with self.assertRaises(PersistenceError):
                restore(path)

    def test_rehashed_nonfinite_optimizer_state_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime()
            trainer = GroundedTrainer(runtime.observer)
            runtime.step(Action("right"))
            before, after, action, frame, prior = runtime.last_transition
            trainer.learn_transition(before, after, action, frame=frame, prior=prior)
            path = Path(directory) / "state.athanor"
            save(runtime, path, trainer=trainer)
            with zipfile.ZipFile(path) as archive:
                content = {name: archive.read(name) for name in archive.namelist()}
            payload = torch.load(io.BytesIO(content["sensory.pt"]), weights_only=True)
            payload["optimizer"]["param_groups"][0]["lr"] = float("nan")
            buffer = io.BytesIO()
            torch.save(payload, buffer)
            content["sensory.pt"] = buffer.getvalue()
            hashes = json.loads(content["manifest.json"])
            hashes["sensory.pt"] = hashlib.sha256(content["sensory.pt"]).hexdigest()
            content["manifest.json"] = json.dumps(hashes).encode()
            with zipfile.ZipFile(path, "w") as archive:
                for name, value in content.items():
                    archive.writestr(name, value)
            with self.assertRaises(PersistenceError):
                restore(path)

    def test_live_learning_changes_parameters_from_actual_consequence(self) -> None:
        r = Runtime()
        t = GroundedTrainer(r.observer)
        before = r.world.sense()
        frame = r.observer.observer._frame
        r.step(Action("forward"))
        after = r.world.sense()
        weights = r.observer.observer.sensory.physical[0].weight.detach().clone()
        t.learn_transition(before, after, "forward", frame=frame, preference=0.2)
        self.assertFalse(torch.equal(weights, r.observer.observer.sensory.physical[0].weight))
        self.assertEqual(t.updates, 1)

    def test_application_commands_change_actual_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = Application(Runtime(), Path(directory) / "save.athanor")
            app.command({"op": "start"})
            self.assertFalse(app.runtime.paused)
            app.command({"op": "pause"})
            self.assertTrue(app.runtime.paused)
            app.command({"op": "manual", "action": "forward"})
            self.assertEqual(app.runtime.world.tick, 1)
            app.command({"op": "train"})
            self.assertEqual(app.trainer.updates, 1)
            app.command({"op": "save"})
            app.command({"op": "manual", "action": "right"})
            app.command({"op": "restore"})
            self.assertEqual(app.runtime.world.tick, 1)
            state = app.state()
            self.assertIn("territory", state)
            self.assertIn("sensor", state)
            before = app.runtime.world.dumps()
            for invalid in ({"op": "invent"}, {"op": []}, {}):
                with self.assertRaises(ValidationError):
                    app.command(invalid)
                self.assertEqual(before, app.runtime.world.dumps())

    def test_component_extraction_reports_truncation(self) -> None:
        mask = torch.zeros(8, 8)
        mask[1:4, 0:3] = 0.9
        color = torch.ones(3, 8, 8)
        components = regions(mask, color)
        self.assertEqual(len(components), 1)
        self.assertTrue(components[0].truncated)

    def test_touching_different_appearances_remain_distinguishable(self) -> None:
        mask = torch.zeros(8, 8)
        mask[2:5, 1:7] = 0.95
        colors = torch.zeros(3, 8, 8)
        colors[0, :, :4] = 1
        colors[1, :, 4:] = 1
        components = regions(mask, colors)
        self.assertEqual(len(components), 2)
        self.assertLess(components[0].center[0], components[1].center[0])

    def test_checkpoint_resumes_identical_actual_feedback_update(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            r = Runtime()
            t = GroundedTrainer(r.observer)
            r.step(Action("forward"))
            for p in r.observer.observer.sensory.image.parameters():
                p.requires_grad_(False)
            path = Path(directory) / "continue.athanor"
            save(r, path, trainer=t)
            other, continued = restore(path)
            self.assertIsNotNone(continued)
            self.assertIsNotNone(other.last_transition)
            for runtime, trainer in ((r, t), (other, continued)):
                before, after, action, frame, prior = runtime.last_transition
                trainer.learn_transition(before, after, action, frame=frame, prior=prior)
            self.assertEqual(t.updates, continued.updates)
            for first, second in (
                (r.observer.observer.model, other.observer.observer.model),
                (r.observer.observer.sensory, other.observer.observer.sensory),
            ):
                for name, parameter in first.named_parameters():
                    restored = dict(second.named_parameters())[name]
                    self.assertEqual(parameter.requires_grad, restored.requires_grad)
                    self.assertTrue(torch.equal(parameter, restored), name)

    def test_interleaved_training_resumes_one_shared_model_and_optimizer(self) -> None:
        from athanor.grounded.runtime import physical_outcome
        from athanor.grounded.training import EmbodiedLesson
        from athanor.neural.curriculum import lessons
        from athanor.semantics.curriculum import lessons as meanings

        with tempfile.TemporaryDirectory() as directory:
            r = Runtime()
            t = GroundedTrainer(r.observer)
            r.step(Action("forward"))
            before, after, action, frame, prior = r.last_transition
            sample = EmbodiedLesson(
                frame,
                before,
                r.observer.operations[action],
                0,
                physical_outcome(before, after),
                prior,
            )
            replay = lessons(8, r.observer.observer.model.config) + meanings(
                2, r.observer.observer.model.config
            )
            path = Path(directory) / "training.athanor"
            save(r, path, trainer=t)
            other, continued = restore(path)
            old = r.observer.observer.model.encoder.weight.detach().clone()
            for trainer in (t, continued):
                self.assertIs(trainer.native.optimizer, trainer.optimizer)
                self.assertIs(trainer.native.model, trainer.observer.observer.model)
                self.assertIs(trainer.native.rng, trainer.rng)
                trainer.train([sample], epochs=1, replay=replay)
            self.assertEqual(t.updates, 2)
            self.assertFalse(torch.equal(old, r.observer.observer.model.encoder.weight))
            for name, value in r.observer.observer.model.state_dict().items():
                self.assertTrue(
                    torch.equal(value, other.observer.observer.model.state_dict()[name]), name
                )

    def test_unresolved_map_position_cannot_establish_exact_nearness(self) -> None:
        from athanor.values import as_ref

        o = GroundedObserver()
        operation = as_ref(o.knowledge.get(o.near).fields["operation"])
        candidate = o.knowledge.create(fields={"status": "monocular", "x": 0, "z": 0})
        before = o.knowledge.dumps()
        with self.assertRaises(ValidationError):
            o.knowledge.execute(operation, (candidate,))
        self.assertEqual(before, o.knowledge.dumps())

    def test_invalid_command_has_no_physical_effect_and_records_rejection(self) -> None:
        from athanor.neural import Proposal

        r = Runtime()
        before = r.world.dumps()
        decision = dataclasses.replace(
            r.observer.observer.propose(),
            proposal=Proposal(
                r.observer.operations["forward"],
                (r.observer.observer.self_ref,),
            ),
        )
        self.assertIsNone(r.observer.command(decision))
        self.assertEqual(before, r.world.dumps())
        node = r.observer.knowledge.get(r.observer.observer.self_map[-1])
        self.assertEqual(node.fields["kind"], "embodied_proposal_rejection")
        self.assertFalse(node.fields["physical_effect"])

    def test_internal_refinement_does_not_repeat_world_odometry(self) -> None:
        r = Runtime()
        r.step(Action("forward"))
        before = r.world.dumps()
        r.observer.observer.think()
        self.assertEqual(before, r.world.dumps())
        self.assertEqual(r.world.tick, 1)

    def test_invalid_compact_sensory_intent_rejected(self) -> None:
        r = Runtime()
        frame = r.observer.observer._frame
        model = r.observer.observer.model
        for invalid in (torch.zeros(3), torch.full_like(frame.intent_features, float("nan"))):
            with self.assertRaises(ValidationError):
                model([dataclasses.replace(frame, intent_features=invalid)], sensory=[None])

    def test_measured_transition_keeps_actual_inference_input_state(self) -> None:
        from athanor.grounded.perception import pixels
        from athanor.grounded.runtime import intent_activation

        r = Runtime()
        observer = r.observer.observer
        expected = observer._output.operation_logits.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "before-action.athanor"
            save(r, path)
            other, _ = restore(path)
        r.step()
        other.step()
        before, _, _, frame, prior = r.last_transition
        embedding = observer.sensory.activation(
            pixels(before), torch.tensor(before.proprio), intent_activation(observer.model, frame)
        )
        reproduced = observer.model([frame], [prior], sensory=[embedding])[0]
        self.assertTrue(torch.equal(expected, reproduced.operation_logits))
        self.assertTrue(torch.equal(prior.field, other.last_transition[-1].field))
        self.assertTrue(
            torch.equal(frame.intent_features, other.last_transition[-2].intent_features)
        )
