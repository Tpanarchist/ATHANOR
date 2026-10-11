import json
import subprocess
import sys
import unittest
from pathlib import Path

from athanor import ValidationError
from athanor.world import Action, SensorFrame, World
from athanor.world.geometry import box_hit, rigid_overlap, sphere_hit


class WorldContracts(unittest.TestCase):
    def test_analytic_geometry(self) -> None:
        self.assertEqual(sphere_hit((0, 0, 0), (0, 0, 1), (0, 0, 3), 1), 2)
        self.assertEqual(box_hit((0, 0, 0), (0, 0, 1), (0, 0, 3), (1, 1, 1)), 2)
        self.assertIsNone(sphere_hit((0, 0, 0), (0, 0, 1), (3, 0, 3), 1))
        self.assertTrue(
            rigid_overlap((0, 0, 0), "box", (1, 0.2, 1), (0, 0.3, 0), "box", (0.2, 0.2, 0.2))
        )
        self.assertFalse(
            rigid_overlap((0, 0, 0), "box", (1, 0.2, 1), (0, 0.5, 0), "box", (0.2, 0.2, 0.2))
        )

    def test_stereo_pixels_have_no_privileged_handles(self) -> None:
        world = World()
        world.add_object((0.6, 0, 3))
        sensor = world.sense()
        self.assertNotEqual(sensor.left, sensor.right)
        self.assertEqual(
            set(SensorFrame.__dataclass_fields__),
            {
                "tick",
                "width",
                "height",
                "left",
                "right",
                "focal",
                "baseline",
                "proprio",
                "executed",
            },
        )
        other = World(namespace="unrelated IDs")
        other.add_object((0.6, 0, 3))
        self.assertEqual(other.sense(), sensor)

    def test_fixed_steps_pinned_identity_and_receipts(self) -> None:
        world = World()
        old = world.body
        event = world.step(Action("forward"))
        self.assertEqual(world.body.id, old.id)
        self.assertGreater(world.body.revision, old.revision)
        self.assertEqual(world.position, (0, 0, 0.25))
        self.assertEqual(world.knowledge.get(old).fields["position"], (0, 0, 0))
        self.assertEqual(world.knowledge.get(event).fields["intended"], "forward")
        self.assertTrue(world.knowledge.provenance(world.body))

    def test_swept_collision_and_proprioception(self) -> None:
        world = World()
        world.add_object((0, 0, 0.8), shape="box", size=(0.5, 0.5, 0.05))
        for _ in range(4):
            world.step(Action("forward"))
        self.assertLess(world.position[2], 0.56)
        self.assertEqual(world.sense().proprio[4], 1)

    def test_deterministic_restore_including_camera_fault(self) -> None:
        a = World()
        a.add_object((0.7, 0, 3), velocity=(0.4, 0, 0))
        a.sense()
        a.camera_hold = True
        b = World.loads(a.dumps())
        for name in ("forward", "turn_left", "right", "interact", "wait"):
            a.step(Action(name))
            b.step(Action(name))
            self.assertEqual(a.dumps(), b.dumps())
            self.assertEqual(a.sense(), b.sense())

    def test_restore_preserves_capture_cache_before_a_later_camera_hold(self) -> None:
        world = World()
        world.add_object((0, 0, 3))
        world.sense()
        world.step(Action("forward"))
        restored = World.loads(world.dumps())
        self.assertEqual(world.dumps(), restored.dumps())
        world.camera_hold = restored.camera_hold = True
        self.assertEqual(world.sense(), restored.sense())

    def test_occlusion_does_not_destroy_territory(self) -> None:
        w = World()
        target = w.add_object((0, 0, 3))
        w.add_object((0, 0, 1.5), shape="box", size=(0.6, 0.6, 0.1))
        _, _, labels = w.render(labels=True)
        self.assertFalse(any(1 in eye for eye in labels))
        self.assertIn(target, w.objects)
        for _ in range(4):
            w.step(Action("left"))
        _, _, labels = w.render(labels=True)
        self.assertTrue(any(1 in eye for eye in labels))

    def test_interaction_changes_reachable_object(self) -> None:
        w = World()
        ref = w.add_object((0, 0, 0.9), size=(0.2, 0.2, 0.2), movable=True)
        w.step(Action("interact"))
        new = w.objects[0]
        self.assertEqual(new.id, ref.id)
        self.assertGreater(new.revision, ref.revision)
        self.assertGreater(
            w.knowledge.get(new).fields["position"][2], w.knowledge.get(ref).fields["position"][2]
        )

    def test_invalid_inputs_preserve_world(self) -> None:
        w = World()
        before = w.dumps()
        with self.assertRaises(ValidationError):
            w.add_object((9, 0, 0))
        with self.assertRaises(ValidationError):
            w.step("jump")  # type: ignore[arg-type]
        self.assertEqual(w.dumps(), before)
        with self.assertRaises(ValidationError):
            Action("jump")
        with self.assertRaises(ValidationError):
            w.add_object((0, 0, 3), size=(0.2, 0.3, 0.4))
        with self.assertRaises(ValidationError):
            w.relocate(w.body, (0, 0))
        self.assertEqual(w.dumps(), before)

    def test_push_does_not_pass_through_another_object(self) -> None:
        w = World()
        ref = w.add_object((0, 0, 0.8), size=(0.2, 0.2, 0.2), movable=True)
        w.add_object((0, 0, 1.3), shape="box", size=(0.3, 0.3, 0.2))
        w.step(Action("interact"))
        self.assertEqual(w.objects[0], ref)
        self.assertFalse(w.sense().proprio[4])

    def test_malformed_native_state_rejects(self) -> None:
        data = json.loads(World().dumps())
        data["yaw"] = float("nan")
        with self.assertRaises(ValidationError):
            World.loads(json.dumps(data))
        data = json.loads(World().dumps())
        data["tick"] = -1
        with self.assertRaises(ValidationError):
            World.loads(json.dumps(data))

    def test_world_import_and_execution_without_third_party_modules(self) -> None:
        src = str(Path(__file__).resolve().parents[1] / "src")
        code = (
            f"import sys;sys.path.insert(0,{src!r});"
            "from athanor.world import World;w=World();w.step();"
            "assert 'torch' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-S", "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
