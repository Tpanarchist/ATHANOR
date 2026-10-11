"""Complete learning acceptance, including a freshly trained sensory extension."""

import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

AVAILABLE = importlib.util.find_spec("torch") is not None
if AVAILABLE:
    from athanor.grounded.demo import run_demo
    from athanor.semantics.demo import run_demo as train_prerequisite


@unittest.skipUnless(AVAILABLE, "Install the neural extra for grounded learning acceptance")
class GroundedLearningAcceptance(unittest.TestCase):
    def test_complete_grounded_learning_and_continuation(self) -> None:
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            base = Path("artifacts/meaning-observer.athanor")
            if not base.is_file():
                base = Path(directory) / "meaning.athanor"
                train_prerequisite(base)
            report = run_demo(Path(directory) / "grounded.athanor", base_checkpoint=base)
            self.assertGreater(report["vision_after"]["foreground_iou"], 0.95)
            self.assertGreaterEqual(report["trained_actions"]["operation_accuracy"], 0.85)
            self.assertGreaterEqual(report["retained_structural"]["proposal"], 0.90)
            self.assertGreater(report["initial_approach_progress"], 0.2)
            self.assertGreater(report["transfer_progress"], 0)
            self.assertTrue(report["checkpoint_next_step_identical"])
            self.assertFalse(report["teacher_required_during_inference"])
            self.assertTrue(all(not step["manual"] for step in report["trace"]))


if __name__ == "__main__":
    unittest.main()
