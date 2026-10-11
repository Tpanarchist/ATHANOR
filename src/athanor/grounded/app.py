"""Local interactive viewer and headless application over the same real runtime."""

import argparse
import json
import threading
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from athanor import AthanorError, Record, Ref, ValidationError
from athanor.world import Action
from athanor.world.engine import vector

from .checkpoint import restore, save, sensor_data
from .runtime import Runtime
from .training import GroundedTrainer


def json_value(value: object) -> object:
    if isinstance(value, Fraction):
        return float(value)
    if isinstance(value, Ref):
        return str(value)
    if isinstance(value, Record):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_value(item) for item in value]
    return value


class Application:
    def __init__(
        self, runtime: Runtime, checkpoint: Path, trainer: GroundedTrainer | None = None
    ) -> None:
        self.runtime, self.checkpoint = runtime, checkpoint
        self.trainer = GroundedTrainer(runtime.observer) if trainer is None else trainer
        runtime.discovery_trainer = self.trainer
        self.last_step: dict[str, object] | None = None
        self.lock = threading.RLock()

    def state(self) -> dict[str, object]:
        with self.lock:
            r, o = self.runtime, self.runtime.observer
            action, decision = o.propose(commit=False)
            sensor = o.observer.sensor
            return {
                "version": "0.5.0",
                "tick": r.world.tick,
                "paused": r.paused,
                "running": r.running,
                "training_updates": self.trainer.updates,
                "sensor": sensor_data(sensor),
                "territory": {
                    "body": r.world.position,
                    "yaw": r.world.yaw,
                    "objects": [
                        {
                            "position": vector(node.fields["position"]),
                            "size": vector(node.fields["size"]),
                            "color": json_value(node.fields["color"]),
                            "shape": node.fields["shape"],
                        }
                        for node in (r.world.knowledge.get(ref) for ref in r.world.objects)
                    ],
                },
                "tracks": [
                    {
                        "id": str(t.ref),
                        "status": t.status,
                        "confidence": t.confidence,
                        "relative": t.relative,
                    }
                    for t in o.tracks
                ],
                "interpretation": {
                    "goal": o.goal,
                    "target_concept": None if o.target is None else str(o.target),
                    "learned_action": action.name,
                    "confidence": decision.confidence,
                    "prediction": o.prediction,
                    "percepts": [
                        {
                            "index": i,
                            "status": p.status,
                            "appearance": p.appearance,
                            "relative": p.relative,
                        }
                        for i, p in enumerate(o.last_percepts)
                    ],
                },
                "feedback": self.last_step
                if self.last_step is not None
                else None
                if o.last_feedback is None
                else json_value(o.knowledge.get(o.last_feedback).fields),
                "self_map": [
                    json_value(o.knowledge.get(ref).fields) for ref in o.observer.self_map[-4:]
                ],
                "discovery": {
                    "status": o.discovery.status,
                    "focus": o.discovery.focus,
                    "remaining": o.discovery.budget - o.discovery.spent,
                    "spent": o.discovery.spent,
                    "priority_trained": o.discovery.priority_trained,
                    "learning": o.discovery.learning,
                    "neural_updates": o.discovery.neural_updates,
                    "hypotheses": [
                        json_value(o.knowledge.get(v.hypothesis).fields)
                        for v in o.discovery.estimates.values()
                        if v.hypothesis is not None
                    ],
                    "experiments": [
                        json_value(o.knowledge.get(v).fields) for v in o.discovery.options
                    ],
                    "scores": o.discovery.last_scores,
                    "selected": json_value(o.discovery.selected),
                    "knowledge": [
                        json_value(o.knowledge.get(v.relationship).fields)
                        for v in o.discovery.estimates.values()
                        if v.relationship is not None
                    ],
                    "evidence": [
                        json_value(o.knowledge.get(v).fields) for v in o.discovery.history[-8:]
                    ],
                    "activation": o.observer.activation_history[-1:],
                },
            }

    def command(self, data: dict[str, object]) -> dict[str, object]:
        with self.lock:
            op = data.get("op")
            if type(op) is not str:
                raise ValidationError("A command requires an operation name")
            r = self.runtime
            message = "Runtime updated"
            if op == "start":
                r.running, r.paused = True, False
                if r.observer.discovery.status == "paused":
                    r.observer.discovery.resume()
            elif op == "stop":
                r.running, r.paused = False, True
                r.observer.discovery.stop()
            elif op == "pause":
                r.paused = True
                r.observer.discovery.pause()
            elif op in {"explore", "investigate"}:
                budget = data.get("budget", 24)
                if type(budget) is not int:
                    raise ValidationError("Investigation budget must be an integer")
                focus = data.get("focus") if op == "investigate" else None
                if focus is not None and type(focus) is not str:
                    raise ValidationError("Investigation focus must be a channel name")
                r.start_discovery(budget, focus=focus)
                message = "Autonomous inquiry started with a bounded action budget"
            elif op == "resume":
                r.observer.discovery.resume()
                r.running, r.paused = True, False
                message = "Resumed inquiry with its retained knowledge and remaining budget"
            elif op == "stop_inquiry":
                r.observer.discovery.stop()
                r.paused = True
            elif op == "pause_learning":
                r.observer.discovery.learning = False
                message = "Neural consolidation paused; evidence remains observable"
            elif op == "resume_learning":
                r.observer.discovery.learning = True
                message = "Continual neural consolidation resumed"
            elif op == "train_inquiry":
                from .training import train_inquiry

                result = train_inquiry(r.observer)
                message = f"Learned general experimental priorities: {result}"
            elif op == "think":
                r.observer.observer.think()
                message = "Refined current activation without advancing the world"
            elif op in {"step", "auto", "manual"}:
                if op == "auto" and (r.paused or not r.running):
                    return {"message": "Paused"}
                manual = Action(str(data["action"])) if op == "manual" else None
                self.last_step = (
                    r.autonomous_step()
                    if op == "auto" and r.observer.discovery.status != "stopped"
                    else r.step(manual)
                )
                message = (
                    "No action: " + str(self.last_step.get("reason", "inquiry paused"))
                    if self.last_step.get("acted") is False
                    else "External action executed"
                    if manual
                    else "Learned proposal executed"
                )
            elif op == "instruct":
                message = r.observer.instruct(str(data.get("text", "")))
                r.observer.observe(r.world.sense())
            elif op == "teach":
                index = data.get("index", 0)
                if type(index) is not int:
                    raise ValidationError("Teaching index must be an integer")
                concept = r.observer.teach_visible(str(data.get("text", "")), index)
                r.observer.observer.think()
                message = (
                    f"Installed camera-grounded example: {concept}; reliability remains measured"
                )
            elif op == "save":
                save(r, self.checkpoint, trainer=self.trainer)
                message = "Saved world, map, parameters and active state"
            elif op == "restore":
                self.runtime, trainer = restore(self.checkpoint)
                self.trainer = (
                    GroundedTrainer(self.runtime.observer) if trainer is None else trainer
                )
                self.runtime.discovery_trainer = self.trainer
                self.last_step = None
                message = "Restored complete state"
            elif op == "train":
                if r.last_transition is None:
                    raise ValidationError("Take an action before learning from its consequence")
                before, after, action, frame, prior = r.last_transition
                discrepancy = self.trainer.learn_transition(
                    before,
                    after,
                    action,
                    frame=frame,
                    prior=prior,
                )
                r.observer.observer.think()
                message = (
                    f"Actual parameters updated from measured feedback; loss {discrepancy:.5f}"
                )
            else:
                raise ValidationError("Unknown application command")
            return {"message": message}


def handler(application: Application) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def reply(self, body: bytes, kind: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/":
                self.reply(
                    Path(__file__).with_name("viewer.html").read_bytes(), "text/html; charset=utf-8"
                )
            elif self.path == "/state":
                self.reply(
                    json.dumps(application.state(), allow_nan=False).encode(), "application/json"
                )
            else:
                self.reply(b"Not found", "text/plain", 404)

        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if self.path != "/command" or not 0 < length <= 8192:
                    raise ValidationError("Invalid bounded command request")
                data = json.loads(self.rfile.read(length))
                if type(data) is not dict:
                    raise ValidationError("Commands must be JSON objects")
                response = application.command(data)
                self.reply(json.dumps(response).encode(), "application/json")
            except (AthanorError, ValueError, KeyError, OSError) as exc:
                self.reply(json.dumps({"error": str(exc)}).encode(), "application/json", 400)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/grounded-observer.athanor")
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--paused", action="store_true", help="Open with inquiry execution paused")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--explore", action="store_true")
    parser.add_argument("--focus", choices=("object", "self", "interface"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    runtime, trainer = restore(args.checkpoint)
    application = Application(runtime, args.checkpoint, trainer)
    if args.paused:
        runtime.paused = True
        runtime.observer.discovery.pause()
    if args.explore:
        runtime.start_discovery(max(1, args.steps), focus=args.focus)
    if args.headless:
        if args.steps < 0:
            parser.error("Steps must be nonnegative")
        for _ in range(args.steps):
            application.command({"op": "auto" if args.explore else "step"})
        print(json.dumps(application.state(), indent=2))
    else:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(application))
        print(f"ATHANOR grounded observer: http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
