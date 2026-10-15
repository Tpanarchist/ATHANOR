"""Atomic complete-state checkpoint; retains the original observer checkpoint format."""

import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import torch

from athanor import PersistenceError, Ref, ValidationError
from athanor.neural import Observer
from athanor.neural.checkpoint import _tensor
from athanor.neural.contracts import Activation
from athanor.neural.encoding import Frame
from athanor.world import SensorFrame, World
from athanor.world.engine import ACTIONS

from .discovery import Discovery
from .induction import Induction
from .perception import SensoryEncoder, perceive
from .processes import Processes
from .representations import Representations
from .runtime import GroundedObserver, PixelObserver, Runtime, Track, contextual_frame
from .training import GroundedTrainer


def ref_data(ref: Ref) -> list[str | int]:
    return [ref.id, ref.revision, ref.namespace]


def sensor_data(frame: SensorFrame | None) -> dict[str, object] | None:
    if frame is None:
        return None
    return {
        "tick": frame.tick,
        "width": frame.width,
        "height": frame.height,
        "left": frame.left.hex(),
        "right": frame.right.hex(),
        "focal": frame.focal,
        "baseline": frame.baseline,
        "proprio": frame.proprio,
        "executed": frame.executed,
    }


def finite(value: object) -> bool:
    """Import checks apply to runtime metadata and optimizer state as well as weights."""
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(v) for v in value)
    return True


def save(runtime: Runtime, path: str | Path, *, trainer: GroundedTrainer | None = None) -> None:
    trainer = runtime.discovery_trainer if trainer is None else trainer
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    grounded = runtime.observer
    if not finite(grounded.processes._retained):
        raise PersistenceError("Non-finite process retention anchors")
    transition_meta = None
    transition_tensors = None
    if runtime.last_transition is not None:
        before, after, action, frame, prior = runtime.last_transition
        if frame is not None and prior is not None:
            transition_meta = {
                "before": sensor_data(before),
                "after": sensor_data(after),
                "action": action,
                "refs": [ref_data(v) for v in frame.refs],
                "query": frame.query_index,
                "operations": frame.operation_indices,
                "candidates": frame.candidate_indices,
                "arities": frame.arities,
                "meanings": frame.meaning_indices,
                "state_refs": [ref_data(v) for v in prior.refs],
                "steps": prior.steps,
            }
            transition_tensors = {
                "features": frame.features,
                "edges": frame.edge_index,
                "edge_features": frame.edge_features,
                "intent": frame.intent_features,
                "field": prior.field,
                "context": prior.context,
            }
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".grounded-") as directory:
        observer_path = Path(directory) / "observer.athanor"
        grounded.observer.save(observer_path)
        payload = io.BytesIO()
        torch.save(
            {
                "parameters": grounded.observer.sensory.state_dict(),
                "trainable": {
                    name: p.requires_grad
                    for name, p in grounded.observer.sensory.named_parameters()
                },
                "optimizer": None if trainer is None else trainer.optimizer.state_dict(),
                "rng": None if trainer is None else trainer.rng.getstate(),
                "updates": None if trainer is None else trainer.updates,
                "process_retention": grounded.processes._retained,
                "transition": transition_tensors,
                "inference_prior": {
                    "refs": [ref_data(v) for v in grounded.observer.input_state.refs],
                    "field": grounded.observer.input_state.field,
                    "context": grounded.observer.input_state.context,
                    "steps": grounded.observer.input_state.steps,
                },
            },
            payload,
        )
        metadata = {
            "version": 6,
            "representations": grounded.representations.metadata(),
            "induction": grounded.induction.metadata(),
            "discovery": grounded.discovery.metadata(),
            "processes": grounded.processes.metadata(),
            "transition": transition_meta,
            "operations": {k: ref_data(v) for k, v in grounded.operations.items()},
            "target_anchor": ref_data(grounded.target_anchor),
            "update_track": ref_data(grounded.update_track),
            "near": ref_data(grounded.near),
            "goal": grounded.goal,
            "target": None if grounded.target is None else ref_data(grounded.target),
            "prediction": grounded.prediction,
            "last_request": None
            if grounded.last_request is None
            else ref_data(grounded.last_request),
            "last_feedback": None
            if grounded.last_feedback is None
            else ref_data(grounded.last_feedback),
            "history": [ref_data(v) for v in grounded.history],
            "motion_evidence": [ref_data(v) for v in grounded.motion_evidence],
            "experience": grounded.experience,
            "consolidations": {k: ref_data(v) for k, v in grounded.consolidations.items()},
            "tracks": [
                {
                    "ref": ref_data(t.ref),
                    "appearance": t.appearance,
                    "relative": t.relative,
                    "confidence": t.confidence,
                    "last_seen": t.last_seen,
                    "status": t.status,
                }
                for t in grounded.tracks
            ],
            "sensor": sensor_data(grounded.observer.sensor),
            "paused": runtime.paused,
            "running": runtime.running,
        }
        members = {
            "world.json": runtime.world.dumps().encode(),
            "observer.athanor": observer_path.read_bytes(),
            "sensory.pt": payload.getvalue(),
            "runtime.json": json.dumps(metadata, allow_nan=False).encode(),
        }
        members["manifest.json"] = json.dumps(
            {name: hashlib.sha256(data).hexdigest() for name, data in members.items()}
        ).encode()
        temporary = Path(directory) / "complete.tmp"
        with temporary.open("wb") as stream:
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in members.items():
                    archive.writestr(name, data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)


def restore(path: str | Path) -> tuple[Runtime, GroundedTrainer | None]:
    try:
        with zipfile.ZipFile(path) as archive:
            expected = {
                "world.json",
                "observer.athanor",
                "sensory.pt",
                "runtime.json",
                "manifest.json",
            }
            if set(archive.namelist()) != expected or len(archive.namelist()) != len(expected):
                raise PersistenceError("Unexpected grounded checkpoint members")
            if any(item.file_size > 256 * 1024 * 1024 for item in archive.infolist()):
                raise PersistenceError("Grounded checkpoint exceeds import budget")
            members = {name: archive.read(name) for name in expected}
        hashes = json.loads(members.pop("manifest.json"))
        if hashes != {name: hashlib.sha256(data).hexdigest() for name, data in members.items()}:
            raise PersistenceError("Grounded checkpoint digest mismatch")
        metadata: dict[str, Any] = json.loads(members["runtime.json"])
        expected_metadata = {
            "version",
            "transition",
            "operations",
            "target_anchor",
            "update_track",
            "near",
            "goal",
            "target",
            "prediction",
            "last_request",
            "last_feedback",
            "history",
            "motion_evidence",
            "experience",
            "consolidations",
            "tracks",
            "sensor",
            "paused",
            "running",
        }
        if metadata.get("version") in (3, 4, 5, 6):
            expected_metadata.add("discovery")
        if metadata.get("version") in (4, 5, 6):
            expected_metadata.add("processes")
        if metadata.get("version") in (5, 6):
            expected_metadata.add("induction")
        if metadata.get("version") == 6:
            expected_metadata.add("representations")
        if (
            type(metadata) is not dict
            or set(metadata) != expected_metadata
            or type(metadata["version"]) is not int
            or metadata["version"] not in (1, 2, 3, 4, 5, 6)
        ):
            raise PersistenceError("Unsupported grounded checkpoint version")
        if (
            not finite(metadata)
            or type(metadata["paused"]) is not bool
            or type(metadata["running"]) is not bool
            or metadata["goal"] not in ("approach", "observe", "search", "unknown", *ACTIONS)
            or set(metadata["operations"]) != set(ACTIONS)
        ):
            raise PersistenceError("Invalid grounded runtime metadata")
        if metadata["prediction"] is not None and len(metadata["prediction"]) != 6:
            raise PersistenceError("Invalid physical expectation")
        world = World.loads(members["world.json"].decode())
        with tempfile.TemporaryDirectory(prefix="athanor-restore-") as directory:
            observer_path = Path(directory) / "observer.athanor"
            observer_path.write_bytes(members["observer.athanor"])
            observer, _ = Observer.restore(observer_path)
        sensory = SensoryEncoder(observer.model.config.width)
        payload = torch.load(
            io.BytesIO(members["sensory.pt"]), map_location="cpu", weights_only=True
        )
        expected_payload = {"parameters", "trainable", "optimizer", "rng", "updates", "transition"}
        if metadata["version"] >= 2:
            expected_payload.add("inference_prior")
        if metadata["version"] >= 4:
            expected_payload.add("process_retention")
        if set(payload) != expected_payload:
            raise PersistenceError("Malformed sensory training state")
        if not finite(payload):
            raise PersistenceError("Non-finite grounded training state")
        if (
            (payload["optimizer"] is None) != (payload["rng"] is None)
            or (payload["optimizer"] is None) != (payload["updates"] is None)
            or payload["updates"] is not None
            and (type(payload["updates"]) is not int or payload["updates"] < 0)
        ):
            raise PersistenceError("Invalid optimizer continuation state")
        if any(not bool(torch.isfinite(value).all()) for value in payload["parameters"].values()):
            raise PersistenceError("Non-finite sensory parameters")
        sensory.load_state_dict(payload["parameters"], strict=True)
        if set(payload["trainable"]) != set(dict(sensory.named_parameters())) or any(
            type(v) is not bool for v in payload["trainable"].values()
        ):
            raise PersistenceError("Invalid sensory trainability state")
        for name, parameter in sensory.named_parameters():
            parameter.requires_grad_(payload["trainable"][name])
        grounded = object.__new__(GroundedObserver)
        pixel = PixelObserver(
            observer.knowledge, observer.model, sensory, self_ref=observer.self_ref
        )
        pixel.__dict__.update(observer.__dict__)
        grounded.observer = pixel

        def ref(raw: list[Any]) -> Ref:
            result = Ref(raw[0], raw[1], raw[2])
            observer.knowledge.get(result)
            return result

        pixel.input_state = observer.state.detached()
        if metadata["version"] >= 2:
            saved_prior = payload["inference_prior"]
            if (
                set(saved_prior) != {"refs", "field", "context", "steps"}
                or type(saved_prior["steps"]) is not int
                or saved_prior["steps"] < 0
            ):
                raise PersistenceError("Invalid inference input activation")
            prior_refs = tuple(ref(v) for v in saved_prior["refs"])
            if len(set(prior_refs)) != len(prior_refs):
                raise PersistenceError("Duplicate inference input handles")
            pixel.input_state = Activation(
                prior_refs,
                _tensor(saved_prior["field"], (len(prior_refs), observer.model.config.width)),
                _tensor(saved_prior["context"], (observer.model.config.width,)),
                saved_prior["steps"],
            )

        grounded.operations = {name: ref(value) for name, value in metadata["operations"].items()}
        for name in ("target_anchor", "update_track", "near"):
            setattr(grounded, name, ref(metadata[name]))
        for name in ("target", "last_request", "last_feedback"):
            setattr(grounded, name, None if metadata[name] is None else ref(metadata[name]))
        grounded.goal = metadata["goal"]
        grounded.prediction = (
            None if metadata["prediction"] is None else tuple(metadata["prediction"])
        )
        grounded.history = [ref(value) for value in metadata["history"]]
        grounded.motion_evidence = [ref(value) for value in metadata["motion_evidence"]]
        grounded.experience = {name: tuple(value) for name, value in metadata["experience"].items()}
        grounded.consolidations = {
            name: ref(value) for name, value in metadata["consolidations"].items()
        }
        grounded.tracks = [
            Track(
                ref(row["ref"]),
                tuple(row["appearance"]),
                None if row["relative"] is None else tuple(row["relative"]),
                row["confidence"],
                row["last_seen"],
                row["status"],
            )
            for row in metadata["tracks"]
        ]
        if len({track.ref.id for track in grounded.tracks}) != len(grounded.tracks) or any(
            len(track.appearance) != 3
            or not all(0 <= v <= 1 for v in track.appearance)
            or track.relative is not None
            and len(track.relative) != 3
            or not 0 <= track.confidence <= 1
            or type(track.last_seen) is not int
            or track.last_seen < 0
            or track.status
            not in {
                "visible",
                "ambiguous",
                "monocular",
                "partial_observation",
                "depth_unknown",
                "ambiguous_identity",
                "occluded_hypothesis",
            }
            for track in grounded.tracks
        ):
            raise PersistenceError("Invalid persistent perceptual hypotheses")
        sensor = metadata["sensor"]
        if sensor is not None:
            pixel.sensor = SensorFrame(
                sensor["tick"],
                sensor["width"],
                sensor["height"],
                bytes.fromhex(sensor["left"]),
                bytes.fromhex(sensor["right"]),
                sensor["focal"],
                sensor["baseline"],
                tuple(sensor["proprio"]),
                sensor["executed"],
            )
        grounded.last_percepts = () if pixel.sensor is None else perceive(sensory, pixel.sensor)
        grounded.discovery = (
            Discovery.restore(grounded, metadata["discovery"])
            if metadata["version"] >= 3
            else Discovery(grounded)
        )
        grounded.processes = (
            Processes.restore(grounded, metadata["processes"])
            if metadata["version"] >= 4
            else Processes(grounded)
        )
        grounded.induction = (
            Induction.restore(grounded, metadata["induction"])
            if metadata["version"] >= 5
            else Induction(grounded)
        )
        grounded.representations = (
            Representations.restore(grounded, metadata["representations"])
            if metadata["version"] >= 6
            else Representations(grounded)
        )
        if (
            metadata["version"] >= 4
            and grounded.processes.neural_updates > 0
            and payload["process_retention"] is None
        ):
            raise PersistenceError("Missing trained process retention anchors")
        if metadata["version"] >= 4 and payload["process_retention"] is not None:
            retention = payload["process_retention"]
            if (
                type(retention) is not tuple
                or len(retention) != 3
                or not isinstance(retention[0], torch.Tensor)
                or retention[0].ndim != 2
                or not 1 <= retention[0].shape[0] <= 16384
            ):
                raise PersistenceError("Invalid process retention anchors")
            n, width = retention[0].shape[0], observer.model.config.width
            grounded.processes._retained = (
                _tensor(retention[0], (n, width)),
                _tensor(retention[1], (n, width)),
                _tensor(retention[2], (n,)),
            )
        if pixel._frame is not None and pixel.sensor is not None:
            pixel._frame = contextual_frame(pixel._frame, pixel.knowledge)
        runtime = object.__new__(Runtime)
        runtime.world, runtime.observer = world, grounded
        runtime.last_transition = None
        transition = metadata["transition"]
        tensors = payload["transition"]
        if (transition is None) != (tensors is None):
            raise PersistenceError("Transition metadata/tensors disagree")
        if transition is not None:
            if (
                transition["action"] not in ACTIONS
                or type(transition["steps"]) is not int
                or transition["steps"] < 0
                or len(transition["arities"]) != len(transition["operations"])
                or any(type(n) is not int or n < 0 for n in transition["arities"])
            ):
                raise PersistenceError("Invalid pending experience interface")

            def sensor_read(value: dict[str, Any]) -> SensorFrame:
                return SensorFrame(
                    value["tick"],
                    value["width"],
                    value["height"],
                    bytes.fromhex(value["left"]),
                    bytes.fromhex(value["right"]),
                    value["focal"],
                    value["baseline"],
                    tuple(value["proprio"]),
                    value["executed"],
                )

            refs = tuple(ref(v) for v in transition["refs"])
            n = len(refs)
            if not 0 < n <= observer.model.config.max_nodes or len(set(refs)) != n:
                raise PersistenceError("Invalid pending training frame")
            from athanor.neural.encoding import EDGE_SIZE, FEATURE_SIZE

            edges = tensors["edges"]
            if (
                not isinstance(edges, torch.Tensor)
                or edges.dtype != torch.int64
                or edges.ndim != 2
                or edges.shape[0] != 2
                or edges.numel()
                and (int(edges.min()) < 0 or int(edges.max()) >= n)
            ):
                raise PersistenceError("Invalid pending relational edges")
            indices = [
                transition["query"],
                *transition["operations"],
                *transition["candidates"],
                *transition["meanings"],
            ]
            if any(type(i) is not int or not 0 <= i < n for i in indices):
                raise PersistenceError("Invalid pending frame pointers")
            frame = Frame(
                refs,
                _tensor(tensors["features"], (n, FEATURE_SIZE)),
                edges,
                _tensor(tensors["edge_features"], (edges.shape[1], EDGE_SIZE)),
                transition["query"],
                tuple(transition["operations"]),
                tuple(transition["candidates"]),
                tuple(transition["arities"]),
                tuple(transition["meanings"]),
                None if tensors["intent"] is None else _tensor(tensors["intent"], (FEATURE_SIZE,)),
            )
            state_refs = tuple(ref(v) for v in transition["state_refs"])
            prior = Activation(
                state_refs,
                _tensor(tensors["field"], (len(state_refs), observer.model.config.width)),
                _tensor(tensors["context"], (observer.model.config.width,)),
                transition["steps"],
            )
            runtime.last_transition = (
                sensor_read(transition["before"]),
                sensor_read(transition["after"]),
                transition["action"],
                frame,
                prior,
            )
        runtime.paused, runtime.running = metadata["paused"], metadata["running"]
        trainer = None
        if payload["optimizer"] is not None:
            trainer = GroundedTrainer(grounded)
            trainer.optimizer.load_state_dict(payload["optimizer"])
            trainer.rng.setstate(payload["rng"])
            assert type(payload["updates"]) is int
            trainer.updates = payload["updates"]
        runtime.discovery_trainer = trainer
        return runtime, trainer
    except PersistenceError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
        ValidationError,
        zipfile.BadZipFile,
        EOFError,
    ) as exc:
        raise PersistenceError(f"Invalid grounded checkpoint: {exc}") from exc
