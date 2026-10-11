"""Atomic local checkpoints: kernel knowledge, parameters, activations and optimizer."""

import hashlib
import io
import json
import os
import pickle
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from athanor import Athanor, AthanorError, PersistenceError, Ref

from .contracts import Activation, ModelConfig, ModelOutput, Observation
from .model import NeuralObserverModel
from .observer import Observer
from .training import Trainer, TrainingConfig

VERSION = 3


def _ref(ref: Ref) -> list[str | int]:
    return [ref.id, ref.revision, ref.namespace]


def _read_ref(value: Any, knowledge: Athanor) -> Ref:
    if (
        type(value) is not list
        or len(value) != 3
        or type(value[0]) is not str
        or type(value[1]) is not int
        or type(value[2]) is not str
    ):
        raise PersistenceError("Malformed checkpoint reference")
    ref = Ref(value[0], value[1], value[2])
    knowledge.get(ref)
    return ref


def _tensor(value: Any, shape: tuple[int, ...]) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.dtype != torch.float32
        or tuple(value.shape) != shape
        or not bool(torch.isfinite(value).all())
    ):
        raise PersistenceError("Invalid checkpoint tensor dimensions, dtype or finite values")
    return value


def save_checkpoint(observer: Observer, path: Path, *, trainer: Trainer | None = None) -> None:
    if trainer is not None and trainer.model is not observer.model:
        raise PersistenceError("Optimizer belongs to another model")
    if observer.observation is not None:
        current = observer.builder.build(
            observer.knowledge,
            observer.observation,
            self_refs=tuple(observer.self_map),
            observer=observer.self_ref,
            operations=observer.active_operations,
        )
        if current.refs != observer.state.refs:
            raise PersistenceError("Process the current observation after teaching before saving")
    output = observer._output
    payload: dict[str, Any] = {
        "parameters": observer.model.state_dict(),
        "field": observer.state.field.cpu(),
        "context": observer.state.context.cpu(),
        "output": None
        if output is None
        else {
            "operation_logits": output.operation_logits.cpu(),
            "argument_logits": tuple(value.cpu() for value in output.argument_logits),
            "applicability_logits": output.applicability_logits.cpu(),
            "success_logit": output.success_logit.cpu(),
            "preference_logit": output.preference_logit.cpu(),
            "outcome": output.outcome.cpu(),
            "operation_index": output.operation_index,
            "argument_indices": output.argument_indices,
            "meaning_logits": None
            if output.meaning_logits is None
            else output.meaning_logits.cpu(),
        },
        "optimizer": None if trainer is None else trainer.optimizer.state_dict(),
        "random_state": None if trainer is None else trainer.rng.getstate(),
    }
    tensor_buffer = io.BytesIO()
    torch.save(payload, tensor_buffer)
    tensor_bytes = tensor_buffer.getvalue()
    knowledge_bytes = observer.knowledge.dumps().encode("utf-8")
    observation = observer.observation
    metadata = {
        "version": 2 if observer.active_operations is None else VERSION,
        "active_operations": None
        if observer.active_operations is None
        else [_ref(ref) for ref in observer.active_operations],
        "model": asdict(observer.model.config),
        "training": None
        if trainer is None
        else {"config": asdict(trainer.config), "updates": trainer.updates},
        "self": _ref(observer.self_ref),
        "self_map": [_ref(ref) for ref in observer.self_map],
        "state_refs": [_ref(ref) for ref in observer.state.refs],
        "steps": observer.state.steps,
        "activation_history": observer.activation_history,
        "capabilities": [
            [_ref(ref), attempts, successes]
            for ref, (attempts, successes) in observer.capabilities.items()
        ],
        "observation": None
        if observation is None
        else {
            "ref": _ref(observation.ref),
            "candidates": [_ref(ref) for ref in observation.candidates],
        },
        "hashes": {
            "knowledge.json": hashlib.sha256(knowledge_bytes).hexdigest(),
            "tensors.pt": hashlib.sha256(tensor_bytes).hexdigest(),
        },
    }
    if observer.active_operations is None:
        metadata.pop("active_operations")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}-", suffix=".tmp", delete=False
        ) as file:
            temporary = file.name
            with zipfile.ZipFile(file, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("metadata.json", json.dumps(metadata, allow_nan=False))
                archive.writestr("knowledge.json", knowledge_bytes)
                archive.writestr("tensors.pt", tensor_bytes)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def restore_checkpoint(path: Path) -> tuple[Observer, Trainer | None]:
    try:
        with zipfile.ZipFile(path) as archive:
            if len(archive.namelist()) != 3 or set(archive.namelist()) != {
                "metadata.json",
                "knowledge.json",
                "tensors.pt",
            }:
                raise PersistenceError("Unexpected checkpoint members")
            # Explicit import budgets prevent an accidentally enormous local checkpoint.
            if any(item.file_size > 256 * 1024 * 1024 for item in archive.infolist()):
                raise PersistenceError("Checkpoint member exceeds import budget")
            metadata = json.loads(archive.read("metadata.json"))
            if (
                set(metadata)
                != (
                    {
                        "version",
                        "model",
                        "training",
                        "self",
                        "self_map",
                        "state_refs",
                        "steps",
                        "activation_history",
                        "capabilities",
                        "observation",
                        "hashes",
                    }
                    | ({"active_operations"} if metadata.get("version") == 3 else set())
                )
                or type(metadata["version"]) is not int
                or metadata["version"] not in (1, 2, VERSION)
            ):
                raise PersistenceError("Unsupported checkpoint metadata")
            data = {name: archive.read(name) for name in ("knowledge.json", "tensors.pt")}
        if metadata["hashes"] != {
            name: hashlib.sha256(value).hexdigest() for name, value in data.items()
        }:
            raise PersistenceError("Checkpoint digest mismatch")
        knowledge = Athanor.loads(data["knowledge.json"].decode("utf-8"))
        config = ModelConfig(**metadata["model"])
        if config.width > 4096 or config.max_nodes > 100000:
            raise PersistenceError("Checkpoint model exceeds import budget")
        model = NeuralObserverModel(config)
        payload = torch.load(io.BytesIO(data["tensors.pt"]), map_location="cpu", weights_only=True)
        if set(payload) != {
            "parameters",
            "field",
            "context",
            "output",
            "optimizer",
            "random_state",
        }:
            raise PersistenceError("Unexpected tensor payload")
        expected = model.state_dict()
        if set(payload["parameters"]) != set(expected):
            raise PersistenceError("Checkpoint parameter names disagree with the architecture")
        for key, value in expected.items():
            _tensor(payload["parameters"][key], tuple(value.shape))
        model.load_state_dict(payload["parameters"], strict=True)
        model.eval()
        observer = Observer(knowledge, model, self_ref=_read_ref(metadata["self"], knowledge))
        if metadata["version"] == 3 and metadata["active_operations"] is not None:
            observer.active_operations = tuple(
                _read_ref(value, knowledge) for value in metadata["active_operations"]
            )
        refs = tuple(_read_ref(value, knowledge) for value in metadata["state_refs"])
        if (
            len(set(refs)) != len(refs)
            or len(refs) > config.max_nodes
            or type(metadata["steps"]) is not int
            or metadata["steps"] < 0
        ):
            raise PersistenceError("Invalid persistent activation handles/steps")
        observer.state = Activation(
            refs,
            _tensor(payload["field"], (len(refs), config.width)),
            _tensor(payload["context"], (config.width,)),
            metadata["steps"],
        )
        observer.self_map = [_read_ref(value, knowledge) for value in metadata["self_map"]]
        if len(observer.self_map) > config.history_limit:
            raise PersistenceError("Self-map exceeds history budget")
        history = metadata["activation_history"]
        if type(history) is not list or len(history) > config.history_limit:
            raise PersistenceError("Invalid activation history")
        for row in history:
            if (
                type(row) is not list
                or len(row) != min(8, config.width)
                or any(type(v) not in (float, int) or not -1 <= v <= 1 for v in row)
            ):
                raise PersistenceError("Malformed activation history")
        observer.activation_history = [tuple(row) for row in history]
        for ref_value, attempts, successes in metadata["capabilities"]:
            ref = _read_ref(ref_value, knowledge)
            if (
                type(attempts) is not int
                or type(successes) is not int
                or not 0 <= successes <= attempts
                or ref in observer.capabilities
            ):
                raise PersistenceError("Invalid capability measurements")
            observer.capabilities[ref] = (attempts, successes)
        observation = metadata["observation"]
        if observation is not None:
            if set(observation) != {"ref", "candidates"}:
                raise PersistenceError("Malformed checkpoint observation")
            observer.observation = Observation(
                _read_ref(observation["ref"], knowledge),
                tuple(_read_ref(value, knowledge) for value in observation["candidates"]),
            )
            frame = observer.builder.build(
                knowledge,
                observer.observation,
                self_refs=tuple(observer.self_map),
                observer=observer.self_ref,
                operations=observer.active_operations,
            )
            if frame.refs != refs:
                raise PersistenceError("Saved activations disagree with reconstructed graph")
            output = payload["output"]
            output_keys = {
                "operation_logits",
                "argument_logits",
                "applicability_logits",
                "success_logit",
                "preference_logit",
                "outcome",
                "operation_index",
                "argument_indices",
            }
            if metadata["version"] >= 2:
                output_keys.add("meaning_logits")
            if type(output) is not dict or set(output) != output_keys:
                raise PersistenceError("Malformed saved proposal")
            if metadata["version"] >= 2 and bool(frame.meanings) != (
                output["meaning_logits"] is not None
            ):
                raise PersistenceError("Saved meaning activations disagree with the graph")
            op_index = output["operation_index"]
            if type(op_index) is not int or not 0 <= op_index <= len(frame.operations):
                raise PersistenceError("Invalid saved operation pointer")
            arity = 0 if op_index == len(frame.operations) else frame.arities[op_index]
            indices = output["argument_indices"]
            if (
                type(indices) is not tuple
                or len(indices) != arity
                or len(output["argument_logits"]) != arity
                or any(
                    type(i) is not int or not 0 <= i < max(len(frame.candidates), 1)
                    for i in indices
                )
            ):
                raise PersistenceError("Invalid saved argument pointers")
            observer._frame = frame
            observer._output = ModelOutput(
                _tensor(output["operation_logits"], (len(frame.operations) + 1,)),
                tuple(
                    _tensor(value, (max(len(frame.candidates), 1),))
                    for value in output["argument_logits"]
                ),
                _tensor(output["applicability_logits"], (len(frame.operations),)),
                _tensor(output["success_logit"], ()),
                _tensor(output["preference_logit"], ()),
                _tensor(output["outcome"], (6,)),
                op_index,
                indices,
                observer.state,
                None
                if output.get("meaning_logits") is None
                else _tensor(output["meaning_logits"], (len(frame.meanings) + 1,)),
            )
            observer.propose()  # Check the boundary expectation contract, too.
        elif refs or payload["output"] is not None:
            raise PersistenceError("Activations/proposal lack their observation")
        training = metadata["training"]
        trainer = None
        if training is not None:
            if (
                set(training) != {"config", "updates"}
                or type(training["updates"]) is not int
                or training["updates"] < 0
            ):
                raise PersistenceError("Invalid saved training state")
            trainer = Trainer(model, TrainingConfig(**training["config"]))
            trainer.optimizer.load_state_dict(payload["optimizer"])
            for parameter, state in trainer.optimizer.state.items():
                if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                    raise PersistenceError("Unexpected optimizer state")
                _tensor(state["step"], ())
                if float(state["step"]) < 0:
                    raise PersistenceError("Invalid optimizer step")
                _tensor(state["exp_avg"], tuple(parameter.shape))
                _tensor(state["exp_avg_sq"], tuple(parameter.shape))
            trainer.rng.setstate(payload["random_state"])
            trainer.updates = training["updates"]
        elif payload["optimizer"] is not None or payload["random_state"] is not None:
            raise PersistenceError("Optimizer lacks its training configuration")
        return observer, trainer
    except PersistenceError:
        raise
    except (
        AthanorError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
        zipfile.BadZipFile,
        EOFError,
        AttributeError,
        pickle.UnpicklingError,
    ) as exc:
        raise PersistenceError(f"Invalid neural checkpoint: {exc}") from exc
