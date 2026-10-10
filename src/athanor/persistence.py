"""Strict portable snapshots and atomic replacement, with no executable deserialization."""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import cast

from .contracts import Receipt, Role, Structure
from .errors import AthanorError, PersistenceError
from .store import Store
from .values import Record, Ref, Value, as_index, as_record, as_ref, as_tuple

FORMAT = "athanor.goal001"
VERSION = 1


def _encode(value: Value) -> object:
    from fractions import Fraction

    if value is None:
        return ["null"]
    if type(value) is bool:
        return ["bool", value]
    if isinstance(value, str):
        return ["string", value]
    if isinstance(value, Fraction):
        return ["rational", str(value.numerator), str(value.denominator)]
    if isinstance(value, Ref):
        return ["ref", value.namespace, value.id, value.revision]
    if isinstance(value, tuple):
        return ["sequence", [_encode(item) for item in value]]
    if isinstance(value, Record):
        return ["record", [[key, _encode(item)] for key, item in value.items()]]
    raise PersistenceError("Unsupported internal value")


def _decode(raw: object) -> Value:
    from fractions import Fraction

    if not isinstance(raw, list) or not raw or type(raw[0]) is not str:
        raise PersistenceError("Malformed tagged value")
    tag = raw[0]
    if tag == "null" and len(raw) == 1:
        return None
    if tag == "bool" and len(raw) == 2 and type(raw[1]) is bool:
        return raw[1]
    if tag == "string" and len(raw) == 2 and type(raw[1]) is str:
        return raw[1]
    if tag == "rational" and len(raw) == 3:
        numerator, denominator = raw[1], raw[2]
        if (
            type(numerator) is str
            and type(denominator) is str
            and re.fullmatch(r"-?(0|[1-9][0-9]*)", numerator)
            and re.fullmatch(r"[1-9][0-9]*", denominator)
        ):
            value = Fraction(int(numerator), int(denominator))
            if str(value.numerator) == numerator and str(value.denominator) == denominator:
                return value
    if (
        tag == "ref"
        and len(raw) == 4
        and type(raw[1]) is str
        and type(raw[2]) is str
        and type(raw[3]) is int
    ):
        return Ref(raw[2], raw[3], raw[1])
    if tag == "sequence" and len(raw) == 2 and isinstance(raw[1], list):
        return tuple(_decode(item) for item in raw[1])
    if tag == "record" and len(raw) == 2 and isinstance(raw[1], list):
        entries = raw[1]
        if all(
            isinstance(item, list) and len(item) == 2 and type(item[0]) is str for item in entries
        ):
            return Record(tuple((item[0], _decode(item[1])) for item in entries))
    raise PersistenceError(f"Malformed or noncanonical tagged value: {tag}")


def _canonical(raw: object) -> str:
    return json.dumps(
        raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _structure(node: Structure) -> dict[str, object]:
    return {
        "ref": _encode(node.ref),
        "role": node.role.value,
        "category": _encode(node.category),
        "fields": _encode(node.fields),
        "members": _encode(node.members),
    }


def _receipt(receipt: Receipt) -> dict[str, object]:
    return {
        "sequence": receipt.sequence,
        "action": receipt.action,
        "result": _encode(receipt.result),
        "inputs": _encode(receipt.inputs),
        "operation": _encode(receipt.operation),
        "causes": list(receipt.causes),
        "details": _encode(receipt.details),
    }


def dumps(store: Store) -> str:
    try:
        payload: dict[str, object] = {
            "format": FORMAT,
            "version": VERSION,
            "namespace": store.namespace,
            "entries": [
                {"structure": _structure(store.get(receipt.result)), "receipt": _receipt(receipt)}
                for receipt in store.receipts
            ],
        }
        digest = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
        return (
            json.dumps(
                {"payload": payload, "sha256": digest},
                sort_keys=True,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
    except (ValueError, TypeError, RecursionError, UnicodeError) as exc:
        raise PersistenceError(f"Could not encode snapshot: {exc}") from exc


def _object(raw: object, keys: set[str]) -> dict[str, object]:
    if not isinstance(raw, dict) or set(raw) != keys:
        raise PersistenceError(f"Snapshot object must contain exactly {sorted(keys)}")
    return cast(dict[str, object], raw)


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PersistenceError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(raw: str) -> object:
    raise PersistenceError(f"Nonfinite JSON constant: {raw}")


def _entry(store: Store, raw: object) -> None:
    entry = _object(raw, {"structure", "receipt"})
    node = _object(entry["structure"], {"ref", "role", "category", "fields", "members"})
    receipt = _object(
        entry["receipt"],
        {"sequence", "action", "result", "inputs", "operation", "causes", "details"},
    )
    ref = as_ref(_decode(node["ref"]))
    if type(node["role"]) is not str:
        raise PersistenceError("Role must be a string")
    role = Role(node["role"])
    category_value = _decode(node["category"])
    category = None if category_value is None else as_ref(category_value)
    fields = as_record(_decode(node["fields"]))
    members = tuple(as_ref(item) for item in as_tuple(_decode(node["members"])))
    if type(receipt["sequence"]) is not int or receipt["sequence"] != len(store.receipts) + 1:
        raise PersistenceError("Receipt sequence is not contiguous")
    if as_ref(_decode(receipt["result"])) != ref:
        raise PersistenceError("Receipt does not produce its structure")
    inputs = tuple(as_ref(item) for item in as_tuple(_decode(receipt["inputs"])))
    raw_operation = _decode(receipt["operation"])
    operation = None if raw_operation is None else as_ref(raw_operation)
    raw_causes = receipt["causes"]
    if not isinstance(raw_causes, list) or any(type(item) is not int for item in raw_causes):
        raise PersistenceError("Receipt causes must be integer sequences")
    causes = tuple(cast(list[int], raw_causes))
    if len(set(causes)) != len(causes):
        raise PersistenceError("Duplicate causal receipt references")
    details = as_record(_decode(receipt["details"]))
    action = receipt["action"]
    declared_roles = {
        "create": Role.DATA,
        "compose": Role.DATA,
        "teach_category": Role.CATEGORY,
        "teach_relation": Role.RELATION,
        "teach_operation": Role.OPERATION,
        "bind": Role.CLAIM,
        "execute": Role.DATA,
        "transform": Role.DATA,
    }
    if type(action) is not str or action not in declared_roles or declared_roles[action] != role:
        raise PersistenceError("Invalid production action/role pair")
    update: Ref | None = None
    if action in {"execute", "transform"}:
        if operation is None:
            raise PersistenceError("Execution receipt has no operation")
        definition = store.require_role(operation, Role.OPERATION)
        spec = as_record(definition.fields["procedure"])
        interface = as_tuple(spec["inputs"])
        if len(inputs) != len(interface) or any(
            not store.matches(arg, as_record(endpoint))
            for arg, endpoint in zip(inputs, interface, strict=True)
        ):
            raise PersistenceError("Persisted execution inputs violate their interface")
        update_index = spec["update"]
        if update_index is not None:
            update = inputs[as_index(update_index)]
            old = store.get(update)
            declared = spec["result"]
            if category != old.category or (
                declared is not None
                and (category is None or as_ref(declared) not in store.ancestors(category))
            ):
                raise PersistenceError("Persisted update violates its preserved/result category")
        elif category != spec["result"]:
            raise PersistenceError("Persisted execution result has the wrong category")
        if (action == "transform") != (update is not None):
            raise PersistenceError("Production action disagrees with procedure update mode")
        if set(details) != {"expression_steps"} or as_index(details["expression_steps"]) < 1:
            raise PersistenceError("Malformed execution details")
    else:
        if operation is not None or causes:
            raise PersistenceError("Nonexecution receipt declares invocation dependencies")
        if action == "bind":
            if details or inputs != tuple(
                as_ref(fields[key]) for key in ("relation", "source", "target")
            ):
                raise PersistenceError("Binding receipt disagrees with its claim")
        else:
            if (
                set(details) != {"source"}
                or not isinstance(details["source"], str)
                or not details["source"]
            ):
                raise PersistenceError("Production requires attributed source details")
            expected_inputs: tuple[Ref, ...] = ()
            if action in {"create", "compose"}:
                expected_inputs = members
            elif action == "teach_category":
                expected_inputs = tuple(as_ref(item) for item in as_tuple(fields["parents"]))
            if inputs != expected_inputs:
                raise PersistenceError("Production inputs disagree with their structure")
    produced = store.publish(
        role,
        category,
        fields,
        members,
        action,
        update=update,
        inputs=inputs,
        operation=operation,
        causes=causes,
        details=details,
    )
    if produced != ref:
        raise PersistenceError("Noncontiguous identity/revision history")


def loads(snapshot: str) -> Store:
    try:
        envelope = _object(
            json.loads(snapshot, object_pairs_hook=_unique, parse_constant=_reject_constant),
            {"payload", "sha256"},
        )
        payload = _object(envelope["payload"], {"format", "version", "namespace", "entries"})
        if (
            payload["format"] != FORMAT
            or type(payload["version"]) is not int
            or payload["version"] != VERSION
        ):
            raise PersistenceError("Unsupported snapshot format/version")
        expected = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
        if type(envelope["sha256"]) is not str or envelope["sha256"] != expected:
            raise PersistenceError("Snapshot digest mismatch")
        if not isinstance(payload["entries"], list):
            raise PersistenceError("Snapshot entries must be a list")
        if type(payload["namespace"]) is not str or not payload["namespace"]:
            raise PersistenceError("Snapshot namespace must be a nonempty string")
        store = Store(namespace=payload["namespace"])
        for entry in payload["entries"]:
            _entry(store, entry)
        return store
    except PersistenceError:
        raise
    except (AthanorError, ValueError, TypeError, KeyError, RecursionError, UnicodeError) as exc:
        raise PersistenceError(f"Invalid snapshot: {exc}") from exc


def save(path: Path, snapshot: str) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(snapshot)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    except (OSError, UnicodeError) as exc:
        raise PersistenceError(f"Could not save snapshot: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise PersistenceError(f"Could not read snapshot: {exc}") from exc
