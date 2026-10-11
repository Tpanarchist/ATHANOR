"""Offline teacher JSON -> independently checked native knowledge, never live inference."""

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import cast

from athanor import (
    Athanor,
    Endpoint,
    Evidence,
    FieldSpec,
    Procedure,
    Ref,
    Role,
    ValidationError,
    record,
)
from athanor.values import Value, as_record, as_ref, as_tuple, freeze

from .teaching import Example, Teacher


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError("Duplicate keys in teacher material")
        result[key] = value
    return result


def _object(
    value: object, required: set[str], optional: set[str] | None = None
) -> dict[str, object]:
    if type(value) is not dict:
        raise ValidationError("Teacher material requires an object")
    result = cast(dict[str, object], value)
    if not required <= set(result) or set(result) - required - (optional or set()):
        raise ValidationError("Missing or unknown fields in teacher material")
    return result


def _string(value: object) -> str:
    if type(value) is not str or not value.strip():
        raise ValidationError("Teacher material requires nonempty text")
    return value


def _list(value: object) -> list[object]:
    if type(value) is not list or len(value) > 64:
        raise ValidationError("Teacher material requires a list of at most 64 items")
    return cast(list[object], value)


def _native(value: object, names: dict[str, Ref], *, depth: int = 0) -> Value:
    if depth > 32:
        raise ValidationError("Teacher material exceeds its nesting budget")
    if type(value) is dict:
        raw = cast(dict[str, object], value)
        if set(raw) == {"ref"}:
            parts = _list(raw["ref"])
            if (
                len(parts) != 3
                or type(parts[0]) is not str
                or type(parts[1]) is not int
                or type(parts[2]) is not str
            ):
                raise ValidationError("Exact teacher references use [id, revision, namespace]")
            return Ref(parts[0], parts[1], parts[2])
        if set(raw) == {"named"}:
            name = _string(raw["named"])
            if name not in names:
                raise ValidationError("Unknown/forward local name in teacher material")
            return names[name]
        if set(raw) == {"fraction"}:
            parts = _list(raw["fraction"])
            if len(parts) != 2 or any(type(part) is not int for part in parts) or parts[1] == 0:
                raise ValidationError("Teacher fractions require two integers and nonzero divisor")
            return Fraction(cast(int, parts[0]), cast(int, parts[1]))
        return record({key: _native(item, names, depth=depth + 1) for key, item in raw.items()})
    if type(value) is list:
        return tuple(_native(item, names, depth=depth + 1) for item in cast(list[object], value))
    return freeze(value)


def import_lesson(
    knowledge: Athanor, text: str, *, names: dict[str, Ref] | None = None
) -> dict[str, Ref]:
    if len(text.encode("utf-8")) > 256 * 1024:
        raise ValidationError("Offline lesson exceeds its explicit import budget")
    try:
        parsed = json.loads(text, object_pairs_hook=_pairs)
    except (ValueError, RecursionError) as exc:
        raise ValidationError("Invalid offline lesson JSON") from exc
    package = _object(
        parsed, {"format", "source", "procedures"}, {"mappings", "specializations", "claims"}
    )
    if package["format"] != "athanor.lesson.v1":
        raise ValidationError("Unsupported offline lesson format")
    source = "offline teacher:" + _string(package["source"])
    registry = {} if names is None else dict(names)
    for ref in registry.values():
        knowledge.get(ref)

    def install(a: Athanor) -> dict[str, Ref]:
        teacher = Teacher(a)
        local = dict(registry)
        installed: dict[str, Ref] = {}
        for material in _list(package["procedures"]):
            entry = _object(
                material,
                {"name", "definition", "procedure", "expressions", "examples"},
                {"roles", "context"},
            )
            name = _string(entry["name"])
            if name in local or name + ".operation" in local:
                raise ValidationError("Offline names cannot silently replace existing knowledge")
            spec = _object(
                entry["procedure"], {"inputs", "result", "fields"}, {"members", "update"}
            )
            interfaces = []
            for item in _list(spec["inputs"]):
                endpoint = _object(item, {"role", "category"})
                raw_category = _native(endpoint["category"], local)
                interfaces.append(
                    Endpoint(
                        None if endpoint["role"] is None else Role(_string(endpoint["role"])),
                        None if raw_category is None else as_ref(raw_category),
                    )
                )
            result_category = _native(spec["result"], local)
            body = as_record(_native(spec["fields"], local))
            members = _native(spec.get("members"), local)
            update = spec.get("update")
            if update is not None and (type(update) is not int or update < 0):
                raise ValidationError("Invalid offline update index")
            procedure = Procedure(
                tuple(interfaces),
                None if result_category is None else as_ref(result_category),
                {key: as_record(value) for key, value in body.items()},
                None if members is None else as_record(members),
                update,
            )
            examples = []
            for example in _list(entry["examples"]):
                row = _object(example, {"arguments", "expected"})
                examples.append(
                    Example(
                        tuple(
                            as_ref(value) for value in as_tuple(_native(row["arguments"], local))
                        ),
                        as_record(_native(row["expected"], local)),
                    )
                )
            if not examples:
                raise ValidationError(
                    "Offline procedures require independently checked exact examples"
                )
            concept = teacher.define(
                name,
                _string(entry["definition"]),
                procedure,
                expressions=tuple(_string(value) for value in _list(entry["expressions"])),
                examples=tuple(examples),
                roles=tuple(_string(value) for value in _list(entry.get("roles", []))),
                context=as_record(_native(entry.get("context", {}), local)),
                source=source,
            )
            local[name] = installed[name] = concept
            local[name + ".operation"] = installed[name + ".operation"] = as_ref(
                a.get(concept).fields["operation"]
            )
        for material in _list(package.get("specializations", [])):
            entry = _object(material, {"name", "parent", "fields", "expressions"})
            name = _string(entry["name"])
            if name in local:
                raise ValidationError("Duplicate offline name")
            fields = {}
            raw_fields = as_record(_native(entry["fields"], local))
            for key, raw in raw_fields.items():
                constraint = as_record(raw)
                if (
                    set(constraint) != {"kind", "required", "minimum", "maximum"}
                    or type(constraint["required"]) is not bool
                ):
                    raise ValidationError("Malformed offline field contract")
                low, high = constraint["minimum"], constraint["maximum"]
                if (
                    low is not None
                    and not isinstance(low, Fraction)
                    or high is not None
                    and not isinstance(high, Fraction)
                ):
                    raise ValidationError("Offline bounds must be exact numbers")
                fields[key] = FieldSpec(
                    _string(constraint["kind"]),
                    constraint["required"],
                    low,
                    high,
                )
            category = teacher.specialize(
                name,
                as_ref(_native(entry["parent"], local)),
                fields,
                expressions=tuple(_string(value) for value in _list(entry["expressions"])),
                source=source,
            )
            local[name] = installed[name] = category
        for material in _list(package.get("mappings", [])):
            entry = _object(material, {"expression", "referent"}, {"context", "stance"})
            teacher.alias(
                _string(entry["expression"]),
                as_ref(_native(entry["referent"], local)),
                context=as_record(_native(entry.get("context", {}), local)),
                evidence=Evidence(source, _string(entry.get("stance", "positive"))),
            )
        for material in _list(package.get("claims", [])):
            entry = _object(material, {"relation", "subject", "target"}, {"stance"})
            teacher.claim(
                as_ref(_native(entry["relation"], local)),
                as_ref(_native(entry["subject"], local)),
                as_ref(_native(entry["target"], local)),
                source=source,
                stance=_string(entry.get("stance", "positive")),
            )
        a.create(
            fields={
                "kind": "offline_lesson_receipt",
                "source": source,
                "installed": tuple(installed.values()),
                "exact_examples_checked": True,
                "empirical_claims_verified_by_teacher": False,
            },
            source=source,
        )
        return installed

    try:
        return knowledge.transaction(install)
    except ValueError as exc:
        raise ValidationError("Invalid enum or value in offline teacher material") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lesson", type=Path)
    parser.add_argument("--knowledge", required=True, type=Path)
    parser.add_argument("--save", required=True, type=Path)
    args = parser.parse_args()
    knowledge = Athanor.restore(args.knowledge)
    installed = import_lesson(knowledge, args.lesson.read_text(encoding="utf-8"))
    args.save.parent.mkdir(parents=True, exist_ok=True)
    knowledge.save(args.save)
    print(json.dumps({name: str(ref) for name, ref in installed.items()}, indent=2))


if __name__ == "__main__":
    main()
