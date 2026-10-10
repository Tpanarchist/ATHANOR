"""Create → Teach → Bind → Transform → Compose → Execute → Inspect → Save → Restore → Reuse."""

import argparse
import json
from fractions import Fraction
from pathlib import Path

from . import (
    Assessment,
    Athanor,
    Endpoint,
    Evidence,
    FieldSpec,
    Procedure,
    Role,
    ValidationError,
)
from .programs import (
    argument,
    binary,
    call,
    choose,
    field,
    fold,
    has_field,
    literal,
    members,
    variable,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"Demonstration contract failed: {message}")


def run_demo(save_path: Path) -> dict[str, object]:
    a = Athanor(namespace="goal001-demo")
    # CREATE: a structure can exist before domain knowledge is installed.
    note = a.create(fields={"label": "An observation awaiting interpretation"})

    # TEACH: install all domain semantics through the ordinary teaching interface.
    measurable = a.teach_category("Measurable")
    quantity = a.teach_category(
        "Quantity", {"value": FieldSpec("number", minimum=0)}, parents=(measurable,)
    )
    labeled = a.teach_category(
        "LabeledQuantity", {"label": FieldSpec("string")}, parents=(quantity,)
    )
    bundle = a.teach_category("Bundle", parents=(measurable,))
    total = a.teach_operation(
        "recursive total",
        Procedure(
            inputs=(Endpoint(Role.DATA, measurable),),
            result=quantity,
            fields={
                "value": choose(
                    has_field(argument(0), "value"),
                    field(argument(0), "value"),
                    fold(
                        members(argument(0)),
                        literal(0),
                        binary(
                            "add",
                            variable("acc"),
                            field(call("self", variable("item")), "value"),
                        ),
                    ),
                )
            },
        ),
    )
    scale = a.teach_operation(
        "scale in place",
        Procedure(
            inputs=(Endpoint(Role.DATA, quantity), Endpoint(Role.DATA, quantity)),
            result=quantity,
            fields={
                "value": binary("mul", field(argument(0), "value"), field(argument(1), "value"))
            },
            update=0,
        ),
    )
    applicable = a.teach_relation(
        "declared applicable procedure",
        Endpoint(Role.CATEGORY, measurable),
        Endpoint(Role.OPERATION),
        inherit_source=True,
    )

    # BIND: attributed category knowledge is reused without copying claims.
    claim = a.bind(applicable, measurable, total, Evidence("structured instructor"))
    leaf = a.create(labeled, {"label": "sample A", "value": Fraction(3, 2)})
    second = a.create(quantity, {"value": 2})
    half = a.create(quantity, {"value": Fraction(1, 2)})
    factor = a.create(quantity, {"value": 2})
    resolved = a.relations(leaf, applicable)
    _require(len(resolved) == 1 and resolved[0].claim == claim, "inherited relation lookup")
    _require(a.assess(leaf, applicable, total) == Assessment.SUPPORTED, "installed support")
    _require(a.assess(leaf, applicable, scale) == Assessment.UNKNOWN, "absence remains unknown")

    # TRANSFORM: general operation works on a specialization, preserving its category.
    changed = a.execute(scale, (leaf, factor))
    _require(changed.id == leaf.id and changed.revision == 2, "persistent update identity")
    _require(a.get(changed).category == labeled, "specialization survives update")
    _require(a.get(leaf).fields["value"] == Fraction(3, 2), "old state remains inspectable")

    # COMPOSE: a resulting composite participates in another composition.
    inner = a.compose((changed, second), bundle)
    outer = a.compose((inner, half), bundle)

    # EXECUTE: recursion comes from the installed procedure, not a demo-specific engine.
    result = a.execute(total, (outer,))
    _require(a.get(result).fields["value"] == Fraction(11, 2), "exact recursive total")
    # The result of execution is also a valid participant in further composition.
    composed_result = a.compose((result, half), bundle)
    further = a.execute(total, (composed_result,))
    _require(a.get(further).fields["value"] == 6, "computed result can be reused")

    # INSPECT: definitions, inputs and nested computation receipts remain explicit.
    inspection = a.inspect(result)
    history = a.provenance(result)
    _require(inspection.receipt.operation == total, "operation provenance")
    _require(bool(inspection.receipt.causes), "nested computation provenance")
    _require(any(receipt.result == inner for receipt in history), "composition provenance")

    # Neither an incompatible input nor an invalid result may change knowledge.
    before = a.dumps()
    try:
        a.execute(total, (note,))
    except ValidationError:
        pass
    else:
        raise RuntimeError("Incompatible input was accepted")
    _require(a.dumps() == before, "invalid input leaves the entire store unchanged")
    negate = a.teach_operation(
        "negate (invalid for positive inputs under this schema)",
        Procedure(
            inputs=(Endpoint(Role.DATA, quantity),),
            result=quantity,
            fields={"value": binary("mul", field(argument(0), "value"), literal(-1))},
            update=0,
        ),
    )
    before = a.dumps()
    try:
        a.execute(negate, (changed,))
    except ValidationError:
        pass
    else:
        raise RuntimeError("Constraint-violating result was accepted")
    _require(a.dumps() == before, "invalid result leaves the entire store unchanged")
    a.bind(applicable, measurable, total, Evidence("dissenting instructor", "negative"))
    _require(
        a.assess(leaf, applicable, total) == Assessment.CONFLICT, "disagreement remains explicit"
    )

    # SAVE and RESTORE: no callbacks or external intelligence need to be reinstalled.
    save_path.parent.mkdir(parents=True, exist_ok=True)
    a.save(save_path)
    restored = Athanor.restore(save_path)
    _require(restored.dumps() == a.dumps(), "canonical restore equality")
    _require(restored.assess(leaf, applicable, total) == Assessment.CONFLICT, "restored evidence")

    # REUSE: restored knowledge runs, and another procedure can be taught afterwards.
    reused = restored.execute(total, (outer,))
    _require(restored.get(reused).fields["value"] == Fraction(11, 2), "restored execution")
    twice_total = restored.teach_operation(
        "twice recursive total",
        Procedure(
            inputs=(Endpoint(Role.DATA, measurable),),
            result=quantity,
            fields={"value": binary("mul", field(call(total, argument(0)), "value"), literal(2))},
        ),
    )
    twice = restored.execute(twice_total, (outer,))
    _require(restored.get(twice).fields["value"] == 11, "new procedure reuses restored procedure")
    reuse_path = save_path.with_name(f"{save_path.stem}-reused{save_path.suffix}")
    restored.save(reuse_path)
    second_restore = Athanor.restore(reuse_path)
    again = second_restore.execute(twice_total, (outer,))
    _require(
        second_restore.get(again).fields["value"] == 11, "new teaching survives another restore"
    )

    return {
        "sequence": (
            "Create -> Teach -> Bind -> Transform -> Compose -> Execute -> Inspect -> "
            "Save -> Restore -> Reuse"
        ),
        "namespace": a.namespace,
        "recursive_total": str(a.get(result).fields["value"]),
        "composed_execution_result_total": str(a.get(further).fields["value"]),
        "restored_total": str(restored.get(reused).fields["value"]),
        "new_taught_procedure_result": str(restored.get(twice).fields["value"]),
        "second_restore_result": str(second_restore.get(again).fields["value"]),
        "updated_identity": changed.id,
        "old_revision": leaf.revision,
        "new_revision": changed.revision,
        "inherited_claim_count": len(resolved),
        "inheritance_path": [str(ref) for ref in resolved[0].via],
        "unasserted_applicability": a.assess(leaf, applicable, scale).value,
        "disputed_applicability": a.assess(leaf, applicable, total).value,
        "invalid_input_and_result_rejected_atomically": True,
        "provenance_receipts_for_total": len(history),
        "saved_revisions": len(a.structures),
        "snapshot": str(save_path.resolve()),
        "snapshot_with_new_teaching": str(reuse_path.resolve()),
        "not_established": [
            "external truth of instructor claims",
            "grounded physical meaning of quantities",
            "learned activation or efficient learning",
            "autonomous self-improvement",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=Path, default=Path("artifacts/cornerstone.json"))
    args = parser.parse_args()
    print(json.dumps(run_demo(args.save), indent=2))


if __name__ == "__main__":
    main()
