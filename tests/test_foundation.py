import unittest
from dataclasses import FrozenInstanceError
from fractions import Fraction

from athanor import (
    Assessment,
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
from athanor.programs import argument, binary, field, literal


class FoundationTests(unittest.TestCase):
    def setUp(self):
        self.a = Athanor(namespace="test")
        self.number = self.a.teach_category("Number", {"value": FieldSpec("number", minimum=0)})

    def rejected(self, action):
        before = self.a.dumps()
        with self.assertRaises(ValidationError):
            action()
        self.assertEqual(self.a.dumps(), before)

    def operation(self, expression, update=None):
        return self.a.teach_operation(
            "numeric operation",
            Procedure(
                (Endpoint(Role.DATA, self.number),),
                self.number,
                {"value": expression},
                update=update,
            ),
        )

    def test_identity_and_state_are_separate(self):
        left = self.a.create(self.number, {"value": 1})
        right = self.a.create(self.number, {"value": Fraction(2, 2)})
        difference = self.a.distinguish(left, right)
        self.assertFalse(difference.same_identity)
        self.assertTrue(difference.same_state)

    def test_boolean_is_not_numeric_one(self):
        left = self.a.create(fields={"value": True})
        right = self.a.create(fields={"value": 1})
        self.assertEqual(self.a.distinguish(left, right).changed_fields, ("value",))
        self.rejected(lambda: self.a.create(self.number, {"value": True}))

    def test_values_copied_and_immutable(self):
        raw = {"nested": {"items": [1, 2]}}
        ref = self.a.create(fields=raw)
        raw["nested"]["items"].append(3)
        self.assertEqual(self.a.get(ref).fields, record(nested={"items": [1, 2]}))
        with self.assertRaises(FrozenInstanceError):
            self.a.get(ref).category = self.number
        with self.assertRaises(TypeError):
            self.a.get(ref).fields["extra"] = 1

    def test_unsupported_and_cyclic_values(self):
        cycle = []
        cycle.append(cycle)
        for value in (0.1, object(), {1: "bad key"}, cycle):
            with self.subTest(value_type=type(value)):
                self.rejected(lambda value=value: self.a.create(fields={"x": value}))

    def test_closed_schema_required_fields_and_bounds(self):
        for fields in ({}, {"value": -1}, {"value": "1"}, {"value": 1, "extra": 2}):
            with self.subTest(fields=fields):
                self.rejected(lambda fields=fields: self.a.create(self.number, fields))
        self.assertEqual(self.a.get(self.a.create(self.number, {"value": 0})).fields["value"], 0)

    def test_optional_field_and_other_value_kinds(self):
        fields = {
            "text": FieldSpec("string", required=False),
            "flag": FieldSpec("bool"),
            "ref": FieldSpec("ref"),
            "list": FieldSpec("sequence"),
            "map": FieldSpec("record"),
            "any": FieldSpec(),
        }
        category = self.a.teach_category("Mixed", fields)
        ref = self.a.create(
            category, {"flag": False, "ref": self.number, "list": [], "map": {}, "any": None}
        )
        self.assertNotIn("text", self.a.get(ref).fields)

    def test_invalid_field_specs(self):
        for spec in (
            FieldSpec("missing"),
            FieldSpec("string", minimum=1),
            FieldSpec("number", minimum=2, maximum=1),
        ):
            with self.subTest(spec=spec):
                self.rejected(lambda spec=spec: self.a.teach_category("bad", {"x": spec}))

    def test_multiple_inheritance_resolves_all_ancestors(self):
        left = self.a.teach_category("Left", {"x": FieldSpec("string")}, parents=(self.number,))
        right = self.a.teach_category("Right", {"y": FieldSpec("bool")}, parents=(self.number,))
        child = self.a.teach_category("Child", parents=(left, right))
        self.assertEqual(set(self.a.schema(child)), {"value", "x", "y"})
        self.assertEqual(self.a.ancestors(child), (child, left, self.number, right))
        conflict = self.a.teach_category("Other", {"value": FieldSpec("string")})
        self.rejected(lambda: self.a.teach_category("Bad", parents=(child, conflict)))

    def test_duplicate_parent_and_override_rejected(self):
        self.rejected(lambda: self.a.teach_category("Bad", parents=(self.number, self.number)))
        self.rejected(
            lambda: self.a.teach_category(
                "Bad", {"value": FieldSpec("number", minimum=1)}, parents=(self.number,)
            )
        )

    def test_exact_rational_execution(self):
        operation = self.operation(binary("div", field(argument(0), "value"), literal(3)))
        ref = self.a.create(self.number, {"value": Fraction(1, 10)})
        self.assertEqual(
            self.a.get(self.a.execute(operation, (ref,))).fields["value"], Fraction(1, 30)
        )

    def test_polymorphic_update_preserves_category_and_history(self):
        subtype = self.a.teach_category(
            "Labeled", {"label": FieldSpec("string")}, parents=(self.number,)
        )
        ref = self.a.create(subtype, {"value": 2, "label": "kept"})
        operation = self.operation(binary("add", field(argument(0), "value"), literal(1)), update=0)
        result = self.a.execute(operation, (ref,))
        self.assertEqual((result.id, result.revision), (ref.id, 2))
        self.assertEqual(self.a.get(result).category, subtype)
        self.assertEqual(self.a.get(result).fields, record(value=3, label="kept"))
        self.assertEqual(self.a.get(ref).fields["value"], 2)
        self.assertEqual(self.a.get(ref.id).ref, result)
        self.rejected(lambda: self.a.execute(operation, (ref,)))

    def test_update_validates_specialized_fields(self):
        subtype = self.a.teach_category(
            "Labeled", {"label": FieldSpec("string")}, parents=(self.number,)
        )
        ref = self.a.create(subtype, {"value": 2, "label": "kept"})
        operation = self.a.teach_operation(
            "invalid patch",
            Procedure(
                (Endpoint(Role.DATA, self.number),), self.number, {"label": literal(1)}, update=0
            ),
        )
        self.rejected(lambda: self.a.execute(operation, (ref,)))

    def test_composite_members_do_not_drift(self):
        leaf = self.a.create(self.number, {"value": 2})
        pair = self.a.compose((leaf, leaf))
        outer = self.a.compose((pair, self.number))
        self.assertEqual(self.a.get(outer).members, (pair, self.number))
        operation = self.operation(literal(3), update=0)
        changed = self.a.execute(operation, (leaf,))
        self.assertEqual(self.a.get(pair).members, (leaf, leaf))
        self.assertNotEqual(self.a.get(pair).members[0], changed)

    def test_definitions_are_reflectable_and_protected(self):
        operation = self.operation(literal(1))
        reflection = self.a.teach_operation(
            "name",
            Procedure((Endpoint(Role.OPERATION),), None, {"name": field(argument(0), "name")}),
        )
        self.assertEqual(
            self.a.get(self.a.execute(reflection, (operation,))).fields["name"], "numeric operation"
        )
        modify = self.a.teach_operation("rewrite", Procedure((Endpoint(),), None, {}, update=0))
        self.rejected(lambda: self.a.execute(modify, (operation,)))

    def test_foreign_namespace_cannot_alias(self):
        foreign = Athanor(namespace="other").create()
        self.assertEqual(foreign.id, self.number.id)
        self.rejected(lambda: self.a.create(fields={"foreign": foreign}))
        self.rejected(lambda: self.a.compose((foreign,)))
        with self.assertRaises(ValidationError):
            self.a.get(foreign)

    def test_default_stores_are_independent(self):
        first, second = Athanor(), Athanor()
        self.assertNotEqual(first.namespace, second.namespace)
        self.assertNotEqual(first.create(), second.create())

    def test_forward_and_missing_references(self):
        missing = Ref("s99999999", namespace=self.a.namespace)
        self.rejected(lambda: self.a.create(fields={"nested": [{"ref": missing}]}))
        self.rejected(lambda: self.a.teach_category("unknown", parents=(missing,)))

    def test_inheritance_path_is_a_path_and_claims_are_not_copied(self):
        sibling = self.a.teach_category("Sibling")
        child = self.a.teach_category("Child", parents=(sibling, self.number))
        relation = self.a.teach_relation(
            "affords", Endpoint(Role.CATEGORY), Endpoint(Role.OPERATION), inherit_source=True
        )
        operation = self.operation(literal(1))
        claim = self.a.bind(relation, self.number, operation, Evidence("teacher"))
        ref = self.a.create(child, {"value": 2})
        resolved = self.a.relations(ref, relation)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].claim, claim)
        self.assertEqual(resolved[0].via, (child, self.number))
        self.assertEqual(len([node for node in self.a.structures if node.role == Role.CLAIM]), 1)

    def test_evidence_assessment_preserves_unknown_and_conflict(self):
        relation = self.a.teach_relation("related")
        left, right = self.a.create(), self.a.create()
        self.assertEqual(self.a.assess(left, relation, right), Assessment.UNKNOWN)
        self.a.bind(relation, left, right, Evidence("observer", "uncertain", Fraction(1, 2)))
        self.assertEqual(self.a.assess(left, relation, right), Assessment.UNKNOWN)
        self.a.bind(relation, left, right, Evidence("skeptic", "negative", Fraction(1, 10)))
        self.assertEqual(self.a.assess(left, relation, right), Assessment.REFUTED)
        self.a.bind(relation, left, right, Evidence("teacher", "positive", 1))
        self.assertEqual(self.a.assess(left, relation, right), Assessment.CONFLICT)

    def test_relationship_interfaces_and_evidence_are_validated(self):
        relation = self.a.teach_relation(
            "typed", Endpoint(Role.DATA, self.number), Endpoint(Role.CATEGORY)
        )
        ref = self.a.create(self.number, {"value": 1})
        self.a.bind(relation, ref, self.number, Evidence("teacher"))
        self.rejected(lambda: self.a.bind(relation, self.number, ref, Evidence("teacher")))
        self.rejected(
            lambda: self.a.teach_relation("bad", Endpoint(Role.DATA), inherit_source=True)
        )
        self.rejected(lambda: self.a.bind(relation, ref, self.number, Evidence("")))
        self.rejected(
            lambda: self.a.bind(relation, ref, self.number, Evidence("teacher", confidence=2))
        )

    def test_direct_claims_apply_to_exact_revisions(self):
        relation = self.a.teach_relation("direct")
        ref = self.a.create(self.number, {"value": 1})
        self.a.bind(relation, ref, self.number, Evidence("teacher"))
        operation = self.operation(literal(2), update=0)
        changed = self.a.execute(operation, (ref,))
        self.assertEqual(self.a.assess(ref, relation, self.number), Assessment.SUPPORTED)
        self.assertEqual(self.a.assess(changed, relation, self.number), Assessment.UNKNOWN)

    def test_provenance_includes_definitions_and_inputs(self):
        operation = self.operation(binary("add", field(argument(0), "value"), literal(1)), update=0)
        ref = self.a.create(self.number, {"value": 2})
        result = self.a.execute(operation, (ref,))
        inspection = self.a.inspect(result)
        self.assertEqual(inspection.receipt.operation, operation)
        self.assertEqual(inspection.receipt.inputs, (ref,))
        history = self.a.provenance(result)
        self.assertEqual(
            {receipt.result for receipt in history}, {self.number, operation, ref, result}
        )
        self.assertEqual(
            [receipt.sequence for receipt in history],
            sorted(receipt.sequence for receipt in history),
        )

    def test_bad_reference_identifiers(self):
        for args in (("bad", 1), ("s00000000", 1), ("s00000001", True), ("s00000001", 0)):
            with self.assertRaises(ValidationError):
                Ref(*args)

    def test_unserializable_unicode_is_rejected_before_publication(self):
        self.rejected(lambda: self.a.create(fields={"x": "\ud800"}))
        self.rejected(lambda: self.a.create(fields={"\ud800": "value"}))
        with self.assertRaises(ValidationError):
            Athanor(namespace="\ud800")

    def test_excessive_immutable_value_nesting_is_rejected(self):
        def nested():
            value = record()
            for _ in range(100):
                value = record(child=value)
            self.a.create(fields={"value": value})

        self.rejected(nested)


if __name__ == "__main__":
    unittest.main()
