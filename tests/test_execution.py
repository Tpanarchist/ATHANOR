import unittest
from fractions import Fraction

from athanor import (
    Athanor,
    Endpoint,
    ExecutionError,
    FieldSpec,
    Limits,
    Procedure,
    ResourceLimitError,
    Role,
    ValidationError,
    record,
)
from athanor.programs import (
    argument,
    binary,
    call,
    choose,
    field,
    fold,
    has_field,
    item,
    literal,
    members,
    sequence,
    variable,
)


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.a = Athanor(namespace="execution")
        self.number = self.a.teach_category("Number", {"value": FieldSpec("number", minimum=0)})
        self.input = self.a.create(self.number, {"value": 2})

    def operation(self, expression, *, category=None, member_expression=None, update=None):
        return self.a.teach_operation(
            "procedure",
            Procedure(
                (Endpoint(Role.DATA, self.number),),
                self.number if category is None else category,
                {"value": expression},
                members=member_expression,
                update=update,
            ),
        )

    def rejected(self, action, error):
        before = self.a.dumps()
        with self.assertRaises(error):
            action()
        self.assertEqual(self.a.dumps(), before)

    def test_arithmetic_and_lazy_conditional(self):
        expression = choose(
            binary("le", field(argument(0), "value"), literal(2)),
            binary("sub", binary("mul", literal(3), literal(2)), literal(1)),
            binary("div", literal(1), literal(0)),
        )
        operation = self.operation(expression)
        self.assertEqual(self.a.get(self.a.execute(operation, (self.input,))).fields["value"], 5)

    def test_type_aware_equality_and_numeric_comparison(self):
        boolean = self.a.teach_category("Boolean", {"value": FieldSpec("bool")})
        for expr, expected in (
            (binary("eq", literal(True), literal(1)), False),
            (binary("eq", literal({"x": True}), literal({"x": 1})), False),
            (binary("eq", literal([1, 2]), literal([1, 2])), True),
            (binary("lt", literal(1), literal(2)), True),
        ):
            operation = self.operation(expr, category=boolean)
            self.assertEqual(
                self.a.get(self.a.execute(operation, (self.input,))).fields["value"], expected
            )

    def test_generic_fold_and_sequence_projection(self):
        operation = self.operation(
            fold(
                sequence(literal(Fraction(1, 3)), literal(Fraction(1, 6))),
                literal(0),
                binary("add", variable("acc"), variable("item")),
            )
        )
        self.assertEqual(
            self.a.get(self.a.execute(operation, (self.input,))).fields["value"], Fraction(1, 2)
        )
        operation = self.operation(item(sequence(literal(3), literal(4)), 1))
        self.assertEqual(self.a.get(self.a.execute(operation, (self.input,))).fields["value"], 4)

    def test_nested_fold_scope_and_record_projection(self):
        expression = fold(
            literal([[1, 2], [3, 4]]),
            literal(0),
            binary(
                "add",
                variable("acc"),
                fold(
                    variable("item"), literal(0), binary("add", variable("acc"), variable("item"))
                ),
            ),
        )
        operation = self.operation(expression)
        self.assertEqual(self.a.get(self.a.execute(operation, (self.input,))).fields["value"], 10)
        operation = self.operation(field(literal({"x": 7}), "x"))
        self.assertEqual(self.a.get(self.a.execute(operation, (self.input,))).fields["value"], 7)

    def test_taught_recursion_on_nested_composites(self):
        root = self.a.teach_category("Root")
        leaf_type = self.a.teach_category("Leaf", {"value": FieldSpec("number")}, parents=(root,))
        bundle_type = self.a.teach_category("Bundle", parents=(root,))
        expression = choose(
            has_field(argument(0), "value"),
            field(argument(0), "value"),
            fold(
                members(argument(0)),
                literal(0),
                binary("add", variable("acc"), field(call("self", variable("item")), "value")),
            ),
        )
        operation = self.a.teach_operation(
            "sum", Procedure((Endpoint(Role.DATA, root),), leaf_type, {"value": expression})
        )
        leaves = [
            self.a.create(leaf_type, {"value": value})
            for value in (Fraction(1, 7), Fraction(2, 7), Fraction(3, 7))
        ]
        inner = self.a.compose(leaves[:2], bundle_type)
        outer = self.a.compose((inner, leaves[2]), bundle_type)
        result = self.a.execute(operation, (outer,))
        self.assertEqual(self.a.get(result).fields["value"], Fraction(6, 7))
        self.assertEqual(len(self.a.inspect(result).receipt.causes), 2)
        computed = self.a.compose((result, leaves[0]), bundle_type)
        self.assertEqual(self.a.get(self.a.execute(operation, (computed,))).fields["value"], 1)

    def test_new_procedure_reuses_existing_operation(self):
        first = self.operation(binary("add", field(argument(0), "value"), literal(1)))
        second = self.operation(binary("mul", field(call(first, argument(0)), "value"), literal(2)))
        result = self.a.execute(second, (self.input,))
        self.assertEqual(self.a.get(result).fields["value"], 6)
        self.assertEqual(len(self.a.inspect(result).receipt.causes), 1)

    def test_operation_produces_composition_and_update_preserves_members(self):
        operation = self.operation(literal(1), member_expression=sequence(argument(0)))
        result = self.a.execute(operation, (self.input,))
        self.assertEqual(self.a.get(result).members, (self.input,))
        update = self.operation(literal(3), update=0)
        self.assertEqual(self.a.get(self.a.execute(update, (result,))).members, (self.input,))

    def test_invalid_grammar_and_variable_scope_rejected_at_teaching(self):
        for expression in (
            record(op="unknown"),
            record(op="literal", value=1, extra=2),
            argument(1),
            argument(-1),
            variable("item"),
            fold(literal([]), literal(0), variable("global")),
            call(self.number, argument(0)),
        ):
            self.rejected(lambda expression=expression: self.operation(expression), ValidationError)

    def test_runtime_field_index_and_operand_errors(self):
        for expression in (
            binary("div", literal(1), literal(0)),
            binary("mul", literal(True), literal(2)),
            choose(literal(1), literal(2), literal(3)),
            field(argument(0), "missing"),
            item(literal([]), 0),
        ):
            operation = self.operation(expression)
            self.rejected(
                lambda operation=operation: self.a.execute(operation, (self.input,)), ExecutionError
            )

    def test_call_arity_and_wrong_role(self):
        operation = self.operation(literal(1))
        for args in ((), (self.input, self.input), (self.number,)):
            self.rejected(lambda args=args: self.a.execute(operation, args), ValidationError)

    def test_outer_invalid_result_rolls_back_nested_creation(self):
        child = self.operation(literal(3))
        parent = self.operation(
            binary("mul", field(call(child, argument(0)), "value"), literal(-1))
        )
        self.rejected(lambda: self.a.execute(parent, (self.input,)), ValidationError)

    def test_outer_invalid_result_rolls_back_nested_update(self):
        child = self.operation(literal(3), update=0)
        parent = self.operation(
            binary("mul", field(call(child, argument(0)), "value"), literal(-1))
        )
        self.rejected(lambda: self.a.execute(parent, (self.input,)), ValidationError)
        self.assertEqual(self.a.get(self.input.id).ref, self.input)

    def test_nested_failure_rolls_back_prior_success(self):
        success = self.operation(literal(3))
        failure = self.operation(binary("div", literal(1), literal(0)))
        parent = self.operation(
            binary(
                "add",
                field(call(success, argument(0)), "value"),
                field(call(failure, argument(0)), "value"),
            )
        )
        self.rejected(lambda: self.a.execute(parent, (self.input,)), ExecutionError)

    def test_recursion_and_work_limits_are_atomic(self):
        recursive = self.operation(field(call("self", argument(0)), "value"))
        self.rejected(
            lambda: self.a.execute(recursive, (self.input,), limits=Limits(max_depth=5)),
            ResourceLimitError,
        )
        operation = self.operation(
            fold(
                literal(list(range(100))),
                literal(0),
                binary("add", variable("acc"), variable("item")),
            )
        )
        self.rejected(
            lambda: self.a.execute(operation, (self.input,), limits=Limits(max_steps=20)),
            ResourceLimitError,
        )

    def test_failure_does_not_consume_identity_or_receipt(self):
        success = self.operation(literal(3))
        fail = self.operation(
            binary("mul", field(call(success, argument(0)), "value"), literal(-1))
        )
        checkpoint = self.a.dumps()
        self.rejected(lambda: self.a.execute(fail, (self.input,)), ValidationError)
        restored = Athanor.loads(checkpoint)
        self.assertEqual(
            self.a.execute(success, (self.input,)), restored.execute(success, (self.input,))
        )
        self.assertEqual(self.a.dumps(), restored.dumps())

    def test_limit_arguments_are_positive_integers(self):
        for kwargs in ({"max_steps": 0}, {"max_depth": True}, {"max_steps": -1}):
            with self.assertRaises(ValidationError):
                Limits(**kwargs)

    def test_nullary_taught_computation(self):
        operation = self.a.teach_operation(
            "constant arithmetic",
            Procedure((), self.number, {"value": binary("div", literal(2), literal(3))}),
        )
        self.assertEqual(self.a.get(self.a.execute(operation)).fields["value"], Fraction(2, 3))


if __name__ == "__main__":
    unittest.main()
