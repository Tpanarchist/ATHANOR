"""Structural execution, evidence, neural participation and integrated preservation."""

import contextlib
import copy
import importlib.util
import io
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from athanor import (
    Athanor,
    AthanorError,
    Endpoint,
    FieldSpec,
    Limits,
    Procedure,
    Role,
    ValidationError,
    record,
)
from athanor.programs import (
    argument,
    binary,
    choose,
    field,
    has_field,
    literal,
    structural,
    variable,
)

AVAILABLE = importlib.util.find_spec("torch") is not None
if AVAILABLE:
    import torch

    from athanor.grounded.app import Application, exact_structural, structural_input
    from athanor.grounded.checkpoint import restore, save
    from athanor.grounded.induction import Induction, discrepancy
    from athanor.grounded.induction_demo import run_demo
    from athanor.grounded.runtime import Runtime
    from athanor.grounded.training import GroundedTrainer
    from athanor.world.situations import ProblemStream


class StructuralAlgebraContracts(unittest.TestCase):
    def setUp(self):
        self.k = Athanor(namespace="structural-contracts")
        self.c = self.k.teach_category("value", fields={"value": FieldSpec()})

    def operation(self, expr):
        return self.k.teach_operation(
            "generic", Procedure((Endpoint(Role.DATA, self.c),), self.c, {"value": expr})
        )

    def compute(self, expr, value):
        op = self.operation(expr)
        data = self.k.create(self.c, {"value": value})
        return self.k.get(self.k.execute(op, (data,))).fields["value"]

    def graph(self):
        return record(nodes=tuple(record(p=v) for v in (3, 1, 2, 1)), edges=((0, 1), (1, 2)))

    def test_directed_roles_and_reverse_preserve_edge_participants(self):
        value = self.graph()
        root = field(argument(0), "value")
        reverse = self.compute(structural("reverse", root), value)
        self.assertEqual(reverse["edges"], ((Fraction(3), Fraction(2)), (Fraction(2), Fraction(1))))
        self.assertEqual(self.compute(structural("reverse", root), reverse), value)
        self.assertEqual(self.compute(structural("in_degree", root, literal(0)), value), 0)
        self.assertEqual(self.compute(structural("out_degree", root, literal(0)), value), 1)

    def test_connectivity_is_exact_grouping_without_given_group_labels(self):
        output = self.compute(structural("components", field(argument(0), "value")), self.graph())
        self.assertEqual(output, ((Fraction(0), Fraction(1), Fraction(2)), (Fraction(3),)))

    def test_role_substitution_and_different_equal_participants(self):
        own = field(variable("item"), "p")
        peer = field(variable("peer"), "p")
        expr = structural(
            "select",
            field(argument(0), "value"),
            structural("exists", variable("root"), binary("eq", own, peer)),
        )
        result = self.compute(expr, (record(p="z"), record(p="a"), record(p="z")))
        self.assertEqual(result, (record(p="z"), record(p="z")))
        self.assertEqual(self.compute(expr, (record(p="q"),)), ())

    def test_ordered_pair_relation_does_not_commute(self):
        predicate = structural(
            "exists",
            variable("root"),
            binary("lt", field(variable("item"), "p"), field(variable("peer"), "p")),
        )
        result = self.compute(
            structural("select", field(argument(0), "value"), predicate),
            (record(p=3), record(p=1), record(p=2)),
        )
        self.assertEqual(result, (record(p=1), record(p=2)))

    def test_missing_property_remains_unknown(self):
        predicate = choose(
            has_field(variable("item"), "p"),
            binary("eq", field(variable("item"), "p"), literal("a")),
            literal(None),
        )
        result = self.compute(
            structural("select", field(argument(0), "value"), predicate),
            (record(p="a"), record(q="a")),
        )
        self.assertIsNone(result)

    def test_bool_number_distinction_and_collection_uniqueness(self):
        result = self.compute(structural("unique", field(argument(0), "value")), (True, 1, True, 1))
        self.assertEqual(result, (True, Fraction(1)))
        self.assertEqual(len(result), 2)

    def test_sorting_bound_variables_reindexes_graph_relations(self):
        result = self.compute(
            structural("sort", field(argument(0), "value"), field(variable("item"), "p")),
            self.graph(),
        )
        self.assertEqual(tuple(n["p"] for n in result["nodes"]), (1, 1, 2, 3))
        self.assertEqual(result["edges"], ((Fraction(3), Fraction(0)), (Fraction(0), Fraction(2))))

    def test_raster_geometry_is_noncommutative(self):
        grid = (("a", "b", None), ("c", None, "d"))
        v = field(argument(0), "value")
        a = self.compute(structural("transpose", structural("reflect_x", v)), grid)
        b = self.compute(structural("reflect_x", structural("transpose", v)), grid)
        self.assertNotEqual(a, b)
        self.assertEqual(a, ((None, "d"), ("b", None), ("a", "c")))

    def test_raw_raster_relations_and_roundtrip(self):
        raw = (("a", "b", None), (None, "c", "d"))
        v = field(argument(0), "value")
        graph = self.compute(structural("raster_graph", v), raw)
        self.assertEqual(
            graph["edges"],
            ((Fraction(0), Fraction(1)), (Fraction(1), Fraction(2)), (Fraction(2), Fraction(3))),
        )
        self.assertEqual(self.compute(structural("rasterize", v), graph), raw)

    def test_exact_translation_and_empty_cell_protocol(self):
        result = self.compute(
            structural("shift_x", field(argument(0), "value"), literal(1)), (("a", "b", "c"),)
        )
        self.assertEqual(result, ((None, "a", "b"),))

    def test_resource_failure_is_atomic_for_per_cell_work(self):
        op = self.operation(structural("transpose", field(argument(0), "value")))
        data = self.k.create(self.c, {"value": ((1,) * 16,) * 16})
        before = self.k.dumps()
        with self.assertRaises(AthanorError):
            self.k.execute(op, (data,), limits=Limits(max_steps=30))
        self.assertEqual(self.k.dumps(), before)

    def test_bad_algorithm_scope_and_shape_reject(self):
        for expr in (
            structural("python_eval", literal("x")),
            structural("count"),
            variable("root"),
        ):
            before = self.k.dumps()
            with self.assertRaises(ValidationError):
                self.operation(expr)
            self.assertEqual(self.k.dumps(), before)
        with self.assertRaises(AthanorError):
            self.compute(structural("transpose", field(argument(0), "value")), ((1, 2), (3,)))
        with self.assertRaises(AthanorError):
            self.compute(
                structural("degree", field(argument(0), "value"), literal(0)),
                record(nodes=(record(p=1),), edges=((0, 8),)),
            )

    def test_native_persistence_contains_executable_structural_language(self):
        op = self.operation(structural("reverse", field(argument(0), "value")))
        data = self.k.create(self.c, {"value": ("x", "y", "z")})
        restored = Athanor.loads(self.k.dumps(compact=True))
        self.assertEqual(restored.dumps(), self.k.dumps())
        self.assertEqual(
            restored.get(restored.execute(op, (data,))).fields["value"], ("z", "y", "x")
        )

    def test_peer_exclusion_applies_only_to_the_same_collection(self):
        expr = structural(
            "select",
            field(argument(0), "value"),
            structural(
                "exists", literal(("other",)), binary("eq", variable("peer"), literal("other"))
            ),
        )
        self.assertEqual(self.compute(expr, ("own",)), ("own",))


@unittest.skipUnless(AVAILABLE, "Install neural extra for structural induction")
class InductionContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.runtime = Runtime()
        self.i = self.runtime.observer.induction
        self.trainer = GroundedTrainer(self.runtime.observer)
        self.i.install()

    def examples(self):
        return tuple((tuple(s), tuple(reversed(s))) for s in ("abc", "defg", "hijklm"))

    def acquire(self):
        rule = self.i.infer(self.examples(), self.trainer, max_calls=1, max_candidates=32)
        self.assertIsNotNone(rule)
        return rule

    def test_shared_model_infers_rule_for_unfamiliar_symbolic_instances(self):
        count = sum(p.numel() for p in self.runtime.observer.observer.model.parameters())
        old = {
            n: p.detach().clone()
            for n, p in self.runtime.observer.observer.model.named_parameters()
        }
        rule = self.acquire()
        self.assertEqual(self.i.apply(rule, ("unseen-a", "unseen-b")), ("unseen-b", "unseen-a"))
        self.assertEqual(
            sum(p.numel() for p in self.runtime.observer.observer.model.parameters()), count
        )
        self.assertTrue(
            any(
                not torch.equal(old[n], p)
                for n, p in self.runtime.observer.observer.model.named_parameters()
            )
        )
        self.assertGreater(self.i.neural_updates, 0)
        self.assertTrue(any(c.status == "qualified" for c in self.i.candidates))

    def test_neural_selection_is_a_real_gate_and_evidence_vetoes(self):
        original = self.i.rank

        def bad_first(candidates, **kwargs):
            return sorted(original(candidates, **kwargs), key=lambda c: c.status == "supported")

        with patch.object(self.i, "rank", side_effect=bad_first):
            self.assertIsNone(
                self.i.infer(self.examples(), self.trainer, max_calls=1, max_candidates=32)
            )
        self.assertFalse(self.i.rules)
        self.assertTrue(any(c.status == "supported" for c in self.i.candidates))

    def test_unnamed_concepts_and_recursive_qualification_preserve_history(self):
        g = ProblemStream(7123, "relational")
        inputs = tuple(g.observe() for _ in range(3))
        self.i.form_concepts(inputs)
        first = next(
            c
            for c in self.i.concepts
            if self.i.knowledge.get(c).fields["predicate"].get("op") in {"lt", "eq"}
        )
        second = self.i.combine_concepts(first, first, inputs)
        witness = self.i.knowledge.create(
            fields=record(observed=False), source="contradiction fixture"
        )
        old = self.i.knowledge.get(first).fields
        self.i.qualify(first, witness)
        revised = next(c for c in self.i.concepts if c.id == second.id)
        self.assertEqual(self.i.knowledge.get(revised).fields["status"], "qualified")
        self.assertEqual(self.i.knowledge.get(first).fields, old)
        self.assertNotIn(self.i.knowledge.get(first).fields["operation"], self.i.available())

    def test_ambiguous_or_insufficient_examples_are_not_adopted(self):
        symmetric = (("a", None, "a"), (None, "b", None), ("a", None, "a"))
        rule = self.i.infer(
            ((symmetric, symmetric),), self.trainer, max_calls=1, max_candidates=256
        )
        self.assertIsNone(rule)
        self.assertGreaterEqual(sum(c.errors == [0] for c in self.i.candidates), 2)
        # A whole unknown outcome is not positive transformation evidence.
        unknown = self.i.operation(literal(None), source="unavailable outcome fixture")
        self.i.library.append(unknown)
        self.assertIsNone(
            self.i.infer(
                tuple(((v,), None) for v in ("u", "v", "w")),
                self.trainer,
                max_calls=1,
                max_candidates=256,
            )
        )
        self.assertFalse(any(c.status == "supported" for c in self.i.candidates))
        self.assertTrue(all(c.status == "unknown" for c in self.i.candidates))

    def test_repeat_success_is_not_reliable_generalization(self):
        rule = self.acquire()
        for _ in range(3):
            self.i.apply(rule, ("u", "v"), expected=("v", "u"), measured=True)
            rule = self.i.rules[-1]
        self.assertEqual(self.i.capabilities[str(rule)]["generalization_successes"], 0)

    def test_wrong_context_and_contradiction_withhold_reuse(self):
        rule = self.acquire()
        self.assertIsNone(self.i.apply(rule, (1, 2)))
        self.i.apply(rule, ("u", "v"), expected=("u", "v"), measured=True)
        current = self.i.rules[-1]
        self.assertEqual(self.i.knowledge.get(current).fields["status"], "qualified")
        self.assertIsNone(self.i.apply(current, ("a", "b")))

    def test_resource_exhaustion_is_unknown_and_initial_goal_is_recognized(self):
        value = ("a", "b")
        self.assertIsNone(
            self.i.solve(
                value, (record(kind="count", value=20),), self.trainer, max_depth=1, max_nodes=1
            )
        )
        self.assertEqual(self.i.status, "unknown")
        self.assertIsNone(self.i.solve(value, (record(kind="count", value=2),), self.trainer))
        self.assertEqual(
            self.i.knowledge.get(self.i.tasks[-1]).fields["status"], "already_satisfied"
        )

    def test_structural_constructor_typing_and_atomic_publication(self):
        before = self.i.knowledge.dumps()
        with self.assertRaises(ValidationError):
            self.runtime.observer.processes.structural_call(
                self.i.category, (self.runtime.observer.operations["forward"],)
            )
        self.assertEqual(self.i.knowledge.dumps(), before)
        operation = self.i.library[0]
        before = self.i.knowledge.dumps()
        original = Athanor.create

        def fail(staged, *args, **kwargs):
            if kwargs.get("source") == "structural process proposal":
                raise ValidationError("late staged failure")
            return original(staged, *args, **kwargs)

        with patch.object(Athanor, "create", fail), self.assertRaises(ValidationError):
            self.runtime.observer.processes.structural_call(self.i.category, (operation,))
        self.assertEqual(self.i.knowledge.dumps(), before)

    def test_checkpoint_restoration_and_next_training_update_are_identical(self):
        rule = self.acquire()
        self.runtime.observer.observe(self.runtime.world.sense())
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "concepts.athanor"
            save(self.runtime, path, trainer=self.trainer)
            other, trainer = restore(path)
        new = other.observer.induction
        self.assertEqual(self.i.metadata(), new.metadata())
        self.i.learn(self.trainer, self.i.candidates, steps=1)
        new.learn(trainer, new.candidates, steps=1)
        for name, p in self.runtime.observer.observer.model.state_dict().items():
            self.assertTrue(torch.equal(p, other.observer.observer.model.state_dict()[name]), name)
        new.examples.clear()
        new.observations.clear()
        self.assertEqual(new.apply(rule, ("new", "other")), ("other", "new"))

    def test_restore_rejects_forged_scopes_handles_and_confidence(self):
        self.acquire()
        for key, value in (
            ("selected", ["s99999999", 1, self.i.knowledge.namespace]),
            ("neural_updates", -1),
            ("candidate_scores", [float("nan")] * len(self.i.candidates)),
            ("resource", {"max_calls": 100, "beam": 1, "max_candidates": 1}),
            ("capabilities", {"fake": {"execution_attempts": 0, "execution_successes": 1}}),
        ):
            data = copy.deepcopy(self.i.metadata())
            data[key] = value
            with self.assertRaises(ValidationError):
                Induction.restore(self.runtime.observer, data)

    def test_application_problem_path_and_inspection_are_real_and_pure(self):
        from athanor.values import freeze

        value = record({"amount": Fraction(1, 7), "values": (True, None, "symbol")})
        self.assertEqual(freeze(structural_input(exact_structural(value))), value)
        with tempfile.TemporaryDirectory() as d:
            app = Application(self.runtime, Path(d) / "app.athanor", self.trainer)
            result = app.command(
                {
                    "op": "infer_rule",
                    "examples": [{"before": list(a), "after": list(b)} for a, b in self.examples()],
                    "max_calls": 1,
                }
            )
            self.assertIsNotNone(result["result"])
            rule = self.i.rules[-1]
            self.assertEqual(
                app.command({"op": "apply_rule", "reference": str(rule), "input": ["x", "y"]})[
                    "result"
                ],
                ["y", "x"],
            )
            before = self.i.knowledge.dumps()
            meta = self.i.metadata()
            app.state()
            app.state()
            self.assertEqual(self.i.knowledge.dumps(), before)
            self.assertEqual(self.i.metadata(), meta)
            app.command({"op": "pause_learning"})
            prior = self.trainer.updates
            self.i.learn(self.trainer, self.i.candidates, steps=2)
            self.assertEqual(self.trainer.updates, prior)

    def test_structural_discrepancy_zero_requires_exact_equivalence(self):
        self.assertGreater(discrepancy(record(x=None), record(y=None)), 0)
        self.assertGreater(discrepancy(True, Fraction(1)), 0)
        self.assertEqual(discrepancy(record(x=(1, 2)), record(x=(1, 2))), 0)

    def test_multistep_goal_creates_provisional_variable_procedure(self):
        self.acquire()
        value = ("a", "a", "b")
        goal = (record(kind="count", value=2), record(kind="equals", value=("b", "a")))
        rule = self.i.solve(value, goal, self.trainer, max_depth=2, max_nodes=128)
        self.assertIsNotNone(rule)
        fields = self.i.knowledge.get(rule).fields
        self.assertGreaterEqual(len(fields["dependencies"]), 2)
        self.assertEqual(fields["status"], "provisional")
        self.assertEqual(self.i.apply(rule, value), ("b", "a"))
        self.i.apply(rule, ("x", "y", "x"))
        failures = [
            self.i.knowledge.get(v).fields
            for v in self.i.history
            if self.i.knowledge.get(v).fields.get("kind") == "structural_application"
        ]
        self.assertFalse(failures[-1]["goal_satisfied"])
        self.assertFalse(failures[-1]["success"])

    def test_checkpoint_can_save_an_active_structural_query(self):
        rule = self.acquire()
        self.i.apply(rule, ("r", "s"))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "active.athanor"
            save(self.runtime, path, trainer=self.trainer)
            other, resumed = restore(path)
        self.assertEqual(self.i.metadata(), other.observer.induction.metadata())
        self.i.learn(self.trainer, self.i.candidates, steps=1)
        other.observer.induction.learn(resumed, other.observer.induction.candidates, steps=1)
        for n, p in self.runtime.observer.observer.model.state_dict().items():
            self.assertTrue(torch.equal(p, other.observer.observer.model.state_dict()[n]), n)

    def test_new_evidence_can_replace_preferred_rule_and_restore_refinement(self):
        rule = self.i.infer(
            tuple(((v,), (v,)) for v in ("a", "b", "c")),
            self.trainer,
            max_calls=1,
            max_candidates=32,
        )
        self.assertIsNotNone(rule)
        old = self.i.knowledge.get(rule).fields
        self.assertGreaterEqual(len(old["alternatives"]), 2)
        value = ("p", "p", "q")
        preferred, _ = self.i.evaluate(old["operation"], value)
        expected = ("p", "q") if preferred == ("q", "p", "p") else ("q", "p", "p")
        self.assertIsNone(self.i.apply(rule, value))
        self.assertEqual(self.i.apply(rule, value, expected=expected, measured=True), expected)
        revised = self.i.rules[-1]
        new = self.i.knowledge.get(revised).fields
        self.assertNotEqual(old["operation"], new["operation"])
        self.assertNotIn(old["operation"], self.i.available())
        self.assertEqual(self.i.knowledge.get(rule).fields, old)
        self.assertTrue(new["contradictions"])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "refined.athanor"
            save(self.runtime, path, trainer=self.trainer)
            other, _ = restore(path)
        self.assertEqual(self.i.metadata(), other.observer.induction.metadata())
        self.assertEqual(other.observer.induction.apply(revised, value), expected)

    def test_repeated_observation_does_not_support_a_new_concept(self):
        graph = record(nodes=(record(p="a"), record(p="b")), edges=((0, 1),))
        self.i.form_concepts((graph, graph, graph))
        self.assertTrue(self.i.concepts)
        self.assertTrue(
            all(self.i.knowledge.get(c).fields["status"] == "provisional" for c in self.i.concepts)
        )


@unittest.skipUnless(AVAILABLE, "Install neural extra for concept acceptance")
class ConceptAcceptance(unittest.TestCase):
    @unittest.skipUnless(
        Path("artifacts/process-observer.athanor").exists(),
        "Restore Goal 006 learned checkpoint first",
    )
    def test_complete_independent_cross_domain_learning_goal_and_restore(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            report = run_demo(Path(d) / "concepts.athanor")
        for capability in (
            "held_out_relational_exact",
            "held_out_spatial_exact",
            "cross_context_exact",
            "goal_achieved",
            "restoration_exact",
            "solving_after_restore",
        ):
            self.assertTrue(report[capability], capability)
        self.assertGreaterEqual(report["plan_depth"], 2)
        self.assertGreaterEqual(report["uncertain_alternatives"], 2)
        self.assertFalse(report["hidden_rule_supplied"])
        self.assertFalse(report["arc_access"])


if __name__ == "__main__":
    unittest.main()
