"""Exact construction, evidence boundaries, native reuse and restoration contracts."""

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
from athanor.programs import argument, binary, field, literal, structural, variable
from athanor.values import as_ref

AVAILABLE = importlib.util.find_spec("torch") is not None
if AVAILABLE:
    import torch

    from athanor.grounded.app import Application
    from athanor.grounded.checkpoint import restore, save
    from athanor.grounded.representations import Representations
    from athanor.grounded.runtime import Runtime
    from athanor.grounded.training import GroundedTrainer
    from athanor.world.interpretations import Correspondences


class RepresentationAlgebra(unittest.TestCase):
    def setUp(self):
        self.k = Athanor(namespace="representation-algebra")
        self.category = self.k.teach_category("generic", fields={"value": FieldSpec()})
        self.root = field(argument(0), "value")

    def execute(self, expr, value, limits=None):
        op = self.k.teach_operation(
            "generic",
            Procedure((Endpoint(Role.DATA, self.category),), self.category, {"value": expr}),
        )
        datum = self.k.create(self.category, {"value": value})
        return self.k.get(self.k.execute(op, (datum,), limits=limits)).fields["value"]

    def test_relation_construction_uses_ordered_roles_and_preserves_equal_individuals(self):
        relation = structural(
            "relate",
            self.root,
            binary("lt", field(variable("item"), "p"), field(variable("peer"), "p")),
        )
        result = self.execute(relation, (record(p=2), record(p=1), record(p=1)))
        self.assertEqual(result["edges"], ((Fraction(1), Fraction(0)), (Fraction(2), Fraction(0))))
        self.assertEqual(len(result["nodes"]), 3)

    def test_unknown_pair_cannot_become_an_authoritative_missing_edge(self):
        result = self.execute(
            structural("relate", self.root, literal(None)), (record(p=1), record(p=2))
        )
        self.assertIsNone(result)

    def test_induced_groups_reopen_participants_and_directed_edges(self):
        graph = record(nodes=(record(p="a"), record(p="b"), record(p="c")), edges=((1, 0),))
        groups = self.execute(structural("groups", self.root), graph)
        self.assertEqual(groups[0]["source_indices"], (Fraction(0), Fraction(1)))
        self.assertEqual(groups[0]["edges"], ((Fraction(1), Fraction(0)),))
        self.assertEqual(groups[1]["nodes"], (record(p="c"),))
        reversed_group = self.execute(structural("reverse", self.root), groups[0])
        self.assertEqual(reversed_group["source_indices"], (Fraction(1), Fraction(0)))
        nested = self.execute(structural("groups", self.root), reversed_group)
        self.assertEqual(nested[0]["source_indices"], (Fraction(1), Fraction(0)))

    def test_exact_reductions_and_unknowns(self):
        values = (Fraction(1, 3), Fraction(2, 7))
        self.assertEqual(self.execute(structural("sum", self.root), values), Fraction(13, 21))
        self.assertEqual(self.execute(structural("minimum", self.root), values), Fraction(2, 7))
        self.assertEqual(self.execute(structural("maximum", self.root), values), Fraction(1, 3))
        self.assertIsNone(self.execute(structural("minimum", self.root), ()))
        self.assertIsNone(self.execute(structural("sum", self.root), (None,)))
        with self.assertRaises(AthanorError):
            self.execute(structural("sum", self.root), (True,))

    def test_relation_resource_failure_rolls_back_entire_execution(self):
        op = self.k.teach_operation(
            "bounded",
            Procedure(
                (Endpoint(Role.DATA, self.category),),
                self.category,
                {"value": structural("relate", self.root, literal(True))},
            ),
        )
        data = self.k.create(self.category, {"value": tuple(record(p=n) for n in range(12))})
        before = self.k.dumps()
        with self.assertRaises(AthanorError):
            self.k.execute(op, (data,), limits=Limits(max_steps=10))
        self.assertEqual(before, self.k.dumps())


@unittest.skipUnless(AVAILABLE, "Install neural extra")
class RepresentationContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.runtime = Runtime()
        self.trainer = GroundedTrainer(self.runtime.observer)
        self.p = self.runtime.observer.representations
        self.stream = Correspondences(820)

    def examples(self):
        s = self.stream
        return tuple(
            (s.collection((2, 3, 4)), s.collection((4, 2, 3)), True) for _ in range(3)
        ) + tuple((s.collection((2, 3, 4)), s.collection((3, 3, 3)), False) for _ in range(3))

    def test_invented_relation_grouping_and_abstraction_recognize_new_instances(self):
        ref = self.p.invent(self.examples(), self.trainer, max_candidates=128, beam=12)
        self.assertIsNotNone(ref)
        concept = self.p.form_concept(
            ref, tuple(self.stream.collection((2, 3, 4)) for _ in range(3))
        )
        self.assertTrue(self.p.recognize(concept, self.stream.collection((2, 3, 4)), expected=True))
        self.assertFalse(
            self.p.recognize(concept, self.stream.collection((3, 3, 3)), expected=False)
        )
        fields = self.p.knowledge.get(ref).fields
        self.assertGreater(len(fields["constituents"]), 2)
        self.assertGreater(self.p.neural_updates, 0)
        self.assertGreater(len(self.p.fragments), 0)

    def test_neural_selection_is_causal_and_evidence_vetoes_it(self):
        original = self.p.induction.rank

        def worst(candidates, **kw):
            return list(reversed(original(candidates, **kw)))

        with patch.object(self.p.induction, "rank", side_effect=worst):
            result = self.p.invent(self.examples(), self.trainer, max_candidates=16, beam=2)
        self.assertIsNone(result)
        self.assertEqual(self.p.status, "unknown")

    def test_insufficient_or_collapsing_evidence_does_not_establish_support(self):
        a = self.stream.collection((2, 3, 4))
        result = self.p.invent(((a, a, True),), self.trainer, max_candidates=4)
        self.assertIsNone(result)
        self.assertTrue(
            any(self.p.knowledge.get(r).fields["status"] == "hypothesized" for r in self.p.active)
        )

    def test_observation_identity_is_distinct_from_interpretation_and_cached_result(self):
        a = self.stream.collection((2, 3, 4))
        self.p.invent(((a, a, True),), self.trainer, max_candidates=4)
        r = self.p.active[0]
        raw = self.p.observe(a)
        first = self.p.interpret(r, raw)
        executions = self.p.search["executed"]
        second = self.p.interpret(r, raw)
        self.assertNotEqual(raw, first)
        self.assertNotEqual(first, second)
        self.assertEqual(
            self.p.knowledge.get(first).fields["result"],
            self.p.knowledge.get(second).fields["result"],
        )
        self.assertTrue(self.p.knowledge.get(second).fields["reused"])
        self.assertEqual(executions, self.p.search["executed"])

    def test_current_metadata_round_trip_and_invalid_active_ref(self):
        self.p.invent(
            ((self.stream.collection((2, 3, 4)), self.stream.collection((2, 3, 4)), True),),
            self.trainer,
            max_candidates=4,
        )
        metadata = self.p.metadata()
        restored = Representations.restore(self.runtime.observer, metadata)
        self.assertEqual(metadata, restored.metadata())
        bad = copy.deepcopy(metadata)
        bad["selected"] = metadata["observations"][0]
        with self.assertRaises(ValidationError):
            Representations.restore(self.runtime.observer, bad)

    def test_incomplete_observation_stays_unknown(self):
        a = self.stream.collection((2, 3, 4))
        self.p.invent(((a, a, True),), self.trainer, max_candidates=4)
        incomplete = self.p.observe(a, sufficient=False)
        self.assertIsNone(self.p.relationship(incomplete, incomplete))

    def test_composite_observation_projection_is_constructed_from_observed_keys(self):
        pairs = tuple(
            (record(channel=(1, 2, 3), noise=n), record(channel=(1, 2, 3), noise=n + 9), True)
            for n in range(3)
        ) + tuple(
            (record(channel=(1, 2, 3), noise=n), record(channel=(1, 2, 4), noise=n), False)
            for n in range(3)
        )
        ref = self.p.invent(pairs, self.trainer, max_candidates=12, beam=4)
        self.assertIsNotNone(ref)
        concept = self.p.form_concept(
            ref, tuple(record(channel=(1, 2, 3), noise=n + 20) for n in range(3))
        )
        self.assertTrue(self.p.recognize(concept, record(channel=(1, 2, 3), noise=99)))
        self.assertFalse(self.p.recognize(concept, record(channel=(1, 2, 4), noise=99)))

    def test_camera_measurement_channels_use_the_existing_encoder_without_scene_labels(self):
        frame = self.runtime.world.sense()
        encoder = self.runtime.observer.observer.sensory
        with patch.object(encoder, "encode", wraps=encoder.encode) as encoded:
            observation = self.p.camera(frame, measurements=True)
        self.assertGreaterEqual(encoded.call_count, 1)
        value = self.p.raw(observation)
        self.assertEqual(set(value), {"foreground", "measurements"})
        for sample in value["measurements"]:
            self.assertEqual(set(sample), {"x", "y", "appearance", "confidence"})
        capture = self.p.knowledge.get(
            as_ref(self.p.knowledge.get(observation).fields["capture"])
        ).fields
        self.assertEqual(capture["rgb"], frame.left.hex())
        self.assertFalse(hasattr(self.p, "world"))

    def test_derived_positions_preserve_translation_and_uniform_scale(self):
        def observation(points, scale=1, offset=0):
            return tuple(record(a=x * scale + offset, b=y * scale - offset) for x, y in points)

        points = ((1, 2), (2, 5), (6, 4))
        a = observation(points)
        pairs = tuple(
            (observation(points, offset=n), observation(points, scale=n + 2, offset=8), True)
            for n in range(3)
        ) + tuple(
            (a, observation(negative), False)
            for negative in (
                ((1, 2), (3, 5), (6, 4)),
                ((1, 2), (2, 3), (6, 4)),
                ((1, 2), (2, 5), (6, 7)),
            )
        )
        ref = self.p.invent(pairs, self.trainer, max_candidates=48, beam=8)
        self.assertIsNotNone(ref)
        concept = self.p.form_concept(ref, tuple(observation(points, offset=n) for n in range(3)))
        self.assertTrue(self.p.recognize(concept, observation(points, scale=7, offset=-9)))
        self.assertFalse(self.p.recognize(concept, observation(((1, 2), (3, 5), (6, 4)))))

    def test_informative_test_revises_alternatives_without_overwriting_pinned_evidence(self):
        a = (Fraction(1), Fraction(2))
        b = (Fraction(2), Fraction(1))
        self.p.invent(((a, a, True),), self.trainer, max_candidates=6)
        self.assertIsNone(self.p.relationship(a, b))
        prior = tuple(self.p.alternatives)
        proposal = self.p.choose_test(((a, b), (a, a)))
        self.assertIsNotNone(proposal)
        self.p.revise(proposal[0], proposal[1], False)
        self.assertFalse(self.p.relationship(a, b))
        rejected = [
            r
            for r in prior
            if self.p.knowledge.get(self.p.current(r)).fields["status"] == "qualified"
        ]
        self.assertTrue(rejected)
        self.assertTrue(
            all(self.p.knowledge.get(r).fields["status"] == "hypothesized" for r in rejected)
        )
        self.assertEqual(
            self.p.metadata(),
            Representations.restore(self.runtime.observer, self.p.metadata()).metadata(),
        )

    def test_contradiction_qualifies_dependent_concept_and_saved_cache_stays_valid(self):
        a = self.stream.collection((2, 3, 4))
        self.p.invent(((a, a, True),), self.trainer, max_candidates=4)
        count = next(
            r
            for r in self.p.alternatives
            if self.p.knowledge.get(r).fields["output"] == record(family="atom")
        )
        concept = self.p.form_concept(count, (a,))
        self.p.revise(a, self.stream.collection((3, 3, 3)), False)
        self.assertEqual(
            self.p.knowledge.get(self.p.current(concept)).fields["status"], "qualified"
        )
        self.assertEqual(self.p.knowledge.get(concept).fields["status"], "provisional")
        self.assertIsNone(self.p.recognize(self.p.current(concept), a))
        self.assertEqual(
            self.p.metadata(),
            Representations.restore(self.runtime.observer, self.p.metadata()).metadata(),
        )

    def test_checkpoint_keeps_active_alternatives_cache_optimizer_and_world(self):
        a = self.stream.collection((2, 3, 4))
        self.p.invent(((a, a, True),), self.trainer, max_candidates=4)
        self.p.interpret(self.p.active[0], a)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "observer.athanor"
            save(self.runtime, path, trainer=self.trainer)
            restored, trainer = restore(path)
            self.assertEqual(self.p.metadata(), restored.observer.representations.metadata())
            self.assertEqual(self.runtime.world.dumps(), restored.world.dumps())
            self.assertEqual(self.trainer.updates, trainer.updates)
            self.assertEqual(
                self.p.raw(self.p.observations[0]),
                restored.observer.representations.raw(self.p.observations[0]),
            )

    def test_headless_commands_share_runtime_and_state(self):
        with tempfile.TemporaryDirectory() as folder:
            app = Application(self.runtime, Path(folder) / "observer.athanor", self.trainer)
            result = app.command(
                {
                    "op": "invent_representation",
                    "examples": [{"left": [1, 2], "right": [1, 2], "equivalent": True}],
                    "max_candidates": 4,
                }
            )
            self.assertIsNone(result["result"])
            state = app.state()["representations"]
            self.assertEqual(state["status"], "unknown")
            self.assertGreater(state["counts"]["representations"], 0)

    def test_representation_aware_planning_reuses_general_existing_operations(self):
        result = self.p.solve(
            (Fraction(1), Fraction(2)),
            (record(kind="equals", value=(2, 1)),),
            self.trainer,
            max_depth=1,
            max_nodes=32,
        )
        self.assertIsNotNone(result)
        self.assertEqual(
            self.p.induction.apply(result, (Fraction(1), Fraction(2))), (Fraction(2), Fraction(1))
        )
        operation = as_ref(self.p.knowledge.get(result).fields["operation"])
        self.assertEqual(
            self.p.induction.evaluate(operation, (Fraction(3), Fraction(4)))[0],
            (Fraction(4), Fraction(3)),
        )
        task = self.p.knowledge.get(self.p.tasks[-1]).fields
        self.assertEqual(task["status"], "supported")
        self.assertEqual(task["attempts"], ())

    def test_restoration_rejects_a_cached_output_with_wrong_execution_provenance(self):
        a = self.stream.collection((2, 3, 4))
        self.p.invent(((a, a, True),), self.trainer, max_candidates=4)
        self.p.interpret(self.p.active[0], a)
        metadata = copy.deepcopy(self.p.metadata())
        wrong = self.p.knowledge.create(self.p.induction.category, {"value": a})
        metadata["cache"][0][2] = [wrong.id, wrong.revision, wrong.namespace]
        with self.assertRaises(ValidationError):
            Representations.restore(self.runtime.observer, metadata)

    @unittest.skipUnless(
        Path("artifacts/concept-observer.athanor").is_file(),
        "The real 0.7 checkpoint is required for continuous acceptance",
    )
    def test_real_checkpoint_continuous_cross_interface_acceptance(self):
        from athanor.grounded.representation_demo import run_demo

        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            report = run_demo(Path(folder) / "representations.athanor")
        self.assertTrue(report["restoration_exact"])
        self.assertTrue(report["continued_without_replay"])
        self.assertTrue(report["perceptual_transfer"])
        self.assertGreater(report["supported_fragments"], 0)
        self.assertFalse(report["arc_accessed"])


if __name__ == "__main__":
    unittest.main()
