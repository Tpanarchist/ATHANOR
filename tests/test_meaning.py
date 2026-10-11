"""Native semantic knowledge and grounding remain usable without a neural runtime."""

import json
import unittest
from fractions import Fraction

from athanor import (
    Assessment,
    Athanor,
    Endpoint,
    Evidence,
    FieldSpec,
    Procedure,
    Role,
    ValidationError,
    record,
)
from athanor.programs import argument, binary, category, field
from athanor.semantics import Example, Lexicon, Teacher
from athanor.semantics.language import Language, meaning_evidence, tokenize
from athanor.semantics.lessons import import_lesson


class MeaningContracts(unittest.TestCase):
    def setUp(self):
        self.a = Athanor(namespace="native-meaning")
        self.number = self.a.teach_category("Quantity", {"value": FieldSpec("number")})
        self.x = self.a.create(self.number, {"value": 7})
        self.y = self.a.create(self.number, {"value": 2})
        self.teacher = Teacher(self.a)
        self.procedure = Procedure(
            (Endpoint(Role.DATA, self.number),) * 2,
            self.number,
            {"value": binary("sub", field(argument(0), "value"), field(argument(1), "value"))},
        )
        self.concept = self.teacher.define(
            "difference",
            "The ordered difference of exact quantities",
            self.procedure,
            expressions=("difference", "subtract"),
            examples=(Example((self.x, self.y), record(value=5)),),
            roles=("minuend", "subtrahend"),
        )

    def unchanged(self, action):
        before = self.a.dumps()
        with self.assertRaises(ValidationError):
            action()
        self.assertEqual(before, self.a.dumps())

    def test_transaction_rolls_back_identity_receipts_and_successful_nested_execution(self):
        before = self.a.dumps()

        def fail(staged):
            staged.execute(self.a.get(self.concept).fields["operation"], (self.x, self.y))
            staged.create(self.number, {"value": "invalid"})

        self.unchanged(lambda: self.a.transaction(fail))
        restored = Athanor.loads(before)
        self.assertEqual(self.a.create(fields={}), restored.create(fields={}))

    def test_transaction_rejects_mutating_original_and_nested_staging_is_valid(self):
        self.unchanged(lambda: self.a.transaction(lambda staged: self.a.create(fields={})))
        result = self.a.transaction(
            lambda staged: staged.transaction(lambda nested: nested.create(fields={"ok": True}))
        )
        self.assertTrue(self.a.get(result).fields["ok"])

    def test_category_reflection_is_exact_optional_and_persistent(self):
        op = self.a.teach_operation(
            "reflection",
            Procedure((Endpoint(Role.DATA),), None, {"category": category(argument(0))}),
        )
        result = self.a.execute(op, (self.x,))
        self.assertEqual(self.a.get(result).fields["category"], self.number)
        bare = self.a.create()
        self.assertIsNone(self.a.get(self.a.execute(op, (bare,))).fields["category"])
        restored = Athanor.loads(self.a.dumps())
        self.assertEqual(
            restored.get(restored.execute(op, (self.y,))).fields["category"], self.number
        )

    def test_failed_exact_examples_rollback_complete_lesson(self):
        self.unchanged(
            lambda: self.teacher.define(
                "bad",
                "wrong expected result",
                self.procedure,
                expressions=("bad",),
                examples=(Example((self.x, self.y), record(value=999)),),
            )
        )
        self.assertIsNone(self.teacher.lexicon.find("bad"))

    def test_synonyms_share_referent_but_keep_symbol_identities(self):
        lex = self.teacher.lexicon
        self.assertNotEqual(lex.find("difference"), lex.find("subtract"))
        self.assertEqual(lex.bindings("subtract")[0].referent, self.concept)
        self.assertEqual(lex.find(" DIFFERENCE "), lex.find("difference"))
        self.assertEqual(lex.bindings("differences"), ())

    def test_polysemy_context_unknown_conflict_and_correction_remain_explicit(self):
        other = self.a.create(
            fields={"kind": "concept", "operation": self.a.get(self.concept).fields["operation"]}
        )
        old = self.teacher.alias("balance", self.concept, context=record(domain="money"))
        self.teacher.alias("balance", other, context=record(domain="movement"))
        self.assertEqual(len(self.teacher.lexicon.bindings("balance")), 2)
        self.assertEqual(
            self.teacher.lexicon.bindings("balance", record(domain="money"))[0].referent,
            self.concept,
        )
        correction = self.teacher.correct(
            "balance",
            (old,),
            other,
            reason="This convention refers to the second concept",
            context=record(domain="money"),
        )
        self.assertEqual(self.a.get(correction).fields["supersedes"], (old,))
        self.assertEqual(
            self.teacher.lexicon.bindings("balance", record(domain="money"))[0].referent, other
        )
        self.assertEqual(self.a.get(old).fields["referent"], self.concept)
        binding = self.teacher.alias("contested", self.concept)
        symbol = self.teacher.lexicon.find("contested")
        self.a.bind(
            self.teacher.lexicon.relation,
            symbol,
            binding,
            Evidence("second instructor", "negative"),
        )
        self.assertEqual(
            self.teacher.lexicon.bindings("contested")[0].assessment, Assessment.CONFLICT
        )
        self.teacher.alias("uncertain", self.concept, evidence=Evidence("unverified", "uncertain"))
        self.assertEqual(
            self.teacher.lexicon.bindings("uncertain")[0].assessment, Assessment.UNKNOWN
        )

    def test_correction_cannot_silently_replace_different_expression(self):
        old = self.teacher.alias("dax", self.concept)
        self.unchanged(
            lambda: self.teacher.correct("subtract", (old,), self.concept, reason="wrong scope")
        )

    def test_context_equality_retains_boolean_numeric_distinction(self):
        binding = self.teacher.alias("typed", self.concept, context=record(flag=1))
        self.assertEqual(self.teacher.lexicon.bindings("typed", record(flag=True)), ())
        self.assertEqual(len(self.teacher.lexicon.bindings("typed", record(flag=1))), 1)
        self.unchanged(
            lambda: self.teacher.correct(
                "typed",
                (binding,),
                self.concept,
                reason="wrong context type",
                context=record(flag=True),
            )
        )

    def test_teaching_requires_exact_refs_and_valid_role_descriptions(self):
        self.unchanged(lambda: self.teacher.alias("bad", self.concept.id))
        self.unchanged(
            lambda: self.teacher.define(
                "bad", "definition", self.procedure, expressions=("bad",), roles=(None, "second")
            )
        )

    def test_specialization_inherits_contracts_and_observation_preserves_attribution(self):
        special = self.teacher.specialize("Positive", self.number, {}, expressions=("positive",))
        participant, witness = self.teacher.observe(
            special, record(value=4), expression="sample", source="instrument record"
        )
        self.assertIn(self.number, self.a.ancestors(special))
        self.assertEqual(self.a.get(witness).fields["participant"], participant)
        self.assertEqual(
            self.a.inspect(participant).receipt.details["source"], "observation:instrument record"
        )
        self.unchanged(
            lambda: self.teacher.observe(
                special, record(value="four"), expression="sample", source="instrument record"
            )
        )

    def test_teacher_assertion_is_unknown_until_separate_matching_observation(self):
        relation = self.a.teach_relation("compared with", Endpoint(Role.DATA), Endpoint(Role.DATA))
        self.teacher.claim(relation, self.x, self.y, source="offline teacher")
        self.assertEqual(self.a.assess(self.x, relation, self.y), Assessment.UNKNOWN)
        fake = self.a.create(
            fields={
                "kind": "relation_observation",
                "relation": relation,
                "subject": self.x,
                "target": self.y,
                "stance": "positive",
            },
            source="teacher",
        )
        self.unchanged(
            lambda: self.teacher.claim(relation, self.x, self.y, source="teacher", observation=fake)
        )
        witness = self.teacher.observe_relation(
            relation, self.x, self.y, source="local measured comparison"
        )
        claim = self.teacher.claim(
            relation, self.x, self.y, source="observed comparison", observation=witness
        )
        self.assertEqual(self.a.assess(self.x, relation, self.y), Assessment.SUPPORTED)
        support = next(
            node for node in self.a.structures if node.fields.get("kind") == "claim_support"
        )
        self.assertEqual(support.fields["claim"], claim)
        self.assertEqual(support.fields["observation"], witness)
        self.assertEqual(self.a.assess(self.y, relation, self.x), Assessment.UNKNOWN)

    def test_language_keeps_textual_order_not_arithmetic_roles(self):
        utterance = Language(self.a, self.number).encode("subtract 2 from 7")
        self.assertEqual(
            tuple(self.a.get(ref).fields["value"] for ref in utterance.mentions),
            (Fraction(2), Fraction(7)),
        )
        self.assertEqual(meaning_evidence(self.a, utterance), {self.concept: Assessment.SUPPORTED})
        self.assertNotIn("operation", self.a.get(utterance.ref).fields)

    def test_multiword_convention_and_exact_literals_use_shared_lexical_mechanism(self):
        self.teacher.alias("ordered gap", self.concept)
        utterance = Language(self.a, self.number).encode("ordered gap -3/2 and 2.5")
        self.assertEqual(utterance.unknown, ())
        self.assertEqual(meaning_evidence(self.a, utterance), {self.concept: Assessment.SUPPORTED})
        self.assertEqual(
            tuple(self.a.get(ref).fields["value"] for ref in utterance.mentions),
            (Fraction(-3, 2), Fraction(5, 2)),
        )

    def test_invalid_lexical_input_and_missing_pronoun_are_explicit(self):
        language = Language(self.a, self.number)
        for text in ("divide 1/0", "difference $2 and3", "x" * 65, "difference 1; add2"):
            self.unchanged(lambda text=text: language.encode(text))
        self.assertIn("it", language.encode("difference it and 2").unknown)
        self.assertEqual(tokenize("Difference 2 and 7?"), ("difference", "2", "and", "7", "?"))

    def test_native_meanings_restore_without_parameters_or_transient_state(self):
        restored = Athanor.loads(self.a.dumps())
        lex = Lexicon(restored)
        self.assertEqual(lex.bindings("subtract")[0].referent, self.concept)
        operation = restored.get(self.concept).fields["operation"]
        self.assertEqual(
            restored.get(restored.execute(operation, (self.x, self.y))).fields["value"], 5
        )

    def package(self, expected=5):
        def ref(value):
            return {"ref": [value.id, value.revision, value.namespace]}

        def expression(value):
            from athanor import Record

            if isinstance(value, Record):
                return {key: expression(item) for key, item in value.items()}
            if isinstance(value, Fraction):
                return {"fraction": [value.numerator, value.denominator]}
            return value

        return {
            "format": "athanor.lesson.v1",
            "source": "offline prepared material",
            "procedures": [
                {
                    "name": "new",
                    "definition": "ordered subtraction",
                    "expressions": ["new"],
                    "procedure": {
                        "inputs": [{"role": "data", "category": ref(self.number)}] * 2,
                        "result": ref(self.number),
                        "fields": {
                            key: expression(value) for key, value in self.procedure.fields.items()
                        },
                    },
                    "examples": [
                        {"arguments": [ref(self.x), ref(self.y)], "expected": {"value": expected}}
                    ],
                }
            ],
        }

    def test_offline_prepared_procedure_executes_after_independent_checks(self):
        installed = import_lesson(self.a, json.dumps(self.package()))
        self.assertEqual(
            self.a.get(self.a.execute(installed["new.operation"], (self.x, self.y))).fields[
                "value"
            ],
            5,
        )
        self.assertEqual(self.teacher.lexicon.bindings("new")[0].referent, installed["new"])

    def test_offline_wrong_examples_code_duplicate_keys_and_missing_checks_rejected(self):
        self.unchanged(lambda: import_lesson(self.a, json.dumps(self.package(999))))
        package = self.package()
        package["procedures"][0]["examples"] = []
        self.unchanged(lambda: import_lesson(self.a, json.dumps(package)))
        package = self.package()
        package["python"] = "print('not executable')"
        self.unchanged(lambda: import_lesson(self.a, json.dumps(package)))
        self.unchanged(lambda: import_lesson(self.a, '{"format":"a","format":"b"}'))

    def test_offline_late_failure_discards_earlier_valid_parts(self):
        package = self.package()
        bad = self.package(999)["procedures"][0]
        bad["name"] = "second"
        package["procedures"].append(bad)
        self.unchanged(lambda: import_lesson(self.a, json.dumps(package)))
        self.assertIsNone(self.teacher.lexicon.find("new"))

    def test_offline_invalid_role_reports_validation_error_and_preserves_state(self):
        package = self.package()
        package["procedures"][0]["procedure"]["inputs"][0]["role"] = "imaginary"
        self.unchanged(lambda: import_lesson(self.a, json.dumps(package)))

    def test_offline_empirical_claims_remain_unknown_even_with_positive_stance(self):
        relation = self.a.teach_relation(
            "teacher assertion", Endpoint(Role.DATA), Endpoint(Role.DATA)
        )

        def ref(value):
            return {"ref": [value.id, value.revision, value.namespace]}

        package = {
            "format": "athanor.lesson.v1",
            "source": "AI author",
            "procedures": [],
            "claims": [
                {
                    "relation": ref(relation),
                    "subject": ref(self.x),
                    "target": ref(self.y),
                    "stance": "positive",
                }
            ],
        }
        import_lesson(self.a, json.dumps(package))
        self.assertEqual(self.a.assess(self.x, relation, self.y), Assessment.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
