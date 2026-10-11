"""Two-family semantic lessons with fresh values, words, contexts and category instances."""

import random
from dataclasses import dataclass
from fractions import Fraction

from athanor import Athanor, Endpoint, FieldSpec, Procedure, Ref, Role, record
from athanor.neural import Decision, Expectation, ModelConfig, Observation, Proposal
from athanor.neural.curriculum import Lesson
from athanor.neural.encoding import GraphBuilder
from athanor.neural.observer import measure, record_self_observation
from athanor.programs import (
    argument,
    binary,
    call,
    category,
    choose,
    field,
    fold,
    literal,
    variable,
)
from athanor.values import as_ref

from .language import Language, Utterance
from .teaching import Example, Teacher


def install(seed: int) -> tuple[Athanor, dict[str, Ref]]:
    a = Athanor(namespace=f"meaning-{seed}")
    rng = random.Random(seed)
    for _ in range(rng.randrange(3)):
        a.create(fields={"label": str(rng.getrandbits(32))})
    number = a.teach_category("Quantity", {"value": FieldSpec("number")})
    boolean = a.teach_category("Answer", {"value": FieldSpec("bool")})
    animal = a.teach_category("Animal", {"tag": FieldSpec("string")})
    teacher = Teacher(a)
    bird = teacher.specialize(
        "Bird", animal, {"wings": FieldSpec("number", minimum=0)}, expressions=("bird",)
    )
    teacher.alias("animal", animal)
    robin, witness = teacher.observe(
        bird, record(tag="sample", wings=2), expression="robin", source="local specimen record"
    )
    stone, _ = teacher.observe(
        animal,
        record(tag="unclassified animal"),
        expression="turtle",
        source="local specimen record",
    )
    refs = {
        "number": number,
        "boolean": boolean,
        "animal": animal,
        "bird": bird,
        "robin": robin,
        "turtle": stone,
        "witness": witness,
    }
    # Vary category depth, field names/types and observed instances. The neural
    # role pointers must transfer beyond the original two fixed category shapes.
    specialized = bird
    measurements: dict[str, object] = {"tag": "observed variant", "wings": 2}
    field_names = ["color", "mass", "height", "mark", "pattern", "speed"]
    rng.shuffle(field_names)
    for depth in range(rng.randint(1, 3)):
        kind = rng.choice(("string", "number", "bool"))
        key = field_names[depth]
        measurements[key] = {"string": "observed", "number": 3, "bool": True}[kind]
        specialized = teacher.specialize(
            "observed specialization",
            specialized,
            {key: FieldSpec(kind)},
            expressions=(f"variant-{chr(97 + depth)}",),
        )
    teacher.alias("subtype", specialized)
    specimen, _ = teacher.observe(
        specialized,
        record(measurements),
        expression="specimen",
        source="local varied specimen record",
    )
    refs["subtype"], refs["specimen"] = specialized, specimen
    samples = (a.create(number, {"value": 12}), a.create(number, {"value": 3}))
    operations = [
        ("add", ("add", "sum"), 15),
        ("sub", ("subtract", "difference"), 9),
        ("div", ("divide", "quotient"), 4),
    ]
    rng.shuffle(operations)
    for key, expressions, expected in operations:
        concept = teacher.define(
            f"operation-{rng.getrandbits(32)}",
            "An exact binary quantity relationship",
            Procedure(
                inputs=(Endpoint(Role.DATA, number), Endpoint(Role.DATA, number)),
                result=number,
                fields={
                    "value": binary(key, field(argument(0), "value"), field(argument(1), "value"))
                },
            ),
            expressions=expressions,
            examples=(Example(samples, record(value=expected)),),
            roles=("first", "second"),
        )
        refs[key] = concept
        refs[f"op_{key}"] = as_ref(a.get(concept).fields["operation"])
    ancestor = a.teach_operation(
        "category ancestry",
        Procedure(
            inputs=(Endpoint(Role.CATEGORY), Endpoint(Role.CATEGORY)),
            result=boolean,
            fields={
                "value": choose(
                    binary("eq", argument(0), argument(1)),
                    literal(True),
                    fold(
                        field(argument(0), "parents"),
                        literal(False),
                        choose(
                            variable("acc"),
                            literal(True),
                            field(call("self", variable("item"), argument(1)), "value"),
                        ),
                    ),
                )
            },
        ),
    )
    membership = teacher.define(
        "category membership",
        "Whether the observed participant has this category or a specialization",
        Procedure(
            inputs=(Endpoint(Role.DATA), Endpoint(Role.CATEGORY)),
            result=boolean,
            fields={
                "value": choose(
                    binary("eq", category(argument(0)), literal(None)),
                    literal(False),
                    field(call(ancestor, category(argument(0)), argument(1)), "value"),
                )
            },
        ),
        expressions=("is", "include"),
        examples=(
            Example((robin, animal), record(value=True)),
            Example((stone, bird), record(value=False)),
        ),
        roles=("participant", "category"),
    )
    refs["member"] = membership
    refs["op_member"] = as_ref(a.get(membership).fields["operation"])
    # Vary installed compositions so selection cannot assume a closed opcode inventory.
    expression = field(call(refs["op_sub"], argument(0), argument(1)), "value")
    variant = rng.randrange(3)
    if variant == 0:
        expression = choose(
            binary("lt", expression, literal(0)), binary("sub", literal(0), expression), expression
        )
        expected_extra = 9
    elif variant == 1:
        expression = binary("mul", expression, literal(2))
        expected_extra = 18
    else:
        expression = field(call(refs["op_add"], argument(1), argument(0)), "value")
        expected_extra = 15
    refs["extra"] = teacher.define(
        "installed composition",
        "A procedure built from other exact procedures",
        Procedure((Endpoint(Role.DATA, number),) * 2, number, {"value": expression}),
        expressions=("combine",),
        examples=(Example(samples, record(value=expected_extra)),),
    )
    refs["op_extra"] = as_ref(a.get(refs["extra"]).fields["operation"])
    a.create(
        fields={"kind": "language_anchor", "quantity": number},
        source="explicit language domain interface",
    )
    refs["self"] = a.create(
        fields={"kind": "observer", "width": 64}, source="semantic observer initialization"
    )
    return a, refs


@dataclass(slots=True)
class SemanticCase:
    knowledge: Athanor
    refs: dict[str, Ref]
    utterance: Utterance
    target: Proposal
    meaning: Ref | None
    expected: Fraction | bool | None
    text: str
    family: str


def case(seed: int, *, mode: str = "valid") -> SemanticCase:
    rng = random.Random(seed ^ 0xBAC)
    a, refs = install(seed)
    teacher = Teacher(a)
    language = Language(a, refs["number"])
    x, y = rng.randint(2, 30), rng.randint(1, 15)
    key = rng.choice(("add", "sub", "div", "member", "extra"))
    reverse = rng.choice((False, True))
    context = record()
    previous_result = None
    alias = rng.choice(("dax", "mep", "zor", "tiv", "nuf", "wug", "pim", "kav"))
    args: tuple[Ref, ...]
    if key == "member":
        subject = rng.choice(["robin", "turtle", "specimen"])
        target_category = rng.choice(["animal", "bird", "subtype"])
        text = (
            f"does {target_category} include {subject}"
            if reverse
            else f"is {subject} a {target_category}"
        )
        args = (refs[subject], refs[target_category])
        participant_category = a.get(args[0]).category
        assert participant_category is not None
        expected: Fraction | bool | None = args[1] in (
            participant_category,
            *a.ancestors(participant_category),
        )
    else:
        texts = {
            "add": (f"add {x} and {y}", f"the sum of {x} and {y}"),
            "sub": (f"the difference of {x} and {y}", f"subtract {y} from {x}"),
            "div": (f"divide {x} by {y}", f"the quotient of {x} and {y}"),
            "extra": (f"combine {x} and {y}", f"combine {x} and {y}"),
        }
        text = texts[key][int(reverse)]
        args = ()
        expected = {
            "add": Fraction(x + y),
            "sub": Fraction(x - y),
            "div": Fraction(x, y),
            "extra": None,
        }[key]
    if mode == "alias":
        # The word carries no stable operation identity across stores.
        teacher.alias(alias, refs[key])
        text = f"{alias} {x} and {y}" if key != "member" else f"{alias} {subject} {target_category}"
        reverse = False
    elif mode in {"ambiguous", "context"}:
        teacher.alias("balance", refs["sub"], context=record(domain="difference"))
        teacher.alias("balance", refs["div"], context=record(domain="ratio"))
        key = rng.choice(("sub", "div"))
        text = f"balance {x} and {y}"
        if mode == "context":
            context = record(domain="difference" if key == "sub" else "ratio")
        expected = Fraction(x - y) if key == "sub" else Fraction(x, y)
        reverse = False
    elif mode == "unknown":
        text = f"uninstalled {x} and {y}"
    elif mode == "missing":
        key, text, expected = "sub", f"subtract {x}", None
    elif mode == "correction":
        old = teacher.alias(alias, refs["add"] if key != "add" else refs["sub"])
        teacher.correct(
            alias, (old,), refs[key], reason="Retain this explicitly corrected convention"
        )
        text = f"{alias} {x} and {y}" if key != "member" else f"{alias} {subject} {target_category}"
        reverse = False
    elif mode == "pronoun":
        key, reverse = "add", False
        previous_result = a.execute(
            refs["op_add"],
            (a.create(refs["number"], {"value": x}), a.create(refs["number"], {"value": 0})),
        )
        text, expected = f"add it and {y}", Fraction(x + y)
    distractors = tuple(
        a.create(refs["number"], {"value": n}) for n in (rng.randint(50, 80), rng.randint(81, 99))
    )
    utterance = language.encode(
        text, context=context, distractors=distractors, previous_result=previous_result
    )
    pool = list(utterance.candidates)
    rng.shuffle(pool)
    utterance = Utterance(
        utterance.ref,
        tuple(pool),
        utterance.bindings,
        utterance.mentions,
        utterance.unknown,
        utterance.context,
    )
    if key != "member":
        args = (
            tuple(reversed(utterance.mentions))
            if key == "sub" and reverse and mode not in {"ambiguous", "context", "alias"}
            else utterance.mentions
        )
    target = Proposal(refs[f"op_{key}"], args)
    if key == "extra" and mode not in {"ambiguous", "unknown", "missing"}:
        result_ref = a.execute(refs["op_extra"], target.arguments)
        raw_expected = a.get(result_ref).fields["value"]
        assert isinstance(raw_expected, Fraction)
        expected = raw_expected
    meaning: Ref | None = refs[key]
    if mode in {"ambiguous", "unknown", "missing"}:
        target, meaning, expected = Proposal(None), None, None
    return SemanticCase(
        a,
        refs,
        utterance,
        target,
        meaning,
        expected,
        text,
        "categorical" if key == "member" else "mathematical",
    )


def lessons(count: int, config: ModelConfig, *, seed: int = 3000) -> list[Lesson]:
    result = []
    modes = (
        "pronoun",
        "valid",
        "valid",
        "correction",
        "alias",
        "context",
        "unknown",
        "missing",
        "ambiguous",
        "valid",
    )
    for index in range(count):
        example = case(seed + index, mode=modes[index % len(modes)])
        observation = Observation(example.utterance.ref, example.utterance.candidates)
        frame = GraphBuilder(config).build(
            example.knowledge, observation, observer=example.refs["self"]
        )
        probe = example.target
        if index % 4 == 0:
            operation = frame.operations[index % len(frame.operations)]
            arity = frame.arities[frame.operations.index(operation)]
            probe = Proposal(operation, example.utterance.candidates[:arity])

        def satisfied(ref: Ref, current: SemanticCase = example) -> bool:
            return (
                current.expected is not None
                and current.knowledge.get(ref).fields["value"] == current.expected
            )

        feedback = measure(
            example.knowledge,
            Decision(probe, 0.5, Expectation(0.5, 0.5, (0.5,) * 6)),
            evaluate=satisfied,
        )
        # Fresh arrival and a DIFFERENT preceding observation with real feedback.
        result.append(Lesson(frame, example.target, probe, feedback, None, example.meaning))
        language = Language(example.knowledge, example.refs["number"])
        episode_rng = random.Random((seed + index) ^ 0xF193)
        history: list[Ref] = []
        history_length = episode_rng.randint(1, 7)
        for step in range(history_length):
            choices = (
                ("add", "sub", "div", "extra")
                if modes[index % len(modes)] == "pronoun" and step == history_length - 1
                else ("add", "sub", "div", "member", "extra", "unknown")
            )
            previous_key = episode_rng.choice(choices)
            p, q = episode_rng.randint(2, 40), episode_rng.randint(1, 15)
            text = {
                "add": f"add {p} and {q}",
                "sub": f"difference {p} and {q}",
                "div": f"divide {p} by {q}",
                "member": "is turtle a bird",
                "extra": f"combine {p} and {q}",
                "unknown": f"uninstalled {p} and {q}",
            }[previous_key]
            previous = language.encode(text)
            prior_observation = Observation(previous.ref, previous.candidates)
            prior = GraphBuilder(config).build(
                example.knowledge,
                prior_observation,
                observer=example.refs["self"],
                self_refs=tuple(history),
            )
            previous_decision = Decision(
                Proposal(None)
                if previous_key == "unknown"
                else Proposal(example.refs[f"op_{previous_key}"], previous.mentions),
                0.5,
                Expectation(0.5, 0.5, (0.5,) * 6),
            )
            previous_feedback = measure(
                example.knowledge, previous_decision, evaluate=lambda _: True
            )
            history.append(
                record_self_observation(
                    example.knowledge, example.refs["self"], prior_observation, previous_feedback
                )
            )
            history.append(
                example.knowledge.create(
                    fields={
                        "kind": "semantic_capability_use",
                        "concept": example.refs.get(previous_key),
                        "recognized": previous_key != "unknown",
                        "operation": previous_decision.proposal.operation,
                        "execution_success": previous_feedback.succeeded,
                        "confidence": Fraction(1, 2),
                        "observation": previous.ref,
                    },
                    source="measured semantic observer performance",
                )
            )
        active_target, active_probe, active_feedback = example.target, probe, feedback
        if modes[index % len(modes)] == "pronoun":
            # The grounded pronoun must denote the actual most recent result,
            # rather than an unrelated old result created before this history.
            assert previous_feedback.result is not None
            y = example.knowledge.get(example.utterance.mentions[-1]).fields["value"]
            last = example.knowledge.get(previous_feedback.result).fields["value"]
            assert isinstance(y, Fraction) and isinstance(last, Fraction)
            active_utterance = language.encode(
                f"add it and {y}",
                previous_result=previous_feedback.result,
                distractors=tuple(
                    ref
                    for ref in example.utterance.candidates
                    if ref not in example.utterance.mentions
                ),
            )
            candidates = list(active_utterance.candidates)
            episode_rng.shuffle(candidates)
            observation = Observation(active_utterance.ref, tuple(candidates))
            active_target = Proposal(example.refs["op_add"], active_utterance.mentions)
            active_probe = active_target

            def active_satisfied(
                ref: Ref, current: Athanor = example.knowledge, expected: Fraction = last + y
            ) -> bool:
                return current.get(ref).fields["value"] == expected

            active_feedback = measure(
                example.knowledge,
                Decision(active_probe, 0.5, Expectation(0.5, 0.5, (0.5,) * 6)),
                evaluate=active_satisfied,
            )
        active_frame = GraphBuilder(config).build(
            example.knowledge, observation, observer=example.refs["self"], self_refs=tuple(history)
        )
        result.append(
            Lesson(
                active_frame, active_target, active_probe, active_feedback, prior, example.meaning
            )
        )
    return result
