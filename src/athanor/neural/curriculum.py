"""Structured teaching is separate from learning to use the installed procedures."""

import random
from dataclasses import dataclass
from fractions import Fraction

from athanor import Athanor, Endpoint, Evidence, FieldSpec, Procedure, Ref, Role
from athanor.programs import (
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

from .contracts import Decision, Expectation, Feedback, ModelConfig, Observation, Proposal
from .encoding import Frame, GraphBuilder
from .observer import measure, record_self_observation


@dataclass(slots=True)
class Situation:
    knowledge: Athanor
    observation: Observation
    target: Proposal
    expected_value: Fraction | None
    self_ref: Ref
    self_refs: tuple[Ref, ...] = ()

    def frame(self, config: ModelConfig) -> Frame:
        return GraphBuilder(config).build(
            self.knowledge, self.observation, self_refs=self.self_refs, observer=self.self_ref
        )


@dataclass(frozen=True, slots=True)
class Lesson:
    frame: Frame
    target: Proposal
    probe: Proposal
    feedback: Feedback
    prior: Frame | None = None
    meaning: Ref | None = None


def install(seed: int, *, novel: bool = False) -> tuple[Athanor, dict[str, Ref]]:
    """Names, allocation order and namespaces vary; they are not neural identifiers."""
    rng = random.Random(seed)
    a = Athanor(namespace=f"curriculum-{seed}")
    for _ in range(rng.randrange(4)):
        a.create(fields={"label": str(rng.getrandbits(32))})
    measurable = a.teach_category(f"M-{seed}")
    number = a.teach_category(f"N-{seed}", {"value": FieldSpec("number")}, parents=(measurable,))
    specialized = a.teach_category(f"S-{seed}", parents=(number,))
    bundle = a.teach_category(f"B-{seed}", parents=(measurable,))
    text = a.teach_category(f"T-{seed}", {"text": FieldSpec("string")})
    definitions = {
        key: Procedure(
            inputs=(Endpoint(Role.DATA, number), Endpoint(Role.DATA, number)),
            result=number,
            fields={"value": binary(key, field(argument(0), "value"), field(argument(1), "value"))},
        )
        for key in ("add", "sub", "div")
    }
    definitions["total"] = Procedure(
        inputs=(Endpoint(Role.DATA, measurable),),
        result=number,
        fields={
            "value": choose(
                has_field(argument(0), "value"),
                field(argument(0), "value"),
                fold(
                    members(argument(0)),
                    literal(0),
                    binary("add", variable("acc"), field(call("self", variable("item")), "value")),
                ),
            )
        },
    )
    if novel:
        definitions["mul"] = Procedure(
            inputs=(Endpoint(Role.DATA, number), Endpoint(Role.DATA, number)),
            result=number,
            fields={
                "value": binary("mul", field(argument(0), "value"), field(argument(1), "value"))
            },
        )
    order = list(definitions)
    rng.shuffle(order)
    refs = {"number": number, "specialized": specialized, "bundle": bundle, "text": text}
    refs.update(
        {key: a.teach_operation(f"P-{rng.getrandbits(32)}", definitions[key]) for key in order}
    )
    relation = a.teach_relation("implements intent", Endpoint(Role.DATA), Endpoint(Role.OPERATION))
    refs["relation"] = relation
    for key in definitions:
        intent = a.create(fields={"kind": "intent", "arity": len(definitions[key].inputs)})
        a.bind(relation, intent, refs[key], Evidence("structured instructor"))
        refs[f"intent_{key}"] = intent
    refs["self"] = a.create(fields={"kind": "observer", "width": 64})
    return a, refs


def situation(seed: int, *, mode: str = "valid", novel: bool = False) -> Situation:
    rng = random.Random(seed ^ 0xAF71)
    a, refs = install(seed, novel=novel)
    keys = ["add", "sub", "div", "total"] + (["mul"] if novel else [])
    key = rng.choice(keys)
    values = [Fraction(rng.randint(-20, 20), rng.randint(1, 5)) for _ in range(4)]
    if values[1] == 0:
        values[1] = Fraction(2)
    numbers = [
        a.create(refs[rng.choice(["number", "specialized"])], {"value": value}) for value in values
    ]
    inner = a.compose(tuple(numbers[:2]), refs["bundle"])
    outer = a.compose((inner, numbers[2]), refs["bundle"])
    distractor = a.create(refs["text"], {"text": "incompatible input"})
    arguments = (outer,) if key == "total" else tuple(numbers[:2])
    expected: Fraction | None = {
        "add": values[0] + values[1],
        "sub": values[0] - values[1],
        "div": values[0] / values[1],
        "total": sum(values[:3], Fraction(0)),
        "mul": values[0] * values[1],
    }[key]
    intent = refs[f"intent_{key}"]
    slots: tuple[Ref | None, ...] = arguments
    target = Proposal(refs[key], arguments)
    if mode == "unsupported":
        intent = a.create(fields={"kind": "intent", "arity": len(arguments)})
        target, expected = Proposal(None), None
    elif mode == "missing":
        slots = (*arguments[:-1], None)
        target, expected = Proposal(None), None
    elif mode == "incompatible":
        slots = (distractor, *arguments[1:])
        target, expected = Proposal(None), None
    elif mode == "correction":
        # A real wrong-but-executable attempt followed by explicit teacher correction.
        # The task's older relation is deliberately insufficient after preference changes.
        key = rng.choice(["add", "sub"])
        intent = refs[f"intent_{key}"]
        arguments = tuple(numbers[:2])
        slots = arguments
        corrected = "sub" if key == "add" else "add"
        target = Proposal(refs[corrected], arguments)
        expected = values[0] - values[1] if corrected == "sub" else values[0] + values[1]
    candidates = [*numbers, inner, outer, distractor]
    rng.shuffle(candidates)
    observation = Observation(
        a.create(fields={"kind": "request", "task": intent, "slots": slots}), tuple(candidates)
    )
    self_refs: tuple[Ref, ...] = ()
    if mode == "correction":
        decision = Decision(Proposal(refs[key], arguments), 0.8, Expectation(0.8, 0.8, (0.5,) * 6))
        feedback = measure(a, decision, evaluate=lambda _: False, correction=target)
        self_refs = (
            record_self_observation(
                a, refs["self"], observation, feedback, attempts=1, successes=1
            ),
        )
    return Situation(a, observation, target, expected, refs["self"], self_refs)


def lessons(count: int, config: ModelConfig, *, seed: int = 1000) -> list[Lesson]:
    result: list[Lesson] = []
    for index in range(count):
        mode = (
            "valid",
            "valid",
            "valid",
            "valid",
            "unsupported",
            "missing",
            "incompatible",
            "correction",
        )[index % 8]
        case = situation(seed + index, mode=mode)
        frame = case.frame(config)
        probe = case.target
        # Execution feedback covers successful, rejected and wrong-but-valid proposals.
        if index % 3 == 0:
            operation = frame.operations[index % len(frame.operations)]
            arity = frame.arities[frame.operations.index(operation)]
            probe = Proposal(operation, case.observation.candidates[:arity])
        decision = Decision(probe, 0.5, Expectation(0.5, 0.5, (0.5,) * 6))

        def satisfied(ref: Ref, current: Situation = case) -> bool:
            return (
                current.expected_value is not None
                and current.knowledge.get(ref).fields.get("value") == current.expected_value
            )

        feedback = measure(case.knowledge, decision, evaluate=satisfied)
        prior = (
            GraphBuilder(config).build(case.knowledge, case.observation, observer=case.self_ref)
            if mode == "correction"
            else frame
        )
        # Learn both initial processing and persistence across further internal steps.
        result.append(Lesson(frame, case.target, probe, feedback, prior if index % 2 else None))
    return result
