"""Bounded lexical observation: tokens and textual mentions, never executable dispatch."""

import re
import unicodedata
from dataclasses import dataclass
from fractions import Fraction

from athanor import Assessment, Athanor, Record, Ref, Role, ValidationError, record
from athanor.values import as_ref

from .lexicon import Binding, Lexicon

WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "please",
        "of",
        "and",
        "by",
        "from",
        "to",
        "with",
        "does",
        "is",
        "then",
        "it",
        "what",
        "between",
        "compute",
        "calculate",
        "take",
        "in",
        "kind",
        "whether",
        "find",
    }
)
_TOKEN = re.compile(r"[+-]?\d+(?:/\d+|\.\d+)?|[^\W\d_]+(?:-[^\W\d_]+)*|[?.,;:]", re.UNICODE)


@dataclass(frozen=True, slots=True)
class Utterance:
    ref: Ref
    candidates: tuple[Ref, ...]
    bindings: tuple[Binding, ...]
    mentions: tuple[Ref, ...]
    unknown: tuple[str, ...]
    context: Record


def tokenize(text: str) -> tuple[str, ...]:
    if type(text) is not str or not text.strip() or len(text) > 1024:
        raise ValidationError("Language input requires 1..1024 characters")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    result: list[str] = []
    previous = 0
    for match in _TOKEN.finditer(normalized):
        if normalized[previous : match.start()].strip():
            raise ValidationError("Unrecognized characters in bounded language")
        token = match.group()
        if len(token) > 64:
            raise ValidationError("Language tokens exceed the explicit character budget")
        result.append(token)
        previous = match.end()
    if normalized[previous:].strip() or not result or len(result) > 64:
        raise ValidationError("Unrecognized or excessive bounded language tokens")
    return tuple(result)


class Language:
    def __init__(self, knowledge: Athanor, quantity: Ref) -> None:
        self.knowledge = knowledge
        self.quantity = quantity
        knowledge.get(quantity)
        self.lexicon = Lexicon(knowledge)

    def encode(
        self,
        text: str,
        *,
        context: Record | None = None,
        previous_result: Ref | None = None,
        distractors: tuple[Ref, ...] = (),
    ) -> Utterance:
        tokens = tokenize(text)
        if any(token in {";", ":"} for token in tokens):
            raise ValidationError("Encode one clause at a time; use the sequence interface")
        scope = record() if context is None else context

        def install(a: Athanor) -> Utterance:
            lexicon = Lexicon(a)
            bindings: dict[Ref, Binding] = {}
            mentions: list[Ref] = []
            token_refs: list[Ref] = []
            unknown: list[str] = []
            offered: list[Ref] = list(distractors)
            # Longest installed lexical span is an exact convention lookup, never
            # a phrase-to-operation parser. All competing bindings of it survive.
            spans: list[tuple[int, str]] = []
            position = 0
            while position < len(tokens):
                length = 1
                if tokens[position].isalpha():
                    for end in range(position + 1, len(tokens) + 1):
                        words = tokens[position:end]
                        if not all(word.replace("-", "").isalpha() for word in words):
                            break
                        phrase = " ".join(words)
                        if len(phrase) > 256:
                            break
                        if lexicon.bindings(phrase, scope):
                            length = end - position
                spans.append((position, " ".join(tokens[position : position + length])))
                position += length
            for position, token in spans:
                if token in {"?", ".", ","}:
                    continue
                linked: list[Ref] = []
                if re.fullmatch(r"[+-]?\d+(?:/\d+|\.\d+)?", token):
                    try:
                        value = Fraction(token)
                    except (ValueError, ZeroDivisionError) as exc:
                        raise ValidationError("Invalid exact literal") from exc
                    participant = a.create(
                        self.quantity, {"value": value}, source="exact lexical numeric observation"
                    )
                    linked.append(participant)
                    mentions.append(participant)
                    offered.append(participant)
                    symbol = None
                else:
                    symbol = lexicon.symbol(token)
                    available = lexicon.bindings(token, scope)
                    bindings.update({binding.ref: binding for binding in available})
                    for binding in available:
                        referent = a.get(binding.referent)
                        if (
                            binding.assessment == Assessment.SUPPORTED
                            and referent.fields.get("kind") != "concept"
                            and referent.role
                            in {
                                Role.DATA,
                                Role.CATEGORY,
                            }
                        ):
                            linked.append(binding.referent)
                            offered.append(binding.referent)
                    if token == "it" and previous_result is not None:
                        a.get(previous_result)
                        linked.append(previous_result)
                        offered.append(previous_result)
                    if linked:
                        # A single grounded mention is positional context, not an argument role.
                        # Competing referents remain on the token and are separately rejected.
                        mentions.extend(dict.fromkeys(linked))
                    elif not available and (token not in WORDS or token == "it"):
                        unknown.append(token)
                token_refs.append(
                    a.create(
                        fields={
                            "kind": "language_token",
                            "text": token,
                            "position": position,
                            "symbol": symbol,
                            "referents": tuple(dict.fromkeys(linked)),
                        },
                        source="bounded lexical observation",
                    )
                )
            for ref in distractors:
                a.get(ref)
            corrections = tuple(
                node.ref
                for node in a.structures
                if node.fields.get("kind") == "meaning_correction"
                and any(
                    node.fields.get("symbol") == binding.symbol for binding in bindings.values()
                )
            )
            ref = a.create(
                fields={
                    "kind": "utterance",
                    "context": scope,
                    "mentions": tuple(mentions),
                    "bindings": tuple(bindings),
                    # Possible referents are explicit observation interfaces. Keep
                    # every alternative; no operation is selected or copied here.
                    "referents": tuple(
                        dict.fromkeys(binding.referent for binding in bindings.values())
                    ),
                    "corrections": corrections,
                    "unknown": tuple(unknown),
                },
                members=tuple(token_refs),
                source="bounded language observation",
            )
            return Utterance(
                ref,
                tuple(dict.fromkeys(offered)),
                tuple(bindings.values()),
                tuple(mentions),
                tuple(unknown),
                scope,
            )

        return self.knowledge.transaction(install)

    def reconstruct(self, ref: Ref, candidates: tuple[Ref, ...]) -> Utterance:
        node = self.knowledge.get(ref)
        if node.fields.get("kind") != "utterance":
            raise ValidationError("Current observation is not a language utterance")
        from athanor.values import as_record, as_string, as_tuple

        scope = as_record(node.fields["context"])
        bindings = []
        for raw in as_tuple(node.fields["bindings"]):
            binding_ref = as_ref(raw)
            stored = self.knowledge.get(binding_ref)
            symbol = as_ref(stored.fields["symbol"])
            expression = as_string(self.knowledge.get(symbol).fields["expression"])
            if binding_ref not in {
                binding.ref for binding in self.lexicon.bindings(expression, scope)
            }:
                continue
            bindings.append(
                Binding(
                    binding_ref,
                    symbol,
                    as_ref(stored.fields["referent"]),
                    as_record(stored.fields["context"]),
                    self.knowledge.assess(symbol, self.lexicon.relation, binding_ref),
                )
            )
        return Utterance(
            ref,
            candidates,
            tuple(bindings),
            tuple(as_ref(value) for value in as_tuple(node.fields["mentions"])),
            tuple(as_string(value) for value in as_tuple(node.fields["unknown"])),
            scope,
        )


def meaning_evidence(knowledge: Athanor, utterance: Utterance) -> dict[Ref, Assessment]:
    grouped: dict[Ref, set[Assessment]] = {}
    for binding in utterance.bindings:
        if knowledge.get(binding.referent).fields.get("kind") == "concept":
            grouped.setdefault(binding.referent, set()).add(binding.assessment)
    result = {}
    for ref, evidence in grouped.items():
        if (
            Assessment.CONFLICT in evidence
            or {Assessment.SUPPORTED, Assessment.REFUTED} <= evidence
        ):
            result[ref] = Assessment.CONFLICT
        elif Assessment.SUPPORTED in evidence:
            result[ref] = Assessment.SUPPORTED
        elif Assessment.REFUTED in evidence:
            result[ref] = Assessment.REFUTED
        else:
            result[ref] = Assessment.UNKNOWN
    return result
