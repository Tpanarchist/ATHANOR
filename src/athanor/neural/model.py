"""One shared recurrent process for structural refinement and executable pointers."""

from collections.abc import Sequence
from math import sqrt
from typing import cast

import torch
from torch import Tensor, nn

from athanor import ValidationError

from .contracts import OUTCOME_SIZE, Activation, ModelConfig, ModelOutput, Proposal
from .encoding import EDGE_SIZE, FEATURE_SIZE, Frame


class NeuralObserverModel(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        config = ModelConfig() if config is None else config
        self.config = config
        w = config.width
        # Initialization has an owned seed and restores the caller's RNG afterwards.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.encoder = nn.Linear(FEATURE_SIZE, w)
            self.edge_encoder = nn.Linear(EDGE_SIZE, w, bias=False)
            self.message = nn.Linear(3 * w, w)
            self.message_gate = nn.Linear(3 * w, 1)
            self.recurrent = nn.GRUCell(3 * w, w)
            self.phase = nn.Linear(3, w)
            self.scorer = nn.Sequential(nn.Linear(3 * w, w), nn.Tanh(), nn.Linear(w, 1))
            self.applicability = nn.Sequential(nn.Linear(3 * w, w), nn.Tanh(), nn.Linear(w, 1))
            self.capability = nn.Linear(2 * w, 1)
            self.preference = nn.Linear(2 * w, 1)
            self.prediction = nn.Linear(2 * w, OUTCOME_SIZE)
            self.null = nn.Parameter(torch.randn(w) / sqrt(w))
            self.carry = nn.Parameter(torch.tensor(-2.0))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _scores(
        self, candidates: Tensor, context: Tensor, *, applicability: bool = False
    ) -> Tensor:
        repeated = context.expand(candidates.shape[0], -1)
        joined = torch.cat((candidates, repeated, candidates * repeated), dim=-1)
        head = self.applicability if applicability else self.scorer
        return cast(Tensor, head(joined)).squeeze(-1)

    def forward(
        self,
        frames: Sequence[Frame],
        states: Sequence[Activation | None] | None = None,
        forced: Sequence[Proposal | None] | None = None,
        *,
        steps: int | None = None,
        sensory: Sequence[Tensor | None] | None = None,
    ) -> tuple[ModelOutput, ...]:
        if not frames:
            raise ValidationError("Neural computation requires at least one graph")
        rounds = self.config.refinement_steps if steps is None else steps
        if type(rounds) is not int or rounds < 1:
            raise ValidationError("Refinement requires a positive number of internal steps")
        previous = [None] * len(frames) if states is None else list(states)
        teachers = [None] * len(frames) if forced is None else list(forced)
        if len(previous) != len(frames) or len(teachers) != len(frames):
            raise ValidationError("Graph/state/forced-proposal batch sizes differ")
        device = self.device
        w = self.config.width
        offsets: list[int] = []
        size = 0
        for frame in frames:
            offsets.append(size)
            size += len(frame.refs)
        features = torch.cat([frame.features for frame in frames]).to(device)
        edges = torch.cat(
            [frame.edge_index + offset for frame, offset in zip(frames, offsets, strict=True)],
            dim=1,
        ).to(device)
        edge_features = torch.cat([frame.edge_features for frame in frames]).to(device)
        groups = torch.cat(
            [torch.full((len(frame.refs),), i, dtype=torch.long) for i, frame in enumerate(frames)]
        ).to(device)
        queries = torch.tensor(
            [offset + frame.query_index for offset, frame in zip(offsets, frames, strict=True)],
            device=device,
        )
        if sensory is not None:
            # Use the declared compact observation interface at the sensory query.
            # Full receipts/track graphs remain available for relational messages;
            # their varying metadata must not alter the encoding of current intent.
            features = features.clone()
            for query_index, frame in zip(queries, frames, strict=True):
                if frame.intent_features is not None:
                    if frame.intent_features.shape != (FEATURE_SIZE,) or not bool(
                        torch.isfinite(frame.intent_features).all()
                    ):
                        raise ValidationError("Compact sensory intent features are invalid")
                    features[query_index] = frame.intent_features.to(device)
        prior = torch.zeros((size, w), device=device)
        contexts: list[Tensor] = []
        for frame, state, offset in zip(frames, previous, offsets, strict=True):
            if state is None:
                contexts.append(torch.zeros(w, device=device))
                continue
            if state.field.shape != (len(state.refs), w) or state.context.shape != (w,):
                raise ValidationError("Persistent activation dimensions disagree with the model")
            if (
                len(set(state.refs)) != len(state.refs)
                or state.steps < 0
                or not bool(torch.isfinite(state.field).all())
                or not bool(torch.isfinite(state.context).all())
            ):
                raise ValidationError("Persistent activation handles/values are invalid")
            positions = {ref: index for index, ref in enumerate(state.refs)}
            for index, ref in enumerate(frame.refs):
                if ref in positions:
                    prior[offset + index] = state.field[positions[ref]].to(device)
            contexts.append(state.context.to(device))
        context = torch.stack(contexts)
        base = torch.tanh(cast(Tensor, self.encoder(features)))
        if sensory is not None:
            if len(sensory) != len(frames):
                raise ValidationError("Sensory/structural batch sizes differ")
            additions = torch.zeros_like(base)
            for query_index, embedding in zip(queries, sensory, strict=True):
                if embedding is not None:
                    if embedding.shape != (w,) or not bool(torch.isfinite(embedding).all()):
                        raise ValidationError("Sensory activation must match shared field width")
                    additions[query_index] = embedding.to(device)
            base = base + additions
        h = base + torch.sigmoid(self.carry) * prior
        edge_encoding = torch.tanh(cast(Tensor, self.edge_encoder(edge_features)))
        sources, targets = edges[0], edges[1]
        for _ in range(rounds):
            context = cast(
                Tensor,
                self.recurrent(
                    torch.cat((h[queries], context, torch.zeros_like(context)), -1), context
                ),
            )
            source_context = context[groups[sources]]
            messages = torch.tanh(
                cast(
                    Tensor, self.message(torch.cat((h[sources], edge_encoding, source_context), -1))
                )
            )
            gate = torch.sigmoid(
                cast(
                    Tensor, self.message_gate(torch.cat((messages, h[targets], source_context), -1))
                )
            )
            aggregated = torch.zeros_like(h).index_add(0, targets, messages * gate)
            normalization = torch.zeros((size, 1), device=device).index_add(0, targets, gate)
            aggregated = aggregated / normalization.clamp_min(1)
            h = cast(Tensor, self.recurrent(torch.cat((base, aggregated, context[groups]), -1), h))
        outputs: list[ModelOutput] = []
        for group, (frame, teacher, state, offset) in enumerate(
            zip(frames, teachers, previous, offsets, strict=True)
        ):
            field = h[offset : offset + len(frame.refs)]
            # Preserve direct relational context at the pointer boundary. Repeated
            # graph aggregation must not erase sequence positions before binding.
            # Every outgoing query edge contributes; no task/slot path is privileged.
            direct = sources == offset + frame.query_index
            relations = torch.zeros_like(h).index_add(0, targets[direct], edge_encoding[direct])[
                offset : offset + len(frame.refs)
            ]
            pointer_field = field + relations
            query = field[frame.query_index]
            if sensory is not None:
                # Current sensor/intention input remains directly available at the
                # decoding boundary. Relational refinement and self-history must
                # contextualize an observation without erasing its current content.
                query = query + base[offset + frame.query_index]
            phase = torch.tensor([1.0, 0.0, 1.0], device=device)
            decoder = cast(
                Tensor,
                self.recurrent(
                    torch.cat((query, context[group], cast(Tensor, self.phase(phase))), -1),
                    context[group],
                ),
            )
            operations = (
                field[list(frame.operation_indices)] if frame.operation_indices else field[:0]
            )
            candidates = torch.cat((operations, self.null.unsqueeze(0)), 0)
            operation_logits = self._scores(candidates, decoder)
            score_trace = [(candidates, decoder)]
            if frame.meaning_indices:
                score_trace.append(
                    (
                        torch.cat((field[list(frame.meaning_indices)], self.null.unsqueeze(0)), 0),
                        decoder,
                    )
                )
            meaning_logits = (
                self._scores(
                    torch.cat((field[list(frame.meaning_indices)], self.null.unsqueeze(0)), 0),
                    decoder,
                )
                if frame.meaning_indices
                else None
            )
            operation_index = int(operation_logits.argmax().item())
            if teacher is not None:
                operation_index = (
                    len(frame.operations)
                    if teacher.operation is None
                    else frame.operations.index(teacher.operation)
                )
            selected = candidates[operation_index]
            applicability = self._scores(operations, decoder, applicability=True)
            argument_logits: list[Tensor] = []
            argument_indices: list[int] = []
            is_abstention = operation_index == len(frame.operations)
            arity = 0 if is_abstention else frame.arities[operation_index]
            if teacher is not None and len(teacher.arguments) != arity:
                raise ValidationError("Forced training proposal disagrees with declared arity")
            for slot in range(arity):
                if not frame.candidate_indices:
                    # Missing offered inputs has an explicit null candidate. No silent repair.
                    pool = self.null.unsqueeze(0)
                else:
                    pool = pointer_field[list(frame.candidate_indices)]
                phase = torch.tensor([0.0, slot / (slot + 1), 1 / (slot + 1)], device=device)
                decoder = cast(
                    Tensor,
                    self.recurrent(
                        torch.cat((selected, query, cast(Tensor, self.phase(phase))), -1), decoder
                    ),
                )
                logits = self._scores(pool, decoder)
                score_trace.append((pool, decoder))
                index = int(logits.argmax().item())
                if teacher is not None:
                    index = frame.candidates.index(teacher.arguments[slot])
                argument_logits.append(logits)
                argument_indices.append(index)
                selected = pool[index]
            summary = torch.cat((decoder, selected), -1)
            outputs.append(
                ModelOutput(
                    operation_logits,
                    tuple(argument_logits),
                    applicability,
                    cast(Tensor, self.capability(summary)).squeeze(-1),
                    cast(Tensor, self.preference(summary)).squeeze(-1),
                    torch.sigmoid(cast(Tensor, self.prediction(summary))),
                    operation_index,
                    tuple(argument_indices),
                    Activation(
                        frame.refs,
                        field,
                        context[group],
                        rounds + (0 if state is None else state.steps),
                    ),
                    meaning_logits,
                    tuple(score_trace),
                )
            )
        return tuple(outputs)
