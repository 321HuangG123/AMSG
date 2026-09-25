"""AMSG model components for joint multimodal entity-relation extraction."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

from Blip.MODELS.blip import blip_feature_extractor
from model.ita_itm_gate import UnifiedAlignmentGate


class LinearDropConnect(nn.Linear):
    """Linear layer with the DropConnect behavior used by the model."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.dropout = dropout
        self._weight = self.weight

    def sample_mask(self) -> None:
        """Sample a weight mask for an individual training step."""

        if self.dropout == 0.0:
            self._weight = self.weight
            return

        mask = torch.empty_like(self.weight, dtype=torch.bool)
        mask.bernoulli_(self.dropout)
        self._weight = self.weight.masked_fill(mask, 0.0)

    def forward(self, inputs: torch.Tensor, sample_mask: bool = False) -> torch.Tensor:
        if self.training:
            if sample_mask:
                self.sample_mask()
            return F.linear(inputs, self._weight, self.bias)

        return F.linear(inputs, self.weight * (1.0 - self.dropout), self.bias)


class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention for sequence-first tensors."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        key_dim: int,
        value_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim

        self.query_projection = nn.Linear(input_size, num_heads * key_dim)
        self.key_projection = nn.Linear(input_size, num_heads * key_dim)
        self.value_projection = nn.Linear(input_size, num_heads * value_dim)
        self.output_projection = nn.Linear(num_heads * value_dim, output_size)

    def _scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = torch.matmul(query, key.transpose(-2, -1))
        logits = logits / math.sqrt(key.size(-1))

        if attention_mask is not None:
            expanded_mask = attention_mask.transpose(0, 1).unsqueeze(0).unsqueeze(-1)
            expanded_mask = expanded_mask.expand(
                self.num_heads,
                -1,
                -1,
                logits.size(-1),
            )
            logits = logits.masked_fill(expanded_mask == 0, float("-inf"))

        attention_weights = F.softmax(logits, dim=-1)
        attention_output = torch.matmul(attention_weights, value)
        return attention_output, attention_weights

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query_length, batch_size = query.size(0), query.size(1)

        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)

        query = query.view(query.size(0), batch_size, self.num_heads, self.key_dim)
        key = key.view(key.size(0), batch_size, self.num_heads, self.key_dim)
        value = value.view(value.size(0), batch_size, self.num_heads, self.value_dim)

        query = query.permute(2, 1, 0, 3)
        key = key.permute(2, 1, 0, 3)
        value = value.permute(2, 1, 0, 3)

        attention_output, attention_weights = self._scaled_dot_product_attention(
            query,
            key,
            value,
            attention_mask,
        )
        attention_output = attention_output.permute(2, 1, 0, 3).contiguous()
        attention_output = attention_output.view(
            query_length,
            batch_size,
            self.num_heads * self.value_dim,
        )
        return self.output_projection(attention_output), attention_weights


class MultiHeadTransformerLayer(nn.Module):
    """Attention, residual normalization, and feed-forward transformation."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_attention_heads: int,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(
            input_size,
            hidden_size,
            hidden_size,
            hidden_size,
            num_attention_heads,
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Dropout(dropout_rate),
        )
        self.attention_norm = nn.LayerNorm(input_size)
        self.output_norm = nn.LayerNorm(input_size)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        gate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attention_output, _ = self.attention(query, key_value, key_value)
        if gate is not None:
            attention_output = gate * attention_output

        attention_output = self.attention_norm(key_value + attention_output)
        feed_forward_output = self.feed_forward(attention_output)
        return self.output_norm(attention_output + feed_forward_output)


class EntityDecoder(nn.Module):
    """Decode span-level entity labels from the fused representations."""

    def __init__(self, args, ner2idx) -> None:
        super().__init__()
        self.hidden_size = args.hidden_size
        self.entity_label_count = len(ner2idx)

        self.global_projection = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.span_projection = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.label_projection = nn.Linear(self.hidden_size, self.entity_label_count)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(args.dropout)
        self.activation = nn.ELU()

    def forward(
        self,
        shared_features: torch.Tensor,
        entity_features: torch.Tensor,
        mask: torch.Tensor,
        token_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sequence_length, batch_size, _ = entity_features.size()

        global_feature = torch.cat([shared_features, entity_features], dim=-1)
        global_feature = torch.tanh(self.global_projection(global_feature))
        global_feature = torch.max(global_feature, dim=0).values
        global_feature = global_feature.unsqueeze(0).repeat(sequence_length, 1, 1)
        global_feature = global_feature.unsqueeze(0).repeat(sequence_length, 1, 1, 1)

        start_features = entity_features.unsqueeze(1).repeat(1, sequence_length, 1, 1)
        end_features = entity_features.unsqueeze(0).repeat(sequence_length, 1, 1, 1)
        span_features = torch.cat([start_features, end_features, global_feature], dim=-1)

        scores = self.span_projection(span_features)
        scores = self.activation(self.dropout(self.layer_norm(scores)))
        scores = torch.sigmoid(self.label_projection(scores))

        upper_triangular = torch.triu(
            torch.ones(sequence_length, sequence_length, device=entity_features.device)
        ).unsqueeze(-1)
        valid_tokens = mask.unsqueeze(1) * mask.unsqueeze(0)
        valid_spans = upper_triangular * valid_tokens

        if token_positions is not None:
            valid_spans = (
                valid_spans
                * token_positions.unsqueeze(1)
                * token_positions.unsqueeze(0)
            )

        return scores * valid_spans.unsqueeze(-1)


class RelationDecoder(nn.Module):
    """Decode relation labels from pairwise fused representations."""

    def __init__(self, args, rel2idx) -> None:
        super().__init__()
        self.hidden_size = args.hidden_size
        self.relation_label_count = len(rel2idx)

        self.global_projection = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.pair_projection = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.label_projection = nn.Linear(self.hidden_size, self.relation_label_count)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(args.dropout)
        self.activation = nn.ELU()

    def forward(
        self,
        shared_features: torch.Tensor,
        relation_features: torch.Tensor,
        mask: torch.Tensor,
        token_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sequence_length, _, _ = relation_features.size()

        global_feature = torch.cat([shared_features, relation_features], dim=-1)
        global_feature = torch.tanh(self.global_projection(global_feature))
        global_feature = torch.max(global_feature, dim=0).values
        global_feature = global_feature.unsqueeze(0).repeat(sequence_length, 1, 1)
        global_feature = global_feature.unsqueeze(0).repeat(sequence_length, 1, 1, 1)

        head_features = relation_features.unsqueeze(1).repeat(1, sequence_length, 1, 1)
        tail_features = relation_features.unsqueeze(0).repeat(sequence_length, 1, 1, 1)
        pair_features = torch.cat([head_features, tail_features, global_feature], dim=-1)

        scores = self.pair_projection(pair_features)
        scores = self.activation(self.dropout(self.layer_norm(scores)))
        scores = torch.sigmoid(self.label_projection(scores))

        valid_tokens = mask.unsqueeze(1) * mask.unsqueeze(0)
        if token_positions is not None:
            valid_tokens = (
                valid_tokens
                * token_positions.unsqueeze(1)
                * token_positions.unsqueeze(0)
            )

        return scores * valid_tokens.unsqueeze(-1)


class AMSG(nn.Module):
    """Alignment-Matching Gated fusion model for JMERE."""

    def __init__(self, args, input_size: int, ner2idx, rel2idx) -> None:
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size

        self.entity_decoder = EntityDecoder(args, ner2idx)
        self.relation_decoder = RelationDecoder(args, rel2idx)
        self.dropout = nn.Dropout(args.dropout)

        self.image_encoder = blip_feature_extractor(
            pretrained=args.blip_local_path,
            image_size=args.img_size,
            vit="base",
        )
        for parameter in self.image_encoder.parameters():
            parameter.requires_grad_(False)

        self.image_feature_projection = LinearDropConnect(
            input_size,
            self.hidden_size * 2,
            dropout=args.dropconnect,
        )
        self.text_feature_projection = LinearDropConnect(
            input_size,
            self.hidden_size * 3,
            dropout=args.dropconnect,
        )

        self.entity_self_attention = MultiHeadTransformerLayer(
            self.hidden_size,
            self.hidden_size,
            num_attention_heads=12,
        )
        self.relation_self_attention = MultiHeadTransformerLayer(
            self.hidden_size,
            self.hidden_size,
            num_attention_heads=12,
        )
        self.entity_cross_attention = MultiHeadTransformerLayer(
            self.hidden_size,
            self.hidden_size,
            num_attention_heads=12,
        )
        self.relation_cross_attention = MultiHeadTransformerLayer(
            self.hidden_size,
            self.hidden_size,
            num_attention_heads=12,
        )

        self.entity_lstm = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            num_layers=2,
            bidirectional=True,
        )
        self.relation_lstm = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            num_layers=2,
            bidirectional=True,
        )
        self.entity_projection = LinearDropConnect(
            self.hidden_size * 2,
            self.hidden_size,
            dropout=args.dropconnect,
        )
        self.relation_projection = LinearDropConnect(
            self.hidden_size * 2,
            self.hidden_size,
            dropout=args.dropconnect,
        )

        special_tokens = ["<s>", "</s>", "<o>", "</o>"]
        self.text_tokenizer = BertTokenizer.from_pretrained(
            args.bert_local_path,
            never_split=special_tokens,
        )
        self.text_encoder = BertModel.from_pretrained(args.bert_local_path)

        clip_path = getattr(args, "clip_local_path", "openai/clip-vit-base-patch32")
        self.alignment_gate = UnifiedAlignmentGate(
            pretrained=clip_path,
            d_align=512,
            tau=0.07,
            lambda_ita=0.1,
            lambda_itm=0.05,
            freeze_clip=True,
        )
        self.last_auxiliary_loss: Optional[torch.Tensor] = None

    def train(self, mode: bool = True):
        """Keep the frozen BLIP encoder in evaluation mode."""

        super().train(mode)
        self.image_encoder.eval()
        return self

    def forward(
        self,
        token_sequences: Sequence[Sequence[str]],
        images: torch.Tensor,
        sentences: Sequence[str],
        mask: torch.Tensor,
        token_positions: Optional[torch.Tensor] = None,
    ):
        device = images.device
        tokenized_text = self.text_tokenizer(
            token_sequences,
            return_tensors="pt",
            padding="longest",
            is_split_into_words=True,
        ).to(device)
        text_features = self.text_encoder(**tokenized_text).last_hidden_state

        image_features = self.image_encoder(
            images,
            sentences,
            mode="multimodal",
        )
        image_features = torch.tanh(self.image_feature_projection(image_features))
        image_feature_1, image_feature_2 = image_features.chunk(2, dim=-1)
        image_feature_1 = torch.max(image_feature_1, dim=1).values
        image_feature_2 = torch.max(image_feature_2, dim=1).values

        text_features = text_features.transpose(0, 1)
        if self.training:
            text_features = self.dropout(text_features)

        text_features = self.text_feature_projection(text_features)
        text_chunk_0, text_chunk_1, text_chunk_2 = text_features.chunk(3, dim=-1)

        initial_hidden = image_feature_1.unsqueeze(0).repeat(4, 1, 1)
        entity_features, _ = self.entity_lstm(
            text_chunk_0,
            (initial_hidden, initial_hidden),
        )
        relation_features, _ = self.relation_lstm(
            text_chunk_0,
            (initial_hidden, initial_hidden),
        )

        entity_features = self.entity_projection(entity_features)
        relation_features = self.relation_projection(relation_features)

        gate, auxiliary_outputs = self.alignment_gate(sentences, images)
        self.last_auxiliary_loss = auxiliary_outputs["aux_loss"]
        gate = gate.view(1, -1, 1)

        image_tokens = image_feature_2.unsqueeze(0).repeat(text_features.size(0), 1, 1)
        entity_context = self.entity_self_attention(entity_features, text_chunk_1)
        relation_context = self.relation_self_attention(relation_features, text_chunk_2)

        fused_entity_features = self.entity_cross_attention(
            image_tokens,
            entity_context,
            gate,
        )
        fused_relation_features = self.relation_cross_attention(
            image_tokens,
            relation_context,
            gate,
        )

        if token_positions is not None:
            token_positions = token_positions.transpose(0, 1)

        entity_scores = self.entity_decoder(
            entity_features,
            fused_entity_features,
            mask,
            token_positions,
        )
        relation_scores = self.relation_decoder(
            relation_features,
            fused_relation_features,
            mask,
            token_positions,
        )

        return entity_scores, relation_scores
