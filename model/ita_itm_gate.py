"""Unified Alignment-Matching Gate (UAG) used by the AMSG framework."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize
from transformers import CLIPModel, CLIPTokenizerFast


CLIP_IMAGE_MEAN = torch.tensor(
    [0.48145466, 0.4578275, 0.40821073]
).view(1, 3, 1, 1)
CLIP_IMAGE_STD = torch.tensor(
    [0.26862954, 0.26130258, 0.27577711]
).view(1, 3, 1, 1)


def preprocess_clip_images(
    pixel_values: torch.Tensor,
    image_size: int = 224,
) -> torch.Tensor:
    """Resize and normalize images for the frozen CLIP encoder."""

    if pixel_values.size(-1) != image_size or pixel_values.size(-2) != image_size:
        pixel_values = resize(pixel_values, [image_size, image_size], antialias=True)

    mean = CLIP_IMAGE_MEAN.to(device=pixel_values.device, dtype=pixel_values.dtype)
    std = CLIP_IMAGE_STD.to(device=pixel_values.device, dtype=pixel_values.dtype)
    return (pixel_values - mean) / std


class UnifiedAlignmentGate(nn.Module):
    """Estimate sample-level image-text reliability with ITA and ITM.

    The CLIP encoder is frozen. The ITA branch learns a task-specific
    alignment space with symmetric in-batch InfoNCE, while the ITM branch
    uses bidirectional in-batch hard negatives for binary matching.
    """

    def __init__(
        self,
        pretrained: str = "openai/clip-vit-base-patch32",
        d_align: int = 512,
        tau: float = 0.07,
        lambda_ita: float = 0.10,
        lambda_itm: float = 0.05,
        init_ita_weight: float = 1.0,
        init_itm_weight: float = 1.0,
        freeze_clip: bool = True,
    ) -> None:
        super().__init__()

        if tau <= 0:
            raise ValueError("tau must be positive.")

        self.clip = CLIPModel.from_pretrained(pretrained)
        if freeze_clip:
            for parameter in self.clip.parameters():
                parameter.requires_grad_(False)
            self.clip.eval()

        clip_hidden_size = self.clip.config.projection_dim
        self.text_proj = nn.Linear(clip_hidden_size, d_align)
        self.image_proj = nn.Linear(clip_hidden_size, d_align)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / tau)))

        self.itm_head = nn.Sequential(
            nn.Linear(clip_hidden_size * 2, clip_hidden_size * 4),
            nn.Tanh(),
            nn.Linear(clip_hidden_size * 4, 2),
        )

        self.ita_gate_weight = nn.Parameter(torch.tensor(init_ita_weight))
        self.itm_gate_weight = nn.Parameter(torch.tensor(init_itm_weight))

        self.lambda_ita = lambda_ita
        self.lambda_itm = lambda_itm
        self.tokenizer = CLIPTokenizerFast.from_pretrained(pretrained)

    def train(self, mode: bool = True):
        """Keep the frozen CLIP encoder in evaluation mode."""

        super().train(mode)
        self.clip.eval()
        return self

    @torch.no_grad()
    def _encode_clip_features(
        self,
        sentences: Sequence[str],
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = images.device
        image_inputs = preprocess_clip_images(images)
        image_features = self.clip.get_image_features(pixel_values=image_inputs)
        image_features = F.normalize(image_features, dim=-1)

        text_inputs = self.tokenizer(
            list(sentences),
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        text_features = self.clip.get_text_features(**text_inputs)
        text_features = F.normalize(text_features, dim=-1)

        return text_features, image_features

    def _compute_ita(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        projected_text = F.normalize(self.text_proj(text_features), dim=-1)
        projected_image = F.normalize(self.image_proj(image_features), dim=-1)

        similarity = projected_image @ projected_text.transpose(0, 1)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits = similarity * logit_scale

        batch_size = text_features.size(0)
        targets = torch.arange(batch_size, device=text_features.device)
        image_to_text_loss = F.cross_entropy(logits, targets)
        text_to_image_loss = F.cross_entropy(logits.transpose(0, 1), targets)
        alignment_loss = (image_to_text_loss + text_to_image_loss) * 0.5

        return alignment_loss, similarity.diagonal()

    def _compute_itm(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = text_features.size(0)
        if batch_size < 2:
            raise ValueError("ITM hard-negative mining requires a batch size of at least 2.")

        positive_pairs = torch.cat([text_features, image_features], dim=-1)
        positive_logits = self.itm_head(positive_pairs)
        positive_margin = positive_logits[:, 1] - positive_logits[:, 0]

        with torch.no_grad():
            image_to_text_similarity = image_features @ text_features.transpose(0, 1)
            text_to_image_similarity = image_to_text_similarity.transpose(0, 1)

            image_to_text_similarity.fill_diagonal_(-1e4)
            text_to_image_similarity.fill_diagonal_(-1e4)

            negative_image_indices = text_to_image_similarity.argmax(dim=1)
            negative_text_indices = image_to_text_similarity.argmax(dim=1)

        image_hard_negative_pairs = torch.cat(
            [text_features, image_features[negative_image_indices]],
            dim=-1,
        )
        text_hard_negative_pairs = torch.cat(
            [text_features[negative_text_indices], image_features],
            dim=-1,
        )

        all_pairs = torch.cat(
            [positive_pairs, image_hard_negative_pairs, text_hard_negative_pairs],
            dim=0,
        )
        all_logits = self.itm_head(all_pairs)
        labels = torch.cat(
            [
                torch.ones(batch_size, dtype=torch.long, device=text_features.device),
                torch.zeros(2 * batch_size, dtype=torch.long, device=text_features.device),
            ],
            dim=0,
        )
        matching_loss = F.cross_entropy(all_logits, labels)

        return matching_loss, positive_margin

    def forward(
        self,
        sentences: Sequence[str],
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return a sample-specific gate and auxiliary training losses."""

        text_features, image_features = self._encode_clip_features(sentences, images)
        alignment_loss, alignment_score = self._compute_ita(text_features, image_features)
        matching_loss, matching_score = self._compute_itm(text_features, image_features)

        gate_logits = (
            self.ita_gate_weight * alignment_score
            + self.itm_gate_weight * matching_score
        )
        gate = torch.sigmoid(gate_logits).unsqueeze(1)

        auxiliary_loss = self.lambda_ita * alignment_loss + self.lambda_itm * matching_loss
        auxiliary_outputs = {
            "loss_ita": alignment_loss.detach(),
            "loss_itm": matching_loss.detach(),
            "aux_loss": auxiliary_loss,
            "sim_ita": alignment_score.detach(),
            "itm_margin": matching_score.detach(),
        }
        return gate, auxiliary_outputs
