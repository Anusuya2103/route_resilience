"""
PART C -- Occlusion-Aware Attention U-Net (Novelty 1).

Architecture rationale (each design choice is intentional and tied to the
occlusion-robustness goal):

* SAME backbone as the Part A baseline (U-Net + ResNet34) so the comparison
  isolates the effect of the attention machinery, not the encoder.

* SPATIAL SELF-ATTENTION at the bottleneck (between encoder and decoder).
  Road pixels occluded in one location can only be reconstructed by
  *referencing visible, distant* road pixels that continue the same road.
  A non-local / self-attention block at the lowest-resolution bottleneck gives
  every position a complete receptive field over the whole tile (long-range
  context) exactly where that cross-reference is cheapest to compute.  This is
  what lets the decoder 'see around' the occluder.

* CHANNEL ATTENTION (squeeze-and-excitation) in every decoder block.  After
  up-sampling + concatenation the feature channels carry BOTH road-continuity
  evidence and occlusion-texture artifacts.  SE lets each decoder block
  adaptively up-weight the road-continuity channels and down-weight the
  occlusion-texture channels when they compete -- channel-wise feature
  selection at the exact point where the conflict must be resolved.

* OCCLUSION-CONDITIONED LOSS (Part A step 2's occlusion masks): a per-pixel
  BCE term whose weight is (1 + OCCLOSS_MULT) *inside* occluded regions.  This
  makes "be robust exactly where occlusion happened" an explicit training
  signal rather than a hoped-for side effect.

The model outputs logits (B,1,H,W).  Train/eval/table live in
scripts/run_part_c.py; this module only defines the architecture + loss.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SpatialSelfAttention2D(nn.Module):
    """Non-local / self-attention block for a 2D feature map (bottleneck use).

    Q,K,V are 1x1-conv projections; the attention matrix is computed over the
    spatial positions (H*W tokens) of the bottleneck map, so any position can
    attend to any other position in the tile -- long-range context.
    """

    def __init__(self, in_ch: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or min(config.ATTENTION_BOTTLENECK_HIDDEN, in_ch)
        self.hidden = hidden
        self.q = nn.Conv2d(in_ch, hidden, 1)
        self.k = nn.Conv2d(in_ch, hidden, 1)
        self.v = nn.Conv2d(in_ch, in_ch, 1)
        self.proj = nn.Conv2d(in_ch, in_ch, 1)
        self.norm = nn.BatchNorm2d(in_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q = self.q(x).view(B, self.hidden, H * W).transpose(1, 2)      # B,N,Hd
        k = self.k(x).view(B, self.hidden, H * W)                       # B,Hd,N
        attn = F.softmax(torch.bmm(q, k) / (self.hidden ** 0.5), dim=-1)  # B,N,N
        v = self.v(x).view(B, C, H * W).transpose(1, 2)                 # B,N,C
        out = torch.bmm(attn, v).transpose(1, 2).view(B, C, H, W)
        return self.norm(self.proj(out) + x)                            # residual


class ChannelAttention2D(nn.Module):
    """Squeeze-and-excitation channel attention (use in the decoder)."""

    def __init__(self, channels: int, reduction: int | None = None):
        super().__init__()
        reduction = reduction or config.ATTENTION_SE_REDUCTION
        mid = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.adaptive_avg_pool2d(x, 1)
        return x * self.fc(w)


class DecoderBlock(nn.Module):
    """U-Net decode block: 3x3 conv-BN-ReLU x2, then channel attention (SE)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.se = ChannelAttention2D(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.conv(x))


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class OcclusionAttentionUNet(nn.Module):
    """U-Net (ResNet34 encoder) + spatial self-attention bottleneck +
    squeeze-and-excitation decoder, for binary road segmentation."""

    def __init__(self, out_channels: int = 1):
        super().__init__()
        self.encoder = self._build_encoder()

        # bottleneck spatial self-attention (between encoder and decoder)
        self.bottleneck_attention = SpatialSelfAttention2D(512)

        # decoder: 512 -> 256 -> 128 -> 64 with skip+SE at each level
        self.dec3 = DecoderBlock(512 + 256, 256)
        self.dec2 = DecoderBlock(256 + 128, 128)
        self.dec1 = DecoderBlock(128 + 64, 64)

        self.head = nn.Conv2d(64, out_channels, 1)

    @staticmethod
    def _build_encoder() -> nn.Module:
        """torchvision ResNet34 feature stages (channel map 64/128/256/512)."""
        try:
            from torchvision.models import resnet34, ResNet34_Weights
            enc = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            print("[attention] ResNet34 encoder (ImageNet pretrained)")
        except Exception as exc:                                  # offline fallback
            from torchvision.models import resnet34
            print(f"[attention] ImageNet weights unavailable ({exc}); random init.")
            enc = resnet34(weights=None)
        return enc

    def _encoder_features(self, x: torch.Tensor):
        e = self.encoder
        s = e.maxpool(F.relu(e.bn1(e.conv1(x))))   # stem, stride 4
        f1 = e.layer1(s)      # 64  ch, /4
        f2 = e.layer2(f1)     # 128 ch, /8
        f3 = e.layer3(f2)     # 256 ch, /16
        f4 = e.layer4(f3)     # 512 ch, /32
        return f1, f2, f3, f4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1, f2, f3, f4 = self._encoder_features(x)
        x = self.bottleneck_attention(f4)                        # (B,512,/32)
        x = self.dec3(torch.cat([self._up(x), f3], 1))
        x = self.dec2(torch.cat([self._up(x), f2], 1))
        x = self.dec1(torch.cat([self._up(x), f1], 1))
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        return self.head(x)

    @staticmethod
    def _up(x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# Occlusion-conditioned loss
# ---------------------------------------------------------------------------

def occlusion_conditioned_bce(logits: torch.Tensor, target: torch.Tensor,
                              occ: torch.Tensor) -> torch.Tensor:
    """Per-pixel BCE whose weight is (1 + OCCLOSS_MULT) at occluded pixels.

    occ: (B,1,H,W) float in {0,1}.  This directly operationalizes
    "occlusion-robust" as a training signal: errors under the occ mask are
    penalized OCCLOSS_MULT times harder than errors on clean pixels.
    """
    p = torch.sigmoid(logits)
    eps = 1e-7
    bce = -(target * torch.log(p.clamp(eps, 1 - eps))
            + (1 - target) * torch.log((1 - p).clamp(eps, 1 - eps)))
    weight = 1.0 + config.OCCLOSS_MULT * occ
    return (bce * weight).mean()


def attention_loss(logits: torch.Tensor, target: torch.Tensor,
                   occ: torch.Tensor | None) -> torch.Tensor:
    """Part C loss = BCE + Dice + OCCLOSS_MIX * occlusion-weighted BCE.

    Unlike the baseline's plain BCE, pixel errors inside occluders are
    explicitly up-weighted via the saved occlusion mask (Part A step 2).
    """
    from src.extraction import dice_loss
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = dice_loss(torch.sigmoid(logits), target)
    total = bce + dice
    if occ is not None:
        total = total + config.OCCLOSS_MIX * occlusion_conditioned_bce(logits, target, occ)
    return total