"""SENet model for Tarneeb AlphaZero.

Architecture
------------
Input: ``(N, 4, 13, 3)`` card grid (NHWC) + ``(N, G)`` global features.

The card grid is permuted to ``(N, 3, 4, 13)`` (NCHW) and processed by a
lightweight SE-ResNet backbone (4 × PreActBlock residual stages with
Squeeze-and-Excitation).  The resulting ``(N, 512)`` feature vector is
concatenated with the global features before two heads:

* **Policy head**: ``Linear(512 + G → ACTION_SIZE)`` — raw logits
* **Value head**: ``Linear(512 + G → 4)`` → ``tanh`` — per-player values
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.alphazero.model import Model
from envs.tarneeb.alphazero_encoding import ACTION_SIZE, GLOBAL_FEATURES_SIZE


class _PreActBlock(nn.Module):
    """Pre-activation residual block with Squeeze-and-Excitation."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

        se_planes = max(planes // 16, 1)
        self.se_fc1 = nn.Conv2d(planes, se_planes, kernel_size=1)
        self.se_fc2 = nn.Conv2d(se_planes, planes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        # Squeeze-and-Excitation
        w = F.avg_pool2d(out, (out.size(2), out.size(3)))
        w = F.relu(self.se_fc1(w))
        w = torch.sigmoid(self.se_fc2(w))
        out = out * w

        return out + shortcut


class TarneebSENet(Model):
    """SE-ResNet for the 4-player Tarneeb card game.

    Args:
        global_features_size: dimensionality of the scalar context vector
                              (default: ``GLOBAL_FEATURES_SIZE`` = 45).
        num_blocks:           number of PreActBlocks per stage.
                              Default ``[1, 1, 1, 1]`` keeps the model
                              lightweight for the small 4×13 card grid.
    """

    def __init__(
        self,
        global_features_size: int = GLOBAL_FEATURES_SIZE,
        num_blocks: list[int] | None = None,
    ) -> None:
        if num_blocks is None:
            num_blocks = [1, 1, 1, 1]

        super().__init__(
            input_shape=(4, 13, 3),
            p_shape=(ACTION_SIZE,),
            v_shape=(4,),
        )

        self._global_features_size = global_features_size

        # Backbone
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        combined = 512 + global_features_size
        self.p_head = nn.Linear(combined, ACTION_SIZE)
        self.v_head = nn.Linear(combined, 4)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(_PreActBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, global_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:               ``(N, 4, 13, 3)`` float32 NHWC card grid
            global_features: ``(N, G)`` float32 scalar context

        Returns:
            p_logits: ``(N, 89)`` policy logits
            v:        ``(N, 4)`` per-player tanh values
        """
        # NHWC → NCHW
        x = x.permute(0, 3, 1, 2)  # (N, 3, 4, 13)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        flat = out.view(out.size(0), -1)  # (N, 512)

        combined = torch.cat([flat, global_features], dim=1)  # (N, 512+G)

        p_logits = self.p_head(combined)
        v = torch.tanh(self.v_head(combined))
        return p_logits, v
