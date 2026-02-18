from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class QueryRepre:
    features: torch.Tensor  # (num_patches, D)
    rgb: Optional[np.ndarray] = None  # (H, W, 3), for visualization only


@dataclass
class TemplateRepre:
    masked_features: torch.Tensor  # (N, D) — features at visible surface points
    vertices: torch.Tensor  # (N, 3) — 3D coordinates of visible surface points
    rgb: Optional[np.ndarray] = None  # (H, W, 3), for visualization
    depth: Optional[np.ndarray] = None  # (H, W), for visualization
    features: Optional[torch.Tensor] = None  # full feature map, for visualization
