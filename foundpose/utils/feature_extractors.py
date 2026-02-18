import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from kornia.feature import DenseSIFTDescriptor
from kornia.color import rgb_to_grayscale
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"


class RGBFeatureExtractor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        # Parse the model name.
        name_items = model_name.split("_")
        assert name_items[0] == "RGB"
        self.patch_size = int(name_items[1])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Assuming images are in the shape (B, C, H, W)
        """
        Args:
            x: Tensor of shape (B, C, H, W)
        Returns:
            Tensor of shape (B, C, H_out, W_out),
            where H_out = H // patch_size and W_out = W // patch_size.
        """
        B, C, H, W = images.shape
        p = self.patch_size

        # Ensure divisible dimensions
        assert H % p == 0 and W % p == 0, "H and W must be divisible by patch_size"

        # Use avg_pool2d to compute per-patch means efficiently
        # return images
        out = F.avg_pool2d(images, kernel_size=p, stride=p)
        result = {
            "feature_maps": out,
        }
        return result


class SIFTFeatureExtractor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        # Parse the model name. Example: "sift_stride=7_bins=4_binsize=4_pool=1_norm=1"
        # should be: bins + binsize = stride + 1
        name_items = model_name.split("_")
        assert name_items[0] == "sift"

        # Default parameters
        self.cell_stride = 7
        self.stride = 2
        self.bins = 2
        self.binsize = 7
        self.apply_norm = True
        self.pool = 1

        for item in name_items[1:]:
            name, value = item.split("=")
            if name == "stride":
                self.stride = int(value)
            elif name == "cell-stride":
                self.cell_stride = int(value)
            elif name == "bins":
                self.bins = int(value)
            elif name == "binsize":
                self.binsize = int(value)
            elif name == "norm":
                self.apply_norm = bool(int(value))
            elif name == "pool":
                self.pool = int(value)
        self.model = DenseSIFTDescriptor(
            num_spatial_bins=self.bins,
            spatial_bin_size=self.binsize,
            cell_stride=self.cell_stride,
            cell_padding=0,
            stride=self.stride,
            padding=0,
        ).to(device)
        # self.model = DenseSIFTDescriptor().to(device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Assuming images are in the shape (B, C, H, W)
        """
        Args:
            x: Tensor of shape (B, C, H, W)
        Returns:
            Tensor of shape (B, C, H_out, W_out),
            where H_out = H // patch_size and W_out = W // patch_size.
        """
        B, C, H, W = images.shape
        p = self.pool

        # Ensure divisible dimensions
        assert H % p == 0 and W % p == 0, "H and W must be divisible by patch_size"

        images = images.to(device)
        images_gray = rgb_to_grayscale(images)
        with torch.no_grad():
            out = self.model(images_gray)
            if self.pool > 1:
                out = F.avg_pool2d(out, kernel_size=p, stride=p)
            # normalize to have mean 0 and std 1 per feature map
            if self.apply_norm:
                out = out - out.mean(dim=1, keepdim=True)
                out = out / (out.std(dim=1, keepdim=True) + 1e-6)

        result = {
            "feature_maps": out,
        }
        return result