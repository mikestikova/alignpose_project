import torch
from transformers import AutoModel, CLIPImageProcessor,AutoImageProcessor

from pathlib import Path

import torch.nn.functional as F
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from kornia.color import rgb_to_grayscale
from PIL import Image

import typing as tp
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RadioFeatureExtractor(nn.Module):

    def __init__(self, model_name: str = None):
        super().__init__()

        # Default parameter values.
        self.version: str = ""
        self.stride: int = 16
        self.facet: str = "token"
        self.layer: int = 9
        self.apply_norm: bool = False

        # Parse the model name.
        name_items = model_name.split("_")
        assert name_items[0] == "radiov2"
        if len(name_items) == 2:
            # Example: "radiov2_H"
            self.version = name_items[1]
        else:
            raise NotImplementedError
            # Example: "radio_version=vitl16_stride=16_facet=token_layer=18_norm=1"
            # for item in name_items[1:]:
            #     name, value = item.split("=")
            #     if name == "version":
            #         self.version = value
            #     elif name == "stride":
            #         self.stride = int(value)
            #     elif name == "facet":
            #         self.facet = value
            #     elif name == "layer":
            #         self.layer = int(value)
            #     elif name == "norm":
            #         self.apply_norm = bool(int(value))

        # Build the base model.
        self.model_base_name: str = f"C-RADIOv2-{self.version}"
        self.processor = CLIPImageProcessor.from_pretrained(
            f"nvidia/{self.model_base_name}",
            do_rescale=False,
            do_resize=False, 
            do_convert_rgb=False
            )
        self.model = AutoModel.from_pretrained(f"nvidia/{self.model_base_name}", trust_remote_code=True)

        self.model.eval().to(device)
        self.model.requires_grad_(False) 
          
        self.patch_size = self.model.config.patch_size
        assert self.facet == "token", f'Only "token" facet is supported. Found {self.facet}.'
        assert self.stride == self.patch_size, f'Only stride equal to patch size is supported. Found stride={self.stride}, patch_size={self.patch_size}.'



    def forward(self, images: torch.Tensor,) -> tp.Dict[str, torch.Tensor]:
        assert images.ndim == 4, f'Expected images to be a batch of images with shape (B, C, H, W). Found {images.shape}'
        assert images.shape[-1] == images.shape[-2] and images.shape[-1] % self.patch_size == 0, f'Expected images to be square with size multiple of patch size {self.patch_size}. Found {images.shape[-2:]}'
        assert images.dtype == torch.float32, f'Expected images to be float32. Found {images.dtype}'
        
        with torch.inference_mode():
            images = self.processor(images=images, return_tensors='pt').pixel_values.to(self.model.device)
            summary, patch_tokens = self.model(images) # BxC, BxNxD

            if self.apply_norm:
                raise NotImplementedError
            
            # Reshape patch tokens to BxDxHxW.
            bsz, _, h, w = images.shape
            d = patch_tokens.shape[-1]
            num_patches = (
                1 + (h - self.patch_size) // self.stride,
                1 + (w - self.patch_size) // self.stride,
            )
            feature_maps = patch_tokens.reshape(
                bsz, num_patches[1], num_patches[0], d
            ).permute(0, 3, 1, 2).contiguous()

            return {
                "cls_tokens": summary,  # BxC, C is different from D, summary is similar to cls token
                "feature_maps": feature_maps, # BxDxHxW
            }