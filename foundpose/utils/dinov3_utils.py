import os
from pathlib import Path
import typing as tp

import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import AutoImageProcessor, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DINOv3FeatureExtractor(nn.Module):
    # REPO_DIR = (Path(os.path.dirname(__file__)).parent.parent.parent.parent / 'dinov3').resolve()

    def __init__(self, model_name: str = "dinov3_version=vitl16_stride=16_facet=token_layer=18_norm=1"):
        super().__init__()

        # Default parameter values.
        self.version: str = "vits16"
        self.stride: int = 16
        self.facet: str = "token"
        self.layer: int = 9
        self.apply_norm: bool = True

        # Parse the model name.
        name_items = model_name.split("_")
        assert name_items[0] == "dinov3"
        if len(name_items) == 2:
            # Example: "dinov3_vits16"
            self.version = name_items[1]
        else:
            # Example: "dinov3_version=vitl16_stride=16_facet=token_layer=18_norm=1"
            for item in name_items[1:]:
                name, value = item.split("=")
                if name == "version":
                    self.version = value
                elif name == "stride":
                    self.stride = int(value)
                elif name == "facet":
                    self.facet = value
                elif name == "layer":
                    self.layer = int(value)
                elif name == "norm":
                    self.apply_norm = bool(int(value))

        # Build the base model.
        self.model_base_name: str = f"dinov3-{self.version}-pretrain-lvd1689m"
        self.processor = AutoImageProcessor.from_pretrained(
            f"facebook/{self.model_base_name}", do_rescale=False, do_resize=False
        )
        self.model = AutoModel.from_pretrained(f"facebook/{self.model_base_name}")

        self.model.eval().to(device)
        self.model.requires_grad_(False)

        self.patch_size = self.model.config.patch_size
        assert self.facet == "token", (
            f'Only "token" facet is supported. Found {self.facet}.'
        )
        assert self.stride == self.patch_size, (
            f"Only stride equal to patch size is supported. Found stride={self.stride}, patch_size={self.patch_size}."
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> tp.Dict[str, torch.Tensor]:
        assert images.ndim == 4, (
            f"Expected images to be a batch of images with shape (B, C, H, W). Found {images.shape}"
        )
        assert (
            images.shape[-1] == images.shape[-2]
            and images.shape[-1] % self.patch_size == 0
        ), (
            f"Expected images to be square with size multiple of patch size {self.patch_size} (e.g. {[self.patch_size * 27, self.patch_size * 27]}). Found {images.shape[-2:]}"
        )
        assert images.dtype == torch.float32, (
            f"Expected images to be float32. Found {images.dtype}"
        )

        with torch.inference_mode():
            images = self.processor(images=images, return_tensors="pt").pixel_values.to(
                self.model.device
            )

            outputs = self.model(images, output_hidden_states=True)
            layer_outputs = outputs.hidden_states[self.layer + 1]

            cls_tokens = layer_outputs[:, 0, :].unsqueeze(1)
            reg_tokens = layer_outputs[:, 1 : self.model.config.num_register_tokens + 1]
            patch_tokens = layer_outputs[
                :, 1 + self.model.config.num_register_tokens :, :
            ]

            # Normalize the tokens. Never for the last layer.
            if self.apply_norm:
                if self.layer != self.model.config.num_hidden_layers - 1:
                    tokens = torch.cat([cls_tokens, patch_tokens], dim=1)
                    tokens = self.model.norm(tokens)
                    cls_tokens = tokens[:, :1, :]
                    patch_tokens = tokens[:, 1:, :]

                # Layer norm is already applied in the last layer of the model
                else:
                    pass
            else:
                if self.layer == self.model.config.num_hidden_layers - 1:
                    raise ValueError(
                        "Skipping normalization on the last layer is not supported."
                    )

            # Reshape patch tokens to BxDxHxW.
            bsz, _, w, h = images.shape
            d = patch_tokens.shape[-1]
            num_patches = (
                1 + (h - self.patch_size) // self.stride,
                1 + (w - self.patch_size) // self.stride,
            )
            feature_maps = patch_tokens.reshape(
                bsz, num_patches[1], num_patches[0], d
            ).permute(0, 3, 1, 2)

        return {
            "cls_tokens": cls_tokens[:, 0, :],  # BxD
            "feature_maps": feature_maps,  # BxDxHxW
        }
