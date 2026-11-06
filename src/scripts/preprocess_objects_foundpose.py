#!/usr/bin/env python3

"""Builds the full FoundPose representation of objects, directly from meshes.

The full representation is needed by the single-view pose estimation (FoundPose) and
also contains the PCA used by the multi-view refinement. Replaces running
`foundpose/scripts/gen_templates.py` + `gen_repre.py`, without a BOP dataset. The
templates are rendered in memory; they are written to disk only with `--save-templates`.

Example:
    python src/scripts/preprocess_objects_foundpose.py \
        --model-paths path/to/obj_000001.ply path/to/obj_000002.ply \
        --object-ids 1 2 \
        --camera path/to/camera.json \
        --distance 600 \
        --output-dir path/to/object_repre

The templates are rendered with the camera in `--camera` (`fx`, `fy`, `cx`, `cy`,
`width`, `height`, see `load_intrinsics`) at `--distance` mm from the object. Use the
camera and the working distance the objects will be seen at. Each representation is
saved to `<output-dir>/<object id>/repre.pth`, with the options used in `config.json`.
With `--save-templates`, the rendered templates are saved to
`<output-dir>/<object id>/templates` for debugging.
"""

import argparse
import os

import torch

from foundpose.utils import feature_util, renderer_builder
from src.utils_multiview.processing.foundpose_repre import (
    DEFAULT_EXTRACTOR_NAME,
    build_foundpose_repre,
    foundpose_repre_config,
    load_intrinsics,
    render_foundpose_templates,
    save_foundpose_repre,
    save_templates,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--model-paths", nargs="+", required=True, help="Object meshes in millimeters.")
    parser.add_argument("--object-ids", nargs="+", required=True, help="Output folder names, one per model path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--camera", required=True, help="Camera intrinsics file (.json or .yml).")
    parser.add_argument("--distance", type=float, required=True, help="Camera-object distance in mm.")
    parser.add_argument("--ssaa-factor", type=float, default=4.0,
                        help="Supersampling for anti-aliasing; use 1 for large camera images.")
    parser.add_argument("--extractor-name", default=DEFAULT_EXTRACTOR_NAME)
    parser.add_argument("--min-num-viewpoints", type=int, default=57)
    parser.add_argument("--num-inplane-rotations", type=int, default=14)
    parser.add_argument("--save-templates", action="store_true",
                        help="Also save the rendered templates, for debugging.")
    args = parser.parse_args()
    if len(args.object_ids) != len(args.model_paths):
        parser.error("--object-ids must have the same length as --model-paths.")

    K, image_size = load_intrinsics(args.camera)
    depth_range = (args.distance, args.distance)
    render_opts = dict(
        ssaa_factor=args.ssaa_factor,
        min_num_viewpoints=args.min_num_viewpoints,
        num_inplane_rotations=args.num_inplane_rotations,
    )

    # Load the extractor and the renderer once and reuse them for all objects.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = feature_util.make_feature_extractor(args.extractor_name)
    renderer = renderer_builder.build(
        renderer_type=renderer_builder.RendererType.PYRENDER_RASTERIZER
    )

    for object_id, model_path in zip(args.object_ids, args.model_paths):
        print(f"Building representation of {model_path}...")

        # Template generation.
        templates = render_foundpose_templates(
            model_path, K, image_size, depth_range, renderer=renderer, **render_opts
        )
        if args.save_templates:
            save_templates(templates, os.path.join(args.output_dir, str(object_id), "templates"))

        # Representation extraction.
        repre = build_foundpose_repre(
            templates, extractor_name=args.extractor_name, extractor=extractor, device=device
        )

        config = foundpose_repre_config(
            model_path, K, image_size, depth_range, args.extractor_name, render_opts
        )
        repre_dir = save_foundpose_repre(repre, args.output_dir, object_id, config=config)
        print(f"Saved to {repre_dir}")


if __name__ == "__main__":
    main()
