#!/usr/bin/env python3

"""Fits the per-object PCA needed by the multi-view refinement, directly from meshes.

Replaces running `gen_templates.py` + `gen_repre.py` when only the multi-view refinement
is needed. No template images are written to disk.

Example:
    python src/scripts/preprocess_objects.py \
        --model-paths path/to/obj_000001.ply path/to/obj_000002.ply \
        --object-ids 1 2 \
        --output-dir path/to/object_repre

Each representation is saved to `<output-dir>/<object id>/repre.pth`. To use the
representations in `refine_multiview.py`, set `--output-dir` to
`<OUTPUT_DIR>/object_repre/<object_dataset>/<repre_version>`.
"""

import argparse

import torch

from foundpose.utils import feature_util, renderer_builder
from src.utils_multiview.processing.object_pca import (
    DEFAULT_EXTRACTOR_NAME,
    fit_object_pca,
    save_object_pca,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--model-paths", nargs="+", required=True, help="Object meshes in millimeters.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--object-ids", nargs="+", required=True, help="Output folder names, one per model path.")
    parser.add_argument("--extractor-name", default=DEFAULT_EXTRACTOR_NAME)
    parser.add_argument("--pca-components", type=int, default=256)
    parser.add_argument("--min-num-viewpoints", type=int, default=57)
    parser.add_argument("--num-inplane-rotations", type=int, default=14)
    args = parser.parse_args()
    if len(args.object_ids) != len(args.model_paths):
        parser.error("--object-ids must have the same length as --model-paths.")

    # Load the extractor and the renderer once and reuse them for all objects.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = feature_util.make_feature_extractor(args.extractor_name)
    renderer = renderer_builder.build(
        renderer_type=renderer_builder.RendererType.PYRENDER_RASTERIZER
    )

    for object_id, model_path in zip(args.object_ids, args.model_paths):
        print(f"Fitting PCA for {model_path}...")
        repre, config = fit_object_pca(
            model_path,
            extractor_name=args.extractor_name,
            extractor=extractor,
            renderer=renderer,
            device=device,
            pca_components=args.pca_components,
            min_num_viewpoints=args.min_num_viewpoints,
            num_inplane_rotations=args.num_inplane_rotations,
        )
        repre_dir = save_object_pca(repre, args.output_dir, object_id, config=config)
        print(f"Saved to {repre_dir}")


if __name__ == "__main__":
    main()
