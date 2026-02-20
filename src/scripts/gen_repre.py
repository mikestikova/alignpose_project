#!/usr/bin/env python3

"""Generates a feature-based object representation."""

import os
from typing import Any, Dict, List, NamedTuple, Optional
from tqdm import tqdm
import torch

from bop_toolkit_lib import inout

from src.utils_multiview.opts import GenRepreOpts
from src.utils_multiview.datasets.object_dataset import make_object_dataset
from src.utils_multiview.constants import OUTPUT_DIR

from foundpose.utils.misc import array_to_tensor
from foundpose.utils import (
    config_util,
    feature_util,
    projector_util,
    repre_util,
    json_util,
    logging,
    misc,
)


def generate_feature_vectors(
    opts: GenRepreOpts,
    object_lid: int,
    extractor: torch.nn.Module,
    device: str = "cuda",
    debug: bool = False,
) -> List[torch.Tensor]:
    """Generates raw feature vectors from template images of the given object.
    Args:
        opts: Options for representation generation.
        object_lid: Local ID of the object.
        extractor: Feature extractor network.
        device: Device to use.
        debug: Whether to print debug info.
    Returns:
        List of feature vectors (torch.Tensor) extracted from the template images.
    """

    # Load the template metadata.
    # metadata_path = "/output/templates/v1/lmo/1/metadata.json"
    metadata_path = os.path.join(
        OUTPUT_DIR,
        "templates",
        opts.templates_version,
        opts.object_dataset,
        str(object_lid),
        "metadata.json",
    )
    metadata = json_util.load_json(metadata_path)

    # Prepare structures for storing data.
    feat_vectors_list = []

    # Use template images specified in the metadata.
    for data_sample in tqdm(metadata):
        # RGB/monochrome and depth images.
        image_path = data_sample["rgb_image_path"]
        mask_path = data_sample["binary_mask_path"]

        image_arr = inout.load_im(image_path)  # H,W,C
        mask_image_arr = inout.load_im(mask_path)

        image_chw = (
            array_to_tensor(image_arr).to(torch.float32).permute(2, 0, 1).to(device)
            / 255.0
        )
        object_mask_modal = array_to_tensor(mask_image_arr).to(torch.float32).to(device)

        # Get the object annotation.
        assert data_sample["dataset"] == opts.object_dataset
        assert data_sample["lid"] == object_lid

        # Extract features from the current template.
        feat_vectors = feature_util.get_masked_features(
            image_chw=image_chw,
            object_mask=object_mask_modal,
            extractor=extractor,
            grid_cell_size=opts.grid_cell_size,
            debug=debug,
        )
        feat_vectors_list.append(feat_vectors)

    # Return the collected feature vectors
    return torch.cat(feat_vectors_list, dim=0)


def generate_object_repre(
    opts: GenRepreOpts,
    object_lid: int,
    device: str = "cuda",
    extractor: Optional[torch.nn.Module] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    
    # Prepare a logger.
    logger = logger or logging.get_logger(level=logging.INFO if opts.debug else logging.WARNING)
    logger.info(
        f"Generating representation for object {object_lid} of dataset {opts.object_dataset}..."
    )

    # Prepare a timer.
    timer = misc.Timer(enabled=opts.debug)
    timer.start()

    # Prepare the output folder.
    base_repre_dir = os.path.join(OUTPUT_DIR, "object_repre")
    output_dir = repre_util.get_object_repre_dir_path(
        base_repre_dir, opts.version, opts.object_dataset, object_lid
    )
    if os.path.exists(output_dir) and not opts.overwrite:
        raise ValueError(f"Output directory already exists: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Save parameters to a JSON file.
    json_util.save_json(os.path.join(output_dir, "config.json"), opts)

    # Prepare a feature extractor.
    extractor.to(device)

    timer.elapsed("Time for preparation")
    timer.start()

    # Build raw object representation.
    logger.info("Building raw object representation...")
    feat_vectors = generate_feature_vectors(
        opts=opts,
        object_lid=object_lid,
        extractor=extractor,
        device=device,
    )
    assert feat_vectors is not None

    timer.elapsed("Time for generating raw representation")
    timer.start()

    # Compute the PCA space.
    if opts.apply_pca:
        logger.info("Preparing PCA...")
        pca_projector = projector_util.PCAProjector(
            n_components=opts.pca_components, whiten=opts.pca_whiten
        )
        pca_projector.fit(feat_vectors, max_samples=opts.pca_max_samples_for_fitting)
    else:
        pca_projector = projector_util.IdentityProjector()

    # Create an object representation and store the raw feature projector.
    repre = repre_util.FeatureBasedObjectRepre(
        feat_opts=repre_util.FeatureOpts(extractor_name=opts.extractor_name),
    )
    repre.feat_raw_projectors.append(pca_projector)
    repre.feat_vis_projectors = [repre.feat_raw_projectors[0]]

    timer.elapsed("Time for PCA")
    timer.start()

    # Save the generated object representation.
    repre_dir = repre_util.get_object_repre_dir_path(
        base_repre_dir, opts.version, opts.object_dataset, object_lid
    )
    repre_util.save_simple_object_repre(repre, repre_dir)

    timer.elapsed("Time for saving the object representation")


def generate_dataset_repre(
    opts: GenRepreOpts,
) -> None:
    
    # Prepare a logger.
    logger = logging.get_logger(level=logging.INFO if opts.debug else logging.WARNING)

    # Create object dataset
    logger.info(f"Creating object dataset for {opts.object_dataset}...")
    object_dataset = make_object_dataset(
        dataset_name=opts.object_dataset,
        model_tpath=opts.path_to_models,
    )
    logger.info(f"Object dataset created with {len(object_dataset.object_ids)} objects.")

    # Get IDs of objects to process.
    object_lids = opts.object_lids
    if object_lids is None:
        # If local (object) IDs are not specified, generate repre for all objects
        # in the specified dataset.
        object_lids = list(object_dataset.object_ids)

    # Prepare a feature extractor.
    extractor = feature_util.make_feature_extractor(opts.extractor_name)

    # Prepare a device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)

    assert opts.object_dataset == object_dataset.dataset_name, (
        f"Object dataset name in options ({opts.object_dataset}) does not match the provided object dataset name ({object_dataset.dataset_name})."
    )
    # Process each image separately.
    for object_lid in object_lids:
        generate_object_repre(
            opts, object_lid, device, extractor, logger=logger
        )

def main() -> None:
    opts = config_util.load_opts_from_json_or_command_line(GenRepreOpts)[0]

    # Generate object representations for the dataset
    generate_dataset_repre(opts)

if __name__ == "__main__":
    main()