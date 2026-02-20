#!/usr/bin/env python3

"""Complete pipeline for multi-view refinement of coarse single-view pose predictions for BOP datasets."""
import os
import pandas as pd
import torch
from pathlib import Path
import tqdm

from foundpose.utils import (
    misc as misc_util,
    logging,
    json_util,
    config_util,
)

from src.utils_multiview.constants import OUTPUT_DIR
from src.utils_multiview.opts import RefineOpts
from src.utils_multiview.datasets.object_dataset import make_object_dataset
from src.utils_multiview.datasets.scene_dataset import make_scene_dataset
from src.utils_multiview.visualization.visualizer import Visualizer
from src.utils_multiview.pipeline.refinement_pipeline import MultiviewPoseRefiner
from src.utils_multiview.data.prediction_utils import (
    load_bop_predictions_scene,
    save_bop_predictions_csv,
)

def run_refine_multiview(
    opts: RefineOpts,
):
    """
    Runs the multi-view refinement pipeline for the given options, object dataset, and scene dataset.
    Args:
        opts: Options for the multi-view refinement pipeline.
    """

    # Prepare a logger.
    logger = logging.get_logger(level=logging.INFO if opts.debug else logging.WARNING)

    # Prepare a timer
    timer = misc_util.Timer(enabled=opts.debug)
    timer.start()

    # Set up the output directory.
    logger.info("Setting up output directory for results...")
    general_results_path = os.path.join(
        "multiview_refinement",
        opts.object_dataset,
        opts.version,
    )
    save_path_dir = os.path.join(
        OUTPUT_DIR,
        general_results_path,
    )
    Path(save_path_dir).mkdir(exist_ok=True, parents=True)
    logger.info(f"Saving results to: {save_path_dir}")

    # Save parameters to a file.
    input_base = opts.input_path.split("/")[-1]
    input_base_multiview = f"multiview-{input_base}"
    config_path = os.path.join(save_path_dir, input_base_multiview.replace(".csv", "_config.json"))
    json_util.save_json(config_path, opts)

    # Prepare a device.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {gpu_name}")
    timer.elapsed("Time for setting up the stage")

    # Create object dataset
    logger.info(f"Creating object dataset for {opts.object_dataset}...")


    # Load object representations
    object_dataset = make_object_dataset(dataset_name=opts.object_dataset)
    object_dataset.load_representations(
        repre_version=opts.repre_version,
        device=device,
    )

    # Load scene dataset (cameras and view groupings)
    scene_dataset = make_scene_dataset(dataset_name=opts.object_dataset, logger=logger)
    scene_dataset.load()
    
    # Load predictions
    predictions = load_bop_predictions_scene(
        camera_names=scene_dataset.camera_im_size.keys(),
        predictions_path=Path(opts.input_path),
        score_threshold=None,
        logger=logger,
    )

    # Visualization renderers for all cameras
    visualizer = None
    if opts.vis_results:
        visualizer = Visualizer(
            camera_im_sizes=scene_dataset.camera_im_size,
            path_to_store_vis=os.path.join(OUTPUT_DIR, "visualizations"),
        )
        
    mv_refiner = MultiviewPoseRefiner(
        opts=opts,
        object_database=object_dataset,
        device=device,
        logger=logger,
        timer=timer,
        visualizer=visualizer,
    )

    results = []
    for scene, candidates_world in tqdm.tqdm(scene_dataset.yield_scenes_with_predictions(predictions)):
        # Run multi-view refinement (candidates are already in world frame)
        refined_world = mv_refiner.refine_scene(scene=scene, candidate_poses=candidates_world)

        # Convert to per-camera output format
        results.append(scene_dataset.predictions_to_output(refined_world, scene.views))

    # Flatten the list of results
    results = pd.concat(results, ignore_index=True)

    # Save results for all sensors
    for sensor_name in scene_dataset.camera_im_size.keys():
        final_camera_preds = results[results["camera_id"] == sensor_name].reset_index(drop=True)

        # Save the results to a CSV file for BOP submission
        save_path = os.path.join(save_path_dir, f"multiview-{sensor_name}-{input_base}")
        
        logger.info("Saving BOP submission file for sensor: {}".format(sensor_name))
        save_bop_predictions_csv(
            predictions=final_camera_preds,
            out_csv_path=save_path,
        )
        logger.info("Results saved to: {}".format(save_path))




def main() -> None:
    # Load options
    opts = config_util.load_opts_from_json_or_command_line(RefineOpts)[0]

    run_refine_multiview(
        opts=opts,
        )


if __name__ == "__main__":
    main()
