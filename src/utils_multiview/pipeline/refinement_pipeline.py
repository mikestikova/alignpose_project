#!/usr/bin/env python3

"""Infers pose from objects."""

import os
import gc

from typing import List, NamedTuple, Optional, Tuple

import torch
import numpy as np
import pandas as pd

import time
import cv2

from src.utils_multiview.refinement import featuremetric_refiner_wrapper
from src.utils_multiview.datasets.object_dataset import ObjectDataset
from src.utils_multiview.structs.data_structures import Scene, View, ObjectInstance
from src.utils_multiview.visualization.visualizer import Visualizer
from src.utils_multiview.processing.view_processor import ViewProcessor
from src.utils_multiview.opts import RefineOpts


from foundpose.utils import (
    misc as misc_util,
    logging,
    misc,
    structs,
)
from src.utils_multiview.nms import nms_3d_base

class MultiviewPoseRefiner:
    def __init__(
        self,
        opts: RefineOpts,
        object_database: ObjectDataset,
        visualizer: Optional[Visualizer] = None,
        device: str = "cpu",
        logger=None,
        timer=None,

    ):
        self.opts = opts
        self.object_database = object_database

        # Finish setting up visualizer
        if visualizer is not None:
            self.visualizer = visualizer
            self.visualizer.add_object_models(object_database)
        else:
            self.visualizer = None

        self.logger = logger or logging.get_logger(level=logging.WARNING)
        self.device = device
        self.timer = timer or misc_util.Timer(enabled=opts.debug)

        self.view_processor = ViewProcessor(
            opts=opts,
            device=device,
        )

    def run_mv_refinement(
        self,
        initial_pose_wm: structs.RigidTransform,
        per_view_info: dict,
    ) -> Tuple[np.ndarray, torch.Tensor, dict, bool]:
        """
        Refines the object pose using multiview featuremetric refinement.

        Returns:
            refined_pose_wm (structs.RigidTransform): Refined pose
            mean_cost (torch.Tensor): Mean final cost from refinement
            updated_per_view_info (dict): Same keys, values updated with costs
            failed (bool): Whether refinement failed
        """

        if per_view_info is None or len(per_view_info) == 0:
            self.logger.warning("No valid views for refinement, skipping.")
            return initial_pose_wm, torch.tensor(1), per_view_info, True

        virtual_cameras = [view["virtual_camera"] for view in per_view_info.values()]
        templates = [view["template"] for view in per_view_info.values()]
        queries = [view["query"] for view in per_view_info.values()]

        refined_pose_wm, failed, costs = (
            featuremetric_refiner_wrapper.refine_multiview_wrapper(
                initial_pose_wm=initial_pose_wm,
                templates=templates,
                queries=queries,
                cameras=virtual_cameras,
                num_iter=self.opts.num_iters,
                loss_fn=self.opts.loss_fn,
            )
        )

        if failed:
            refined_pose_wm = initial_pose_wm
            mean_cost = torch.tensor(1)  # or any fallback
        else:
            initial_costs = costs[0]
            final_costs = costs[-1]
            mean_cost = torch.mean(final_costs)

            for (camera_id, view_id), init_cost, final_cost in zip(
                per_view_info.keys(), initial_costs, final_costs
            ):
                per_view_info[(camera_id, view_id)]["initial_loss"] = init_cost
                per_view_info[(camera_id, view_id)]["final_loss"] = final_cost

        return refined_pose_wm, mean_cost, per_view_info, failed

    def refine_object_pose(
        self,
        scene: Scene,
        object_id: int,
        object_pose_wm: structs.RigidTransform,
    ) -> Tuple[structs.RigidTransform, float, bool]:
        """
        Refines all candidate poses for the given scene.
        Args:
            scene (Scene): Scene containing multiple views.
            object_id (int): Object ID. Object must be in the object database.
            object_pose_wm (structs.RigidTransform): Initial object pose.

        Returns:
            refined_pose_wm (structs.RigidTransform): Refined object pose.
            score (float): Score of the refined pose, between 0 and 1. Higher is better.
            failed (bool): Whether refinement failed.
        """
        cand_object = self.object_database.objects[object_id]

        # Preprocess all views
        time_start_processing = time.time()
        per_view_info = {}
        for view in scene.views:   
            # self.logger.info(
            #     f"Processing view {view.image_id} from camera {view.camera_id} for object {cand_object.object_id}"
            # )
            virtual_camera, query, template, failed = self.view_processor(
                cand_object=cand_object,
                object_pose_wm=misc.get_rigid_matrix(object_pose_wm),
                view=view,
            )
            # Store the view information
            if not failed:
                per_view_info[(view.camera_id, view.image_id)] = {
                    "virtual_camera": virtual_camera,
                    "query": query,
                    "template": template,
                }

        time_end_processing = time.time()

        # If no valid views, skip
        if len(per_view_info) == 0:
            self.logger.warning("No valid views.")
            return object_pose_wm, 0.0, True

        # Refine the pose
        time_start_refinement = time.time()
        refined_pose_wm, cost, per_view_info, failed = self.run_mv_refinement(
            initial_pose_wm=object_pose_wm, per_view_info=per_view_info
        )
        if failed:  # refined pose will be the initial pose
            self.logger.info("Refinement failed.")
            
        time_end_refinement = time.time()
        self.logger.info(f"Time: processing {time_end_processing - time_start_processing:.2f}s, refining {time_end_refinement - time_start_refinement:.2f}s")

        # Visualize results
        if self.visualizer:
            self.visualizer.visualize_all(
                scene=scene,
                per_view_info=per_view_info,
                cand_object=cand_object,
                object_pose_wm=object_pose_wm,
                refined_pose_wm=refined_pose_wm,
            )
        del per_view_info

        return refined_pose_wm, 1 - cost.item(), failed

    def refine_scene(
        self,
        scene: Scene,
        candidate_poses: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Refines all candidate poses for the given scene.
        Args:
            scene (Scene): Scene containing multiple views.
            candidate_poses (pd.DataFrame): DataFrame of candidate poses to refine in world coordinates.
        Returns:
            pd.DataFrame: DataFrame of refined candidate poses.
        """

        # In BOP format, all candidates have the same time which is cumulative time over all objects in the scene
        # Summing unique times is equal to summing times from all cameras. 
        unique_times = set(candidate_poses["time"].tolist())
        sum_of_times = sum(unique_times)
        self.timer.start()
        if self.opts.nms:
            #  Non Maximum Suppression (NMS) of the predictions
            filtered_candidates = nms_3d_base.nms_candidates(
                candidates = candidate_poses,
                object_dataset=self.object_database,
                iou_threshold=self.opts.nms_threshold,
                box_type=self.opts.nms_box_type,
                nms_mode=self.opts.nms_mode,
            )
            self.logger.info(
                f"NMS: Filtered candidates: {len(filtered_candidates)} out of {len(candidate_poses)}"
            )
            candidate_poses = filtered_candidates

        # Multiview Pose Refinement
        for i in range(len(candidate_poses)):
            self.logger.info(f"Refining candidate pose {i + 1}/{len(candidate_poses)}")

            candidate = candidate_poses.iloc[i]

            consistent_pose_wm = candidate["pose"]
            object_id = candidate["obj_id"]

            refined_pose_wm, score, failed = self.refine_object_pose(
                scene=scene,
                object_id=object_id,
                object_pose_wm=consistent_pose_wm,
            )

            # Update the candidate pose and score in the DataFrame
            candidate_poses.iloc[i, candidate_poses.columns.get_loc("pose")] = refined_pose_wm
            candidate_poses.iloc[i, candidate_poses.columns.get_loc("score")] = score

        self.logger.info(f"Refinement done. Got {len(candidate_poses)} results.")

        if self.opts.nms:
            # Perform NMS again on the refined poses
            filtered_results = nms_3d_base.nms_candidates(
                candidates=candidate_poses,
                object_dataset=self.object_database,
                iou_threshold=self.opts.nms_threshold,
                box_type=self.opts.nms_box_type,
                nms_mode=self.opts.nms_mode,
            )
            self.logger.info(
                f"NMS 2: Filtered results: {len(filtered_results)} out of {len(candidate_poses)}"
            )
            candidate_poses = filtered_results

        scene_time = self.timer.elapsed("Time for processing a scene")
        self.logger.info(f"Processing time for this scene: {scene_time:.4f} seconds.")

        # Update time stamps
        candidate_poses["time"] = sum_of_times + scene_time

        return candidate_poses