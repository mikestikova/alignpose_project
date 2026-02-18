#!/usr/bin/env python3
from typing import Dict, Tuple
from pathlib import Path
import pandas as pd


from bop_toolkit_lib import dataset_params, inout
import bop_toolkit_lib.config as bop_config
from foundpose.utils import data_util
from src.utils_multiview.structs.data_structures import View
from src.utils_multiview.datasets.scene_dataset import SceneDataset
from src.utils_multiview.data.prediction_utils import (
    df_camera_to_world_poses,
    df_world_to_camera_poses,
)
from src.utils_multiview.constants import BOP_DATASETS, BOP_DATASETS_PATH, BOP_DATASETS_PATH

class BOPSceneDataset(SceneDataset):
    """
    BOP-specific implementation of SceneDataset.
    """

    def __init__(self, dataset_name, logger=None):
        super().__init__(dataset_name, logger)
        
        self.scene_cameras = None
        self._camera_im_size = None

    @property
    def camera_im_size(self) -> Dict[str, Tuple[int, int]]:
        if self._camera_im_size is None:
            raise ValueError("Camera image sizes not loaded. Call load() first.")
        return self._camera_im_size

    def load(self, split="test"):
        if split not in ["test"]:
            raise ValueError(f"Unsupported split: {split}")

        # Get properties of the split of the specified dataset.
        self.logger.info(f"Loading BOP {split} split parameters...")

        self.bop_test_split_props = get_bop_test_split_props(self.dataset_name)

        self._camera_im_size = self.bop_test_split_props["im_size"]
        self.scene_cameras = self._get_scene_cameras()

        path = Path(
            f"./data/bop_test_targets/{self.dataset_name}/test_targets_multiview_bop25.json"
        )
        self.grouping = self._load_multiview_targets(path)

        self.logger.info("Dataset loading complete.")

    def _predictions_to_world(self, predictions: pd.DataFrame, views: list) -> pd.DataFrame:
        """BOP predictions are in camera coordinates — transform to world."""
        return df_camera_to_world_poses(predictions, views)

    def predictions_to_output(self, predictions: pd.DataFrame, views: list) -> pd.DataFrame:
        """Convert world-frame predictions back to per-camera coordinates for output."""
        return df_world_to_camera_poses(predictions, views)

    def _camera_path(self, scene_id, camera_id, modality):
        return self.bop_test_split_props[
            f"scene_camera_{modality}_{camera_id}_tpath"
        ].format(scene_id=scene_id)

    def _get_scene_cameras(self):
        self.logger.info(
            f"Loaded scene cameras poses for dataset {self.dataset_name}..."
        )
        scene_cameras = {}
        for scene_id in self.bop_test_split_props["scene_ids"]:
            scene_cameras[scene_id] = {}

            for camera_id, modalities in self.bop_test_split_props[
                "im_modalities"
            ].items():
                cam_path = self._camera_path(scene_id, camera_id, modalities[0])
                scene_cameras[scene_id][camera_id] = data_util.load_chunk_cameras(
                    cam_path,
                    self.camera_im_size[camera_id],
                )
        return scene_cameras

    def _get_camera_pose(self, scene_id, im_id, camera_id=None):
        if self.scene_cameras is None:
            raise ValueError("Scene cameras not loaded. Call load() first.")
        return self.scene_cameras[scene_id][camera_id][im_id]

    def _get_image_path(self, scene_id, im_id, camera_id):
        modality = self.bop_test_split_props["im_modalities"][camera_id][0]
        return self.bop_test_split_props[f"{modality}_{camera_id}_tpath"].format(
            scene_id=scene_id, im_id=im_id
        )

    def _load_image(self, scene_id, im_id, camera_id="default"):
        if self.bop_test_split_props is None:
            raise ValueError("Dataset split not loaded. Call load() first.")

        im_path = self._get_image_path(scene_id, im_id, camera_id)
        image = inout.load_im(im_path)
        return self._normalize_image(image)

    def load_view(self, scene_id, im_id, camera_id="default"):
        image = self._load_image(scene_id, im_id, camera_id)
        camera = self._get_camera_pose(scene_id, im_id, camera_id)
        return View(
            camera_id=camera_id,
            camera=camera,
            image_id=im_id,
            image=image,
        )
    
def get_bop_test_split_props(dataset_name):
    """Get properties of the test split for the specified BOP dataset."""
    if dataset_name not in BOP_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    bop_test_split_props = dataset_params.get_split_params(
        datasets_path=BOP_DATASETS_PATH,
        dataset_name=dataset_name,
        split="test",
    )

    # Convert BOP Classic datasets to expected format multiview datasets
    if dataset_name == "ycbv" or dataset_name == "tless":
        bop_test_split_props["eval_sensor"] = "default"
        bop_test_split_props["eval_modality"] = "rgb"
        bop_test_split_props["im_size"] = {"default": (640,480)}
        modalities = bop_test_split_props["im_modalities"]
        bop_test_split_props["im_modalities"] = {"default": modalities}
        bop_test_split_props["scene_camera_rgb_default_tpath"] = bop_test_split_props["scene_camera_tpath"]
        bop_test_split_props[f"rgb_default_tpath"] = bop_test_split_props["rgb_tpath"]
        return bop_test_split_props
    
    elif dataset_name == "xyzibd":
        bop_test_split_props["im_size"] = {
            "xyz": (1440, 1080)
        }
        return bop_test_split_props

    else: # ipd, itoddmv
        return bop_test_split_props
