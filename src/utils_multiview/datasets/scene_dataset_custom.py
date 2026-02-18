#!/usr/bin/env python3
"""
Custom scene dataset following BOP multiview directory conventions.

Expected directory layout:
    dataset_root/
    ├── models/
    │   ├── obj_000001.ply
    │   └── ...
    ├── test/
    │   └── 000001/                           # Scene folder per scene
    │       ├── scene_camera_CAMX.json        # Per-image cameras for each sensor
    │       ├── CAMX/                         # Image folder per sensor
    │       │   ├── 000000.png
    │       │   ├── 000001.png
    │       │   └── ...
    │       ├── scene_camera_CAMY.json
    │       ├── CAMY/
    │       │   └── ...
    │       └── ...
    |   └── 000002/
    |       └── ...
    └── test_targets_multiview.json           # Multi-view grouping file

scene_camera_CAMX.json format (keys are image IDs as strings):
    {
        "0": {
            "cam_K": [fx, 0, cx, 0, fy, cy, 0, 0, 1],
            "cam_R_w2c": [r00, r01, ..., r22],
            "cam_t_w2c": [tx, ty, tz],
            "depth_scale": 0.1
        },
        "1": { ... }
    }

test_targets_multiview.json format:
    [
        {
            "scene_id": 1,
            "im_id": [
                ["CAMX", 0],
                ["CAMY", 0],
                ...
            ]
        },
        ...
    ]
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from pathlib import Path

import cv2

from foundpose.utils import data_util
from src.utils_multiview.structs.data_structures import View
from src.utils_multiview.datasets.scene_dataset import SceneDataset
from src.utils_multiview.data.prediction_utils import df_world_to_camera_poses
from utils_multiview.constants import BOP_DATASETS, BOP_DATASETS_PATH


class CustomSceneDataset(SceneDataset):
    """
    Scene dataset for custom data organized in BOP multiview directory conventions.

    Args:
        dataset_name: Name identifier for the dataset.
        logger: Optional logger.
    """

    def __init__(
        self,
        dataset_name: str,
        logger=None,
    ):
        super().__init__(dataset_name, logger)

        if dataset_name not in CUSTOM_DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        self._camera_im_size = CUSTOM_DATASETS[dataset_name]["camera_im_size"]
        self.dataset_root = Path(CUSTOM_DATASETS[dataset_name]["root_dir"])
        self.split_path = self.dataset_root / "test"

    @property
    def camera_im_size(self) -> Dict[str, Tuple[int, int]]:
        return self._camera_im_size

    def load(self):
        """Load dataset: cameras from each scene and view groupings."""
        if not self.split_path.exists():
            raise FileNotFoundError(f"Test split not found at {self.split_path}")

        targets_path = self.dataset_root / "test_targets_multiview.json"
        self.grouping = self._load_multiview_targets(targets_path)
        self.scene_cameras = self._load_scene_cameras()

        self.logger.info("Custom dataset loading complete.")

    def _load_scene_cameras(self) -> Dict:
        """Load per-scene, per-camera camera parameters."""
        scene_ids = {vg.scene_id for vg in self.grouping}
        scene_cameras = {}

        for scene_id in scene_ids:
            scene_cameras[scene_id] = {}
            for camera_id, im_size in self._camera_im_size.items():
                cam_path = (
                    self.split_path
                    / f"{scene_id:06d}"
                    / f"scene_camera_{camera_id}.json"
                )
                if not cam_path.exists():
                    raise FileNotFoundError(
                        f"Camera file not found: {cam_path}"
                    )
                scene_cameras[scene_id][camera_id] = (
                    data_util.load_chunk_cameras(str(cam_path), im_size)
                )

        self.logger.info(
            f"Loaded camera parameters for {len(scene_ids)} scenes."
        )
        return scene_cameras

    def _predictions_to_world(self, predictions: pd.DataFrame, views: list = None) -> pd.DataFrame:
        """Custom dataset predictions are already in world coordinates — no transform needed."""
        return predictions

    def predictions_to_output(self, predictions: pd.DataFrame, views: list = None) -> pd.DataFrame:
        """Convert world-frame predictions to the output format (if any conversion is needed)."""
        return df_world_to_camera_poses(predictions, views)

    def load_view(self, scene_id, im_id, camera_id) -> View:
        if self.scene_cameras is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        image = self._load_image(scene_id, im_id, camera_id)
        camera = self.scene_cameras[scene_id][camera_id][im_id]
        return View(
            camera_id=camera_id,
            camera=camera,
            image_id=im_id,
            image=image,
        )

    def _get_image_path(self, scene_id, im_id, camera_id) -> Path:
        """Resolve the image path for a given scene/image/camera."""
        scene_dir = self.split_path / f"{scene_id:06d}" / camera_id

        for ext in ["png", "jpg", "tif"]:
            path = scene_dir / f"{im_id:06d}.{ext}"
            if path.exists():
                return path

        raise FileNotFoundError(
            f"No image found for scene={scene_id}, im_id={im_id}, "
            f"camera={camera_id} in {scene_dir}"
        )

    def _load_image(self, scene_id, im_id, camera_id) -> np.ndarray:
        """Load and normalize an image."""
        im_path = self._get_image_path(scene_id, im_id, camera_id)

        image = cv2.imread(str(im_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise IOError(f"Failed to load image: {im_path}")

        # BGR to RGB (for 3-channel images)
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return self._normalize_image(image)



# HERE SPECIFY THE DETAILS OF YOUR CUSTOM DATASET
CUSTOM_DATASETS = {
    "custom1": {
        "root_dir": BOP_DATASETS_PATH / "custom1",
        "camera_im_size": {
            "cam0": (640, 480),
            "cam1": (640, 480),
            "cam2": (640, 480),
        },

    }
}