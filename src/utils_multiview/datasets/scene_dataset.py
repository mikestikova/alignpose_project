#!/usr/bin/env python3
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator, List, Tuple, Dict

import numpy as np
import pandas as pd

from foundpose.utils import logging
from src.utils_multiview.structs.data_structures import Scene, View
from src.utils_multiview.constants import BOP_DATASETS


@dataclass(frozen=True)
class ViewGroup:
    scene_id: int
    group_id: int
    view_info: List[Tuple[str, int]]  # (camera_id, im_id)


class SceneDataset(ABC):
    """
    Abstract interface for a multi-view scene dataset.

    Responsible for:
    - Providing scene batches containing multiple views.
    - Converting predictions to/from world coordinates for refinement.
    """

    def __init__(self, dataset_name: str, logger=None):
        self.dataset_name = dataset_name
        self.logger = logger or logging.get_logger(level=logging.WARNING)
        self.grouping: List[ViewGroup] = None

    @property
    @abstractmethod
    def camera_im_size(self) -> Dict[str, Tuple[int, int]]:
        """Must return a dictionary mapping camera IDs to image sizes."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load dataset metadata."""
        pass

    def __iter__(self) -> Iterator[Scene]:
        """Yield Scene objects containing information about the scenes and their views."""
        if self.grouping is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        for view_group in self.grouping:
            views = []
            for camera_id, im_id in view_group.view_info:
                view = self.load_view(
                    scene_id=view_group.scene_id,
                    im_id=im_id,
                    camera_id=camera_id,
                )
                views.append(view)

            yield Scene(
                scene_id=view_group.scene_id,
                group_id=view_group.group_id,
                views=views,
            )

    def __len__(self) -> int:
        """Return the number of scenes in the dataset."""
        if self.grouping is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        return len(self.grouping)

    def _load_multiview_targets(self, path: Path) -> List[ViewGroup]:
        """Load multi-view groupings from a JSON file.

        Args:
            path: Path to the multiview targets JSON file.
        Returns:
            List of ViewGroup objects.
        """
        if not path.exists():
            raise FileNotFoundError(f"Multiview targets file not found: {path}")

        with open(path, "r") as f:
            groups = json.load(f)

        grouping = []
        for group_id, group in enumerate(groups):
            grouping.append(
                ViewGroup(
                    scene_id=group["scene_id"],
                    group_id=group_id,
                    view_info=[(view[0], view[1]) for view in group["im_id"]],
                )
            )

        self.logger.info(f"Loaded multiview targets from {path} with {len(grouping)} groups")
        return grouping

    @staticmethod
    def _normalize_image(image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] float32 RGB.

        Handles grayscale→RGB conversion and safe min-max normalization.
        """
        image = image.astype(np.float32)

        # Normalize to [0, 1]
        im_min, im_max = image.min(), image.max()
        if im_max > im_min:
            image = (image - im_min) / (im_max - im_min)
        else:
            image = np.zeros_like(image)

        # Convert grayscale to RGB
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            image = np.stack((image.squeeze(),) * 3, axis=-1)

        return image

    def yield_scenes_with_predictions(self, predictions: pd.DataFrame) -> Iterator[Tuple[Scene, pd.DataFrame]]:
        """
        Yield (Scene, candidates_in_world_frame) pairs ready for multi-view refinement.

        Matches predictions to scenes by scene_id and im_id, then converts
        candidates to world coordinates via _predictions_to_world().

        Args:
            predictions (pd.DataFrame): Predicted object poses. Must contain
                'scene_id', 'im_id', and 'pose' columns, may contain others, e.g. 'camera_id' if needed for conversion to world frame.
        Yields:
            (Scene, pd.DataFrame): Scene and its candidates with poses in world frame.
        """
        for scene in self:
            view_ids = {view.image_id for view in scene.views}
            candidates = predictions[
                (predictions["scene_id"] == scene.scene_id)
                & (predictions["im_id"].isin(view_ids))
            ].reset_index(drop=True)

            self.logger.info(
                f"Group {scene.group_id}/{len(self)} in scene {scene.scene_id} images "
                f"{[(view.camera_id, view.image_id) for view in scene.views]} has {len(candidates)} candidates."
            )

            if candidates.empty:
                continue

            # Convert to world frame for refinement
            candidates_world = self._predictions_to_world(candidates, scene.views)
            yield scene, candidates_world

    @abstractmethod
    def _predictions_to_world(self, predictions: pd.DataFrame, views: list) -> pd.DataFrame:
        """
        Convert input predictions to world coordinates for multi-view refinement.

        Args:
            predictions: Matched candidates for a scene.
            views: The scene's views (contain camera extrinsics).
        Returns:
            Predictions with 'pose' in world coordinates.
        """
        pass

    @abstractmethod
    def predictions_to_output(self, predictions: pd.DataFrame, views: list) -> pd.DataFrame:
        """
        Convert world-frame predictions to the output format.

        Args:
            predictions: Refined predictions with 'pose' in world coordinates.
            views: The scene's views (contain camera extrinsics).
        Returns:
            Predictions in the dataset's output coordinate convention.
        """
        pass

    @abstractmethod
    def load_view(self, scene_id, im_id, camera_id) -> View:
        """
        Load a single view given the scene ID, image ID and camera ID.
        Returns a View dataclass containing the camera parameters and image.
        """
        pass

def make_scene_dataset(
    dataset_name: str,
    logger=None,
) -> SceneDataset:

    # BOP challenge splits
    if dataset_name in BOP_DATASETS:
        from src.utils_multiview.datasets.scene_dataset_bop import BOPSceneDataset
        return BOPSceneDataset(dataset_name=dataset_name, logger=logger)
    # Custom dataset (expects a specific directory structure and multiview targets file)
    elif dataset_name == "custom":
        from src.utils_multiview.datasets.scene_dataset_custom import CustomSceneDataset
        return CustomSceneDataset(dataset_name=dataset_name, logger=logger)