#!/usr/bin/env python3
from dataclasses import dataclass
from typing import List

import numpy as np
import trimesh
from foundpose.utils import (
    repre_util,
    structs,
)


@dataclass
class ObjectInstance():
    object_id: int
    model_path: str
    mesh: trimesh.Trimesh
    representation: repre_util.FeatureBasedObjectRepre

@dataclass(frozen=True)
class View():
    camera_id: int
    camera: structs.PinholePlaneCameraModel
    image_id: int
    image: np.ndarray

@dataclass
class Scene():
    scene_id: int
    group_id: int
    views: List[View]