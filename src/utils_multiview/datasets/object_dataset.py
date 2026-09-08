#!/usr/bin/env python3
import os
from typing import List, Dict
import trimesh
from tqdm import tqdm

from bop_toolkit_lib import dataset_params
import bop_toolkit_lib.config as bop_config

from foundpose.utils import (
    repre_util,
    projector_util
)
from src.utils_multiview.structs.data_structures import ObjectInstance
from src.utils_multiview.constants import BOP_DATASETS, BOP_DATASETS_PATH, OUTPUT_DIR

class ObjectDataset():
    """
    Stores information about objects in an object dataset, including their meshes and PCA representations.
    """
    def __init__(
        self, dataset_name: str, model_tpath: str, object_ids: List[int] = None,
    ):    
        """
        Args:
            dataset_name: Name of the dataset, e.g. "ycbv".
            model_tpath: The path to the object models. Has to be a format string with a placeholder for the object ID, e.g. `path/to/models/{obj_id}.ply`. Path must exist for all object IDs in `object_ids`.
            object_ids: Optional list of object IDs in the dataset.
        """   
        self.dataset_name = dataset_name
        self.model_tpath = model_tpath
        self.object_ids = object_ids

        # If not specified explicitly add all object ids from the model path  
        if self.object_ids is None:
            model_dir = os.path.dirname(self.model_tpath)
            self.object_ids = []
            for file in os.listdir(model_dir):
                if file.endswith(".ply"):
                    obj_id = os.path.splitext(file)[0]
                    self.object_ids.append(obj_id)


        # Load object meshes and create ObjectInstance dataclasses for each object.
        self.load_meshes()

    @property
    def objects(self) -> Dict[int, ObjectInstance]:
        """
        Returns a dictionary mapping object IDs to ObjectInstance dataclasses
        containing mesh and representation information.
        """
        return self._objects

    def load_meshes(self):
        objects = {}
        for object_id in tqdm(self.object_ids):
            model_path = self.model_tpath.format(obj_id=object_id)
            objects[object_id] = ObjectInstance(
                object_id=object_id,
                model_path=model_path,
                mesh=self.get_object_mesh(model_path),
                representation=None,
            )
        self._objects = objects

    def get_object_mesh(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return trimesh.load_mesh(model_path)
    
    def load_representations(
        self, repre_version, device, full: bool = False, output_root: str = None
    ):
        """Loads a representation for every object in the dataset.

        Args:
            repre_version: Version of the representation to load.
            device: Device the projectors are placed on.
            full: See `get_object_representation`.
            output_root: Root holding `object_repre/` (default: `OUTPUT_DIR`).
        """
        for object_id in self.object_ids:
            repre = self.get_object_representation(
                object_id, repre_version, device, full=full, output_root=output_root
            )
            self.objects[object_id].representation = repre

    def get_object_representation(
        self, object_id, repre_version, device, full: bool = False, output_root: str = None
    ):
        """Loads one object's representation.

        Args:
            full: If False (default) only the PCA projectors are read, which is all
                the multi-view refiner needs. Single-view coarse pose estimation
                additionally needs the template cameras and the tf-idf descriptor
                options, so it must pass True.
            output_root: Root holding `object_repre/` (default: `OUTPUT_DIR`).
        """
        base_repre_dir = os.path.join(output_root or OUTPUT_DIR, "object_repre")
        repre_dir = repre_util.get_object_repre_dir_path(
            base_repre_dir, repre_version, self.dataset_name, object_id
        )
        repre = repre_util.load_object_repre(
            repre_dir=repre_dir,
            tensor_device=device,
            load_fields=None
            if full
            else [
                "feat_raw_projectors",
                "feat_vis_projectors",
            ],
        )

        if repre is None:
            raise FileNotFoundError(
                f"Representation not found for object {object_id} in {repre_dir}"
            )

        raw_projector = repre.feat_raw_projectors[0]
        raw_torch_projector = self._foundpose_projector_to_torchprojector(
            raw_projector, device
        )
        repre.feat_raw_projectors = [raw_torch_projector]

        vis_projector = repre.feat_vis_projectors[0]
        vis_torch_projector = self._foundpose_projector_to_torchprojector(
            vis_projector, device
        )
        repre.feat_vis_projectors = [vis_torch_projector]

        return repre

    def _foundpose_projector_to_torchprojector(self, projector, device):
        if isinstance(projector, projector_util.PCAProjector):
            torch_projector = projector_util.TorchPCAProjector(
                n_components=projector.n_components,
                whiten=projector.whiten,
                device=device,
            )
            torch_projector.fit_from_numpy(projector.pca)
            return torch_projector
        else:
            raise NotImplementedError(
                f"Conversion for projector type {type(projector)} not implemented."
            )

    @staticmethod
    def from_bop(dataset_name: str) -> "ObjectDataset":
        assert dataset_name in BOP_DATASETS, f"Unsupported BOP dataset: {dataset_name}. Allowed datasets: {BOP_DATASETS}"

        # Get dataset path from bop_toolkit_lib config
        datasets_path = BOP_DATASETS_PATH

        model_params = dataset_params.get_model_params(
            datasets_path=datasets_path,
            dataset_name=dataset_name,
        )
        object_ids = model_params["obj_ids"]
        model_tpath = model_params["model_tpath"]

        return ObjectDataset(
            dataset_name=dataset_name,
            model_tpath=model_tpath,
            object_ids=object_ids,
        )

def make_object_dataset(dataset_name: str, model_tpath: str = None, object_lids: list = None) -> ObjectDataset:
    if dataset_name in BOP_DATASETS:
        # Get dataset path from bop_toolkit_lib config
        datasets_path = BOP_DATASETS_PATH

        model_params = dataset_params.get_model_params(
            datasets_path=datasets_path,
            dataset_name=dataset_name,
        )
        object_ids = model_params["obj_ids"]
        if model_tpath is None:
            model_tpath = model_params["model_tpath"]

        return ObjectDataset(
            dataset_name=dataset_name,
            model_tpath=model_tpath,
            object_ids=object_ids,
        )
    else:
        assert model_tpath is not None, "For custom datasets, model_tpath must be provided."
        return ObjectDataset(
            dataset_name=dataset_name,
            model_tpath=model_tpath,
            object_ids=object_lids,
        )