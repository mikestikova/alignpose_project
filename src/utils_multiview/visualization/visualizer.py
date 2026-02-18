# This is experimental code.
#!/usr/bin/env python3

"""Infers pose from objects."""

import os

from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import json
import trimesh


from src.utils_multiview.visualization import visualization_util
from src.utils_multiview.processing.query_template_repre import TemplateRepre, QueryRepre
from src.utils_multiview.datasets.object_dataset import ObjectDataset
from src.utils_multiview.structs.data_structures import Scene, View, ObjectInstance

from bop_toolkit_lib import inout

from foundpose.utils import renderer_builder
from foundpose.utils import (
    misc as misc_util,
    vis_util,
    misc,
    structs,
)

class Visualizer:
    """
    Visualization helper.
    """
    def __init__(self, camera_im_sizes, path_to_store_vis: str):
        os.makedirs(path_to_store_vis, exist_ok=True)
        self.path_to_store_vis = path_to_store_vis
        self.vis_renderers = {}
        self.counter = 0

        self.initialize_renderers(camera_im_sizes=camera_im_sizes)


    def initialize_renderers(self, camera_im_sizes):
        camera_renderers = {}
        renderer_type = renderer_builder.RendererType.PYRENDER_RASTERIZER
        for cam, _ in camera_im_sizes.items():
            vis_renderer = renderer_builder.build(renderer_type=renderer_type)
            camera_renderers[cam] = vis_renderer
        self.vis_renderers.update(camera_renderers)
        
    def add_object_models(self, object_database: ObjectDataset):
        # Add all object models to the renderers
        for object_id in object_database.object_ids:
            model_path = object_database.objects[object_id].model_path
            for renderer in self.vis_renderers.values():
                renderer.add_object_model(object_id, model_path)

    def visualize_refinement(
        self,
        views: List[View],
        object_id: int,
        before_pose: structs.RigidTransform,
        after_pose: structs.RigidTransform,
        save_path: str,
    ):
        
        visualization_util.visualize_refinement_results(
            views,
            object_id,
            misc.get_rigid_matrix(before_pose),
            misc.get_rigid_matrix(after_pose),
            self.vis_renderers,
            save_path,
        )

    def visualize_template(self, template: TemplateRepre, save_path: str):
        # Store the rendered template image and depth
        rgb_image = template.rgb

        inout.save_im(f"{save_path}/template_img.png", rgb_image)

        # Visualize PCA projected feature map
        patch_size = int(np.sqrt(template.features.shape[0]))
        feat_vectors_full_reshaped = (
            template.features.reshape(patch_size, patch_size, -1)
            .permute(2, 0, 1)
            .contiguous()
        )  # features shape was (H*W, D)

        vis_pca_features = vis_util.vis_pca_feature_map_projected(
            feat_vectors_full_reshaped,
            image_height=420,
            image_width=420,
        )

        inout.save_im(
            f"{save_path}/template_features.png", vis_pca_features
        )

        # Store the pointcloud with features as vertex colors for visualization in meshlab
        vertices = template.vertices.cpu().numpy().astype(np.float32)
        max = template.features.max()
        min = template.features.min()

        normalized_features = (template.masked_features - min) / (max - min) * 255
        colors = normalized_features[:, :3].cpu().numpy().astype(np.uint8)

        pointcloud = trimesh.PointCloud(vertices, colors=colors)
        pointcloud.export(f"{save_path}/template_pointcloud.ply")
       

    def visualize_query(self, query: QueryRepre, save_path: str):
        # Store the query image
        inout.save_im(f"{save_path}/query_img.png", query.rgb)

        # Visualize PCA projected feature map
        patch_size = int(np.sqrt(query.features.shape[0]))
        feat_vectors_full_reshaped = (
            query.features.reshape(patch_size, patch_size, -1)
            .permute(2, 0, 1)
            .contiguous()
        ) # features shape was (H*W, D)

        vis_pca_features = vis_util.vis_pca_feature_map_projected(
            feat_vectors_full_reshaped,
            image_height=query.rgb.shape[0],
            image_width=query.rgb.shape[1],
        )
        inout.save_im(f"{save_path}/query_features.png", vis_pca_features)

    def visualize_view_info(self, view_info: dict, save_path: str):
        self.visualize_template(view_info["template"], save_path=save_path)
        self.visualize_query(view_info["query"], save_path=save_path)

        # Json serialization does not support numpy types, convert to native python types
        crop_camera_c2w = view_info["virtual_camera"]
        crop_camera_dict = {
            "fx": crop_camera_c2w.f[0].item(),
            "fy": crop_camera_c2w.f[1].item(),
            "cx": crop_camera_c2w.c[0].item(),
            "cy": crop_camera_c2w.c[1].item(),
            "width": crop_camera_c2w.width,
            "height": crop_camera_c2w.height,
            "T_world_from_eye": crop_camera_c2w.T_world_from_eye.tolist(),
        }
        with open(f"{save_path}/virtual_camera.json", "w") as f:
            json.dump(crop_camera_dict, f, indent=4)

    def visualize_all(
        self,
        scene: Scene,
        per_view_info: dict,
        cand_object: ObjectInstance,
        object_pose_wm: structs.RigidTransform,
        refined_pose_wm: structs.RigidTransform,
        debug: bool = False,
    ):
        obj_path = f"group_{scene.group_id}_scene_{scene.scene_id}_object_{cand_object.object_id}_{self.counter}"
        self.counter += 1
        output_path = os.path.join(self.path_to_store_vis, obj_path)
        os.makedirs(output_path, exist_ok=True)

        if debug:
            print(f"Visualizing object {cand_object.object_id} at {output_path}")
            # Visualize each view's template and query
            for (camera_id, view_id), view_info in per_view_info.items():
                view_path = os.path.join(output_path, f"view_{camera_id}_{view_id}")
                os.makedirs(view_path, exist_ok=True)
                self.visualize_view_info(
                    view_info=view_info,
                    save_path=view_path
                )

        # Visualize pre vs post refinement poses
        self.visualize_refinement(
            views=scene.views,
            object_id=cand_object.object_id,
            before_pose=object_pose_wm,
            after_pose=refined_pose_wm,
            save_path=output_path
        )