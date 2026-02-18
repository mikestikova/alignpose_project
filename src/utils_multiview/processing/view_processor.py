import os
import gc

from typing import List, NamedTuple, Optional, Tuple

import torch
import numpy as np
import pandas as pd

import cv2

from src.utils_multiview.processing.query_template_repre import TemplateRepre, QueryRepre
from src.utils_multiview.structs.data_structures import View, ObjectInstance
from src.utils_multiview.processing import template_retrieval
from src.utils_multiview.opts import RefineOpts

from foundpose.utils import renderer_builder
from foundpose.utils.renderer_base import RenderType
from foundpose.utils import (
    feature_util,
    misc as misc_util,
    projector_util,
    structs,
)
from foundpose.utils.structs import AlignedBox2f, PinholePlaneCameraModel
from foundpose.utils.misc import array_to_tensor, warp_image


class ViewProcessor:
    def __init__(
        self,
        opts: RefineOpts,
        device: str = "cpu",
    ):
        self.device = device
        self.dataset_name = opts.object_dataset
        self.crop_size = opts.crop_size
        self.crop_rel_pad = opts.crop_rel_pad
        self.grid_cell_size = opts.grid_cell_size
        self.use_offline_templates = opts.use_offline_templates

        # Prepare feature extractor.
        self.extractor = feature_util.make_feature_extractor(opts.extractor_name)
        self.extractor.to(device)

        # Prepare renderer
        renderer_type = renderer_builder.RendererType.PYRENDER_RASTERIZER
        self.renderer = renderer_builder.build(
            renderer_type=renderer_type, model_path=None, device=device
        )


    def _perform_pca_on_feature_map(self, feature_map_chw, repre):
        assert len(repre.feat_raw_projectors) > 0, (
            "No projector found in representation."
        )
        # Potentially project features to a PCA space.
        if (
            feature_map_chw.shape[0] != repre.feat_raw_projectors[0].n_components
            and feature_map_chw.shape[0] > 0
        ):
            _c, _h, _w = feature_map_chw.shape
            feature_map_chw_proj = (
                projector_util.project_features(
                    feat_vectors=feature_map_chw.permute(1, 2, 0).view(-1, _c),
                    projectors=repre.feat_raw_projectors,
                )
                .view(_h, _w, -1)
                .permute(2, 0, 1)
            )
        elif feature_map_chw.shape[0] == 0:
            # If no query features are found, return None
            print("Warning: No query features found")
            return None
        else:
            feature_map_chw_proj = feature_map_chw

        return feature_map_chw_proj

    def process_query_features(
        self,
        image_np_hwc: np.ndarray,
        cand_object: ObjectInstance,
    ) -> QueryRepre:
        """
        Query representation is extracted from a cropped image around the object bounding box.
        """
        # Verify image shape
        assert image_np_hwc.ndim == 3

        # Extract feature map
        image_tensor_chw = (
            array_to_tensor(image_np_hwc)
            .to(torch.float32)
            .permute(2, 0, 1)
            .to(self.device)
        )
        image_tensor_bchw = image_tensor_chw.unsqueeze(0)
        extractor_output = self.extractor(image_tensor_bchw)
        feature_map_chw = extractor_output["feature_maps"][0]

        # Project features to a PCA space if needed.
        feature_map_chw_proj = self._perform_pca_on_feature_map(
            feature_map_chw, cand_object.representation
        )

        # Get the query feature representation
        query = QueryRepre(
            features=feature_map_chw_proj.reshape(
                feature_map_chw_proj.shape[0], -1
            ).transpose(0, 1),
            rgb=(image_np_hwc * 255).astype(np.uint8),
        )

        return query

    def process_template(
        self,
        cand_object: ObjectInstance,
        object_pose_wm: np.ndarray,
        camera: structs.PinholePlaneCameraModel,
        grid_cell_size: int = 14,
    ) -> Optional[TemplateRepre]:
        """
        Template representation is extracted from a rendered image of the object in the cropped view.
        """

        object_pose_cm = np.linalg.inv(camera.T_world_from_eye) @ np.array(
            object_pose_wm
        )

        # Render object in given camera
        render_types = [RenderType.COLOR, RenderType.DEPTH, RenderType.MASK]
        
        # Add to main renderer
        self.renderer.add_object_model(cand_object.object_id, cand_object.model_path)
        rendered_image_template = self.renderer.render_object(
            obj_id=cand_object.object_id,
            pose_m2c=object_pose_cm,
            camera_intrinsics=camera,
            render_types=render_types,
            return_tensors=True,
        )

        # Get feature map from the rendered image
        device = self.device
        projector = cand_object.representation.feat_raw_projectors[0]
        image_chw = (
            rendered_image_template[RenderType.COLOR]
            .permute(2, 0, 1)
            .contiguous()
            .to(device)
        )
        depth_image_hw = rendered_image_template[RenderType.DEPTH].to(device)
        object_mask = rendered_image_template[RenderType.MASK].to(device)

        object_pose_cm = array_to_tensor(object_pose_cm).to(device)
        object_pose_mc = torch.inverse(object_pose_cm).to(torch.float32)

        # image_mask = torch.tensor(image_mask, device=device)
        (
            feat_vectors_full,  # only for visualization shape is (H*W, D)
            masked_features,  # features for the vertices, shape is (N, D)
            _,
            _,
            vertices,  # shape is (N, 3)
        ) = feature_util.get_visual_features_registered_in_3d(
            image_chw=image_chw,
            depth_image_hw=depth_image_hw,
            object_mask=object_mask,
            camera=camera,
            T_model_from_camera=object_pose_mc,
            extractor=self.extractor,
            grid_cell_size=grid_cell_size,
            debug=False,
            # image_mask=image_mask
        )

        # PCA projection
        if masked_features.shape[0] == 0:
            # If no template features are found, return None
            return None
        
        features_proj = projector.transform(feat_vectors_full)
        masked_features_proj = projector.transform(masked_features)

        # Create template representation
        template = TemplateRepre(
            rgb=np.asarray(255.0 * rendered_image_template[RenderType.COLOR], np.uint8),
            depth=np.asarray(rendered_image_template[RenderType.DEPTH], np.uint8),
            features=features_proj,  # used for visualization
            vertices=vertices,
            masked_features=masked_features_proj,
        )

        return template

    def process_template_offline(
        self,
        dataset_name: str,
        cand_object,
        object_pose_wm,
        camera: structs.PinholePlaneCameraModel,
    ) -> TemplateRepre:
        object_pose_cm = np.linalg.inv(camera.T_world_from_eye) @ object_pose_wm

        object_lid = cand_object.object_id
        # Get the best template id based on the consistent pose
        best_template_id = template_retrieval.get_template_by_pose(
            object_pose_cm, object_lid, dataset_name
        )["template_id"]

        # Get the template feature representation
        repre = cand_object.representation
        template_mask = torch.where(repre.feat_to_template_ids == best_template_id)[0]

        # get the vertices for the template, shape is (N,3)
        vertices = repre.vertices[template_mask]
        masked_features = repre.feat_vectors[template_mask]
        projector = cand_object.representation.feat_raw_projectors[0]
        # masked_features_proj = projector.transform(masked_features) ???

        template = TemplateRepre(
            vertices=vertices,
            masked_features=masked_features,
        )
        return template


    def _construct_virtual_camera(
        self,
        orig_camera_c2w: PinholePlaneCameraModel,
        orig_image_np_hwc: np.ndarray,
        orig_box_amodal: AlignedBox2f,
    ) -> Tuple[PinholePlaneCameraModel, np.ndarray]:
        # Get box for cropping.
        crop_box = misc_util.calc_crop_box(
            box=orig_box_amodal,
            make_square=True,
        )

        # Construct a virtual camera focused on the crop.
        crop_camera_model_c2w = misc_util.construct_crop_camera(
            box=crop_box,
            camera_model_c2w=orig_camera_c2w,
            viewport_size=self.crop_size,
            viewport_rel_pad=self.crop_rel_pad,
        )

        # Map images to the virtual camera.
        interpolation = (
            cv2.INTER_AREA
            if crop_box.width >= crop_camera_model_c2w.width
            else cv2.INTER_LINEAR
        )

        crop_image_np_hwc = warp_image(
            src_camera=orig_camera_c2w,
            dst_camera=crop_camera_model_c2w,
            src_image=orig_image_np_hwc,
            interpolation=interpolation,
        )
        return crop_camera_model_c2w, crop_image_np_hwc

    # per_view_info[im_id] = process_view(...)
    def __call__(
        self,
        view: View,
        cand_object: ObjectInstance,
        object_pose_wm: np.ndarray,
    ) -> Optional[dict]:
        """Process one view for multiview refinement."""

        object_pose_cm = np.linalg.inv(view.camera.T_world_from_eye) @ object_pose_wm

        # Get object pose in camera frame
        orig_box_amodal = misc_util.get_object_bbox_in_camera(
            vertices=cand_object.mesh.vertices,
            pose_cm=object_pose_cm,
            camera=view.camera,
        )

        # Construct a virtual camera focused on the crop.
        crop_camera_c2w, crop_image_np_hwc = self._construct_virtual_camera(
            view.camera,
            view.image,
            orig_box_amodal,
        )

        # Extract query representation from the cropped image.
        query = self.process_query_features(
            crop_image_np_hwc,
            cand_object,
        )

        # Extract template representation from the rendered image.
        if self.use_offline_templates:
            template = self.process_template_offline(
                self.dataset_name,
                cand_object,
                object_pose_wm,
                crop_camera_c2w,
            )
        else:
            template = self.process_template(
                cand_object,
                object_pose_wm,
                crop_camera_c2w,
                grid_cell_size=self.grid_cell_size,
            )

        if (
            template is None
            or template.masked_features is None
            or template.masked_features.shape[0] == 0
        ):
            return None, None, None, True

        return crop_camera_c2w, query, template, False