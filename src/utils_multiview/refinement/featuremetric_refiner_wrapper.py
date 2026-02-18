from typing import List, Tuple
from torch import Tensor
import numpy as np
import torch

from foundpose.utils.structs import PinholePlaneCameraModel
from foundpose.utils import structs, misc
from src.utils_multiview.processing.query_template_repre import TemplateRepre
from src.utils_multiview.refinement.featuremetric_refiner import refine_multiview


def queries_to_tensor(queries: List, img_size: Tuple) -> Tuple[Tensor, Tensor]:
    feature_map_chw_proj_ref = []
    for query in queries:
        patch_size = int(np.sqrt(query.features.shape[0]))
        feature_map = query.features.T.reshape(-1, patch_size, patch_size).unsqueeze(0)
        feature_map = torch.nn.functional.interpolate(
            feature_map,
            (img_size[0], img_size[1]),
            mode="bilinear",
            align_corners=False,
        )
        feature_map_chw_proj_ref.append(feature_map.squeeze(0))
    return feature_map_chw_proj_ref

def refine_multiview_wrapper(
    initial_pose_wm: structs.RigidTransform,
    templates: List[TemplateRepre],
    queries: List[TemplateRepre],
    cameras: List[PinholePlaneCameraModel],
    num_iter: int = 30,
    loss_fn: str = "scaled_barron(-5, 0.5)",
) -> Tuple[np.array, bool, List[float]]:
    """
    Refine the pose using the Multiview featuremetric refiner.
    Args:
        initial_pose_wm: Initial obj pose in world coordinates.
        templates: List of templates. Each template contains vertices and features.
        queries: List of queries. Each query contains features.
        cameras: List of camera models.
    Returns:
        Tuple[np.array, bool, List[float]]: Optimized pose, failure flag, and cost sequence.
    """

    # Assume all cameras have the same size (for query upscaling)
    img_size = (cameras[0].width, cameras[0].height)

    # Get data from templates
    template_vertices_ref = [template.vertices for template in templates]
    template_masked_features_ref = [template.masked_features for template in templates]

    # Get data from queries (need to upscale the feature map to the image size), remove grad
    feature_map_chw_proj_ref = queries_to_tensor(queries, img_size)

    # Refine
    optimized_pose_m2w, failed, cost_seq = refine_multiview(
        initial_pose_m2w=initial_pose_wm,
        template_vertices_ref=template_vertices_ref,
        template_masked_features_ref=template_masked_features_ref,
        feature_map_chw_proj_ref=feature_map_chw_proj_ref,
        cameras=cameras,
        num_iters=num_iter,
        loss_fn=loss_fn,
    )

    return optimized_pose_m2w, failed, cost_seq
