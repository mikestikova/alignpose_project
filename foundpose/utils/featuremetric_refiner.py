from pixloc.pixlib.models.classic_optimizer import ClassicOptimizer
from pixloc.pixlib.geometry import Camera, Pose
from typing import List, Tuple
from torch import Tensor
from foundpose.utils.structs import PinholePlaneCameraModel, ObjectPose
import numpy as np
import torch
from foundpose.utils import misc


def refine(
    initial_pose_m2c: ObjectPose,
    template_vertices_ref: Tensor,
    template_masked_features_ref: Tensor,
    feature_map_chw_proj_ref: Tensor,
    camera_c2w: PinholePlaneCameraModel,
    num_iters: int = 30,
) -> Tuple[ObjectPose, Tensor]:
    """
    Refine the pose using the ClassicOptimizer.
    Args:
        initial_pose_m2c: Initial pose.
        template_vertices_ref: Template vertices. Shape (1, N, 3).
        template_masked_features_ref: Template masked features. Shape (1, N, C).
        feature_map_chw_proj_ref: Query feature map. Shape (1, C, H, W). Heights and widths are the same as image size.
        camera_c2w: Camera model. Only intrinsics matter. Shape (1, 4, 4).
    Returns:
        Tuple[Pose, Tensor]: Optimized pose and failure flag.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert initial pose to Pixloc format
    initial_pose_m2c_array = misc.get_rigid_matrix(initial_pose_m2c)
    initial_pose_m2c_tensor = torch.tensor(initial_pose_m2c_array, dtype=torch.float32)
    initial_pose_m2c_pose = Pose.from_4x4mat(initial_pose_m2c_tensor.unsqueeze(0)).to(
        device
    )

    # Convert camera to Pixloc format
    camera_intrinsic = torch.tensor(
        [
            camera_c2w.width,
            camera_c2w.height,
            camera_c2w.f[0],
            camera_c2w.f[1],
            camera_c2w.c[0],
            camera_c2w.c[1],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)
    camera = Camera(data=camera_intrinsic).to(device)

    # Create an instance of ClassicOptimizer
    conf = {
        "num_iters": 30,
        "lambda_": 1e-2,
        "lambda_max": 1e4,
        "normalize_features": True,
        "jacobi_scaling": False,
        "interpolation": dict(
            mode="linear",
            pad=4,
        ),
        "loss_fn": "scaled_barron(-5, 0.5)",
        "num_iters": num_iters,
    }

    # Optimize the pose
    optimizer = ClassicOptimizer(conf)
    try:
        optimized_pose, failed = optimizer.run(
            p3D=template_vertices_ref,
            F_ref=template_masked_features_ref,
            F_query=feature_map_chw_proj_ref,
            T_init=initial_pose_m2c_pose,
            camera=camera,
        )
    except Exception as e:
        print("Optimization failed with exception:", e)
        return initial_pose_m2c, True

    # Check if the optimization failed
    if failed:
        return initial_pose_m2c, failed

    # Convert the optimized pose back to ObjectPose
    optimized_pose = ObjectPose(
        R=optimized_pose.R.squeeze().detach().cpu(),
        t=optimized_pose.t.squeeze().detach().cpu(),
    )

    return optimized_pose, failed


