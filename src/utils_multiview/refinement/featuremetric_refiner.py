from typing import List, Tuple
from torch import Tensor
import torch

from foundpose.utils.structs import PinholePlaneCameraModel, ObjectPose
from foundpose.utils import misc

from pixloc.pixlib.models.multiview_optimizer import ClassicMultiviewOptimizer
from pixloc.pixlib.geometry import Camera, Pose

def refine_multiview(
    initial_pose_m2w: ObjectPose,
    template_vertices_ref: List[Tensor],
    template_masked_features_ref: List[Tensor],
    feature_map_chw_proj_ref: List[Tensor],
    cameras: List[PinholePlaneCameraModel],
    num_iters: int = 30,
    loss_fn="scaled_barron(-5, 0.5)",
) -> Tuple[Pose, Tensor]:
    """
    Refine the pose using the ClassicOptimizer. M is number of views, N is number of registered features per view.
    Args:
        initial_pose_m2w: Initial pose in world coordinates.
        template_vertices_ref: Template vertices. For each view shape (N, 3).
        template_masked_features_ref: Template registered features. For each view shape (N, C).
        feature_map_chw_proj_ref: Query feature map. For each view shape (C, H, W). Heights and widths are the same as image size.
        cameras: Camera models, len=M.
    Returns:
        Tuple[Pose, Tensor]: Optimized pose and failure flag.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert initial pose to Pixloc format
    initial_pose_m2c_array = misc.get_rigid_matrix(initial_pose_m2w)
    initial_pose_m2c_tensor = torch.tensor(initial_pose_m2c_array, dtype=torch.float32)
    initial_pose_m2c_pose = Pose.from_4x4mat(initial_pose_m2c_tensor.unsqueeze(0)).to(
        device
    )

    # Convert all list of tensors to padded, for batch processing
    mask = []
    max_len = max([vertices.shape[0] for vertices in template_vertices_ref])
    for i, (vertices, features) in enumerate(
        zip(template_vertices_ref, template_masked_features_ref)
    ):
        # Pad the vertices and features to the same length
        pad_len = max_len - vertices.shape[0]
        if pad_len > 0:
            template_vertices_ref[i] = torch.nn.functional.pad(
                vertices, (0, 0, 0, pad_len), value=0
            )
            template_masked_features_ref[i] = torch.nn.functional.pad(
                features, (0, 0, 0, pad_len), value=0
            )
            mask.append(
                torch.tensor(
                    [True] * vertices.shape[0] + [False] * pad_len, dtype=torch.bool
                )
            )
        else:
            template_vertices_ref[i] = vertices[:max_len]
            template_masked_features_ref[i] = features[:max_len]
            mask.append(torch.ones(max_len, dtype=torch.bool))

    template_vertices_ref = torch.stack(template_vertices_ref, dim=0)  # Shape (M, N, 3)
    template_masked_features_ref = torch.stack(
        template_masked_features_ref, dim=0
    )  # Shape (M, N, C)
    # Convert feature maps to tensor for batch processing
    feature_map_chw_proj_ref = torch.stack(
        feature_map_chw_proj_ref, dim=0
    )  # Shape (M, C, H, W)
    mask = torch.stack(mask, dim=0).to(
        device=template_vertices_ref.device
    )  # Shape (M, N)

    feature_map_chw_proj_ref = feature_map_chw_proj_ref.detach().requires_grad_(False)

    # Convert cameras to Pixloc format
    cam_intrinsics = []
    cam_poses = []
    for camera_c2w in cameras:
        camera_intrinsic = torch.tensor(
            [
                *(camera_c2w.width, camera_c2w.height),
                camera_c2w.f[0],
                camera_c2w.f[1],
                camera_c2w.c[0],
                camera_c2w.c[1],
            ],
            dtype=torch.float32,
        ).to(device)
        cam_intrinsics.append(camera_intrinsic)

        camera_pose_c2w = camera_c2w.T_world_from_eye
        camera_pose_c2w = torch.tensor(camera_pose_c2w, dtype=torch.float32).to(device)
        cam_poses.append(torch.linalg.inv(camera_pose_c2w))

    camera_objects = Camera(data=torch.stack(cam_intrinsics)).to(device)
    camera_poses_w2c = Pose.from_4x4mat(torch.stack(cam_poses))

    # Create an instance of ClassicOptimizer
    conf = {
        "num_iters": num_iters,
        "lambda_": 1e-2,
        "lambda_max": 1e4,
        "normalize_features": True,
        "jacobi_scaling": False,
        "interpolation": dict(
            mode="linear",
            pad=4,
        ),
        "loss_fn": loss_fn,
    }
    optimizer = ClassicMultiviewOptimizer(conf)
    optimizer.eval()

    # Run the optimizer
    try:
        T_pose, failed, cost_seq = optimizer.run(
            p3D=template_vertices_ref,
            F_ref=template_masked_features_ref,
            F_query=feature_map_chw_proj_ref,
            T_init_wo=initial_pose_m2c_pose,
            T_cw=camera_poses_w2c,
            cameras=camera_objects,
            mask=mask,
        )
    except Exception as e:
        print("Optimization failed with exception:", e)
        return initial_pose_m2w, True, None

    # Check if the optimization failed
    if failed:
        return initial_pose_m2w, failed, None

    # Convert the optimized pose back to ObjectPose
    optimized_pose_m2w = ObjectPose(
        R=T_pose.R.squeeze().detach().cpu(), t=T_pose.t.squeeze().detach().cpu()
    )

    return optimized_pose_m2w, failed, cost_seq
