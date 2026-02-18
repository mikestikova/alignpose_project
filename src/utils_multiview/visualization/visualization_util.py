#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio
import cv2
from typing import Dict, List, Any, Tuple, Optional
from src.utils_multiview.structs.data_structures import ObjectInstance, View


from foundpose.utils import repre_util, vis_util
from foundpose.utils import geometry, renderer_builder
from foundpose.utils.structs import PinholePlaneCameraModel
from foundpose.utils.renderer_base import RenderType
from src.utils_multiview.constants import OUTPUT_DIR


def get_mask_from_rgb(img: np.ndarray) -> np.ndarray:
    img_t = torch.as_tensor(img)
    mask = torch.zeros_like(img_t)
    mask[img_t > 0] = 255
    mask = torch.max(mask, dim=-1)[0]
    mask_np = mask.numpy().astype(np.bool_)
    return mask_np


def make_contour_overlay(
    img: np.ndarray,
    render: np.ndarray,
    color: Optional[Tuple[int, int, int]] = None,
    dilate_iterations: int = 1,
) -> Dict[str, Any]:
    # megapose

    if color is None:
        color = (0, 255, 0)

    mask_bool = get_mask_from_rgb(render)
    mask_uint8 = (mask_bool.astype(np.uint8) * 255)[:, :, None]
    mask_rgb = np.concatenate((mask_uint8, mask_uint8, mask_uint8), axis=-1)

    canny = cv2.Canny(mask_rgb, threshold1=30, threshold2=100)

    if dilate_iterations > 0:
        kernel = np.ones((3, 3), np.uint8)
        canny = cv2.dilate(canny, kernel, iterations=dilate_iterations)

    img_contour = np.copy(img)
    img_contour[canny > 0] = color

    return {
        "img": img_contour,
        "mask": mask_bool,
        "canny": canny,
    }


def visualize_repre_template(
    version: str, dataset: str, lid: int, template_id: int = 19
):
    """Visualize the PCA-projected features for a given object representation."""

    # Get the representation directory
    base_repre_dir = os.path.join(OUTPUT_DIR, "object_repre")
    repre_dir = repre_util.get_object_repre_dir_path(
        base_repre_dir, version, dataset, lid
    )

    # Load the object representation
    repre = repre_util.load_object_repre(repre_dir)

    # Get the feature vectors
    feat_vectors = repre.feat_vectors_full

    # Project features to 3D visualization space using the stored PCA projector
    if repre.feat_vis_projectors and len(repre.feat_vis_projectors) > 0:
        # Select a template ID
        feat_vectors_3d = feat_vectors[template_id]
        feat_vectors_3d = feat_vectors_3d.permute(1, 0).reshape(-1, 30, 30)

        # Visualize the PCA-projected features
        vis_pca_features = vis_util.vis_pca_feature_map(
            feat_vectors_3d,  # this has to be the chw map (full image), not only the features of the masked object
            image_height=repre.template_cameras_cam_from_model[template_id].height,
            image_width=repre.template_cameras_cam_from_model[template_id].width,
            pca_projector=repre.feat_vis_projectors[0],
        )

        # Store the visualization
        plt.imshow(vis_pca_features)
        plt.savefig("pca_features.png")
        print("PCA-projected features saved to pca_features.png")

    else:
        print("No visualization projectors found in the representation.")


def draw_correspondences(
    query_features: torch.Tensor, template_features: torch.Tensor, mapping: torch.Tensor, corresp_visible=True
):
    """
    Draws correspondences between two maps using the given mapping.

    Parameters:
        query_features (torch.Tensor): First map (H1, W1) or (H1, W1, C)
        template_features (torch.Tensor): Second map (H2, W2) or (H2, W2, C)
        mapping (torch.Tensor): Tensor of shape (N, 2, 2), where each row contains pairs [(y1, x1), (y2, x2)]
    """
    assert len(query_features.shape) in [2, 3], "query_features should be 2D or 3D"
    assert len(template_features.shape) in [2, 3], "template_features should be 2D or 3D"
    assert mapping.shape[1:] == (2, 2), "mapping should be of shape (N, 2, 2)"

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(query_features)
    axes[1].imshow(template_features)

    axes[0].set_title("Query")
    axes[1].set_title("Template")

    for (x1, y1), (x2, y2) in mapping.cpu().numpy():
        axes[0].scatter([x1], [y1], color="red", marker="o", s=1)
        axes[1].scatter([x2], [y2], color="red", marker="o", s=1)

        # Convert to figure coordinates
        xy1 = axes[0].transData.transform((x1, y1))
        xy2 = axes[1].transData.transform((x2, y2))

        # Transform back to figure coordinates
        inv = fig.transFigure.inverted()
        fig_xy1 = inv.transform(xy1)
        fig_xy2 = inv.transform(xy2)

        # Draw line between corresponding points
        if corresp_visible:
            line = plt.Line2D(
                [fig_xy1[0], fig_xy2[0]],
                [fig_xy1[1], fig_xy2[1]],
                transform=fig.transFigure,
                color=(230 / 255, 230 / 255, 230 / 255),
                linestyle="-",
                linewidth=1,
            )
            fig.lines.append(line)

    return fig


def poses_giff(
    poses_cm: np.ndarray,
    camera_K: np.ndarray,
    cand_object: ObjectInstance,
    output_path: str,
    vis_type = "contour",
    background_img: np.ndarray = None,
    residuals: torch.Tensor = None, # shape (i, num_points, dim)
    template_vertices: torch.Tensor = None, # shape (num_points, dim)
):
    
    # Prepare renderer
    renderer_type = renderer_builder.RendererType.PYRENDER_RASTERIZER
    renderer = renderer_builder.build(
        renderer_type=renderer_type, model_path=None
    )
    renderer.add_object_model(cand_object.object_id, cand_object.model_path)

    # Render object in each pose and store images for GIF
    images = []
    for pose in poses_cm:
        img = renderer.render_object(
            obj_id=cand_object.object_id,
            pose_m2c=pose,
            camera_intrinsics=camera_K,
            render_types=[RenderType.COLOR],
        )[RenderType.COLOR]
        images.append(np.uint8(img))

    
    if background_img is not None:
        background_img = background_img * 255
        background_img = np.uint8(background_img)

        if vis_type == "overlay":
            opacity = 0.5
            for i in range(len(images)):
                images[i] = images[i] * (1 - opacity) + background_img * opacity
                images[i] = np.clip(images[i], 0, 255)
                images[i] = np.uint8(images[i])

        if vis_type == "contour":
            for i in range(len(images)):
                images[i] = make_contour_overlay(
                    background_img, images[i].copy(), color=(0, 255, 0), dilate_iterations=2
                )["img"]

        if vis_type == "residuals":
            assert residuals is not None and template_vertices is not None, "Residuals and template vertices must be provided for residual visualization"

            # Normalize residuals to [0, 1] for visualization
            residuals = (residuals - torch.min(residuals)) / (
                torch.max(residuals) - torch.min(residuals)
            )

            for i in range(len(images)):
                image_with_contour = make_contour_overlay(
                    background_img, images[i].copy(), color=(0, 255, 0), dilate_iterations=1
                )["img"]
        
                images[i] = add_residuals_on_img(
                    image_with_contour,
                    residuals[i],
                    template_vertices,
                    poses_cm[i],
                    camera_K,
                )
    else:
        # If no background image, just convert renders to uint8
        images = [np.uint8(img) for img in images]
        
    # Save images as a GIF
    imageio.mimsave(output_path, images, duration=1)


def add_residuals_on_img(
    img: np.ndarray,
    residuals: torch.Tensor,
    vertices: torch.Tensor,
    pose_cm: np.ndarray,
    camera: PinholePlaneCameraModel,
) -> np.ndarray:
    
    # Check residuals are in range [0, 1]
    assert torch.min(residuals) >= 0 and torch.max(residuals) <= 1, "Residuals should be in the range [0, 1]"
   
    # Apply color map on the residuals
    residuals_np = residuals.detach().cpu().numpy()
    residuals_colored = cv2.applyColorMap(
        np.uint8(residuals_np * 255), cv2.COLORMAP_COOL
    )
    
    # Draw residuals as colored points on the image
    for i in range(len(vertices)):
        pt_in_object = vertices[i].detach().cpu().numpy()
        pt_in_camera = pose_cm @ np.append(pt_in_object, 1)
        pt_in_camera = pt_in_camera[:3]

        pt_in_img = camera.eye_to_window(pt_in_camera)
        pt_in_img = pt_in_img.astype(int)

        color_bgr = [int(col) for col in residuals_colored[i][0]]
        color_rgb = [color_bgr[2], color_bgr[1], color_bgr[0]]

        cv2.circle(img, (pt_in_img[0], pt_in_img[1]), 2, tuple(color_rgb), -1)

    return img

def visualize_refinement_results(
    views: List[View],
    object_lid: int,
    consistent_pose_wm: np.ndarray,
    refined_pose_wm: np.ndarray,
    renderers: Any,
    path_testing=None,
):
    if renderers is None:
        print("No renderers provided for visualization.")
        return

    for view in views:
        if view.camera_id not in renderers:
            print(f"No renderer found for camera {view.camera_id}. Skipping visualization.")
            return

    visualizations = {
        "id": [],
        "initial": [],
        "refined": [],
    }

    for view_info in views:
        camera_id, view_id = view_info.camera_id, view_info.image_id
        camera = view_info.camera
        renderer = renderers[camera_id]
        visualizations["id"].append((camera_id, view_id))

        # -----------------------
        # 1. Visualize initial pose
        # -----------------------
        initial_pose_cm = np.linalg.inv(camera.T_world_from_eye) @ np.array(
            consistent_pose_wm
        )

        # render image
        img = renderer.render_object(
            obj_id=object_lid,
            pose_m2c=initial_pose_cm,
            camera_intrinsics=camera,
            render_types=[RenderType.COLOR],
        )[RenderType.COLOR]

        if (
            img.shape[0] == view_info.image.shape[0]
            and img.shape[1] == view_info.image.shape[1]
        ):
            contour = make_contour_overlay(view_info.image, img.copy(), (0, 1.0, 0))
            visualizations["initial"].append(contour["img"])

        else:
            img = img / 255
            visualizations["initial"].append(img)

        # -----------------------
        # 2. Visualize final pose
        # -----------------------
        final_pose_cm = np.linalg.inv(camera.T_world_from_eye) @ refined_pose_wm

        # render image
        img = renderer.render_object(
            obj_id=object_lid,
            pose_m2c=final_pose_cm,
            camera_intrinsics=camera,
            render_types=[RenderType.COLOR],
        )[RenderType.COLOR]

        if (
            img.shape[0] == view_info.image.shape[0]
            and img.shape[1] == view_info.image.shape[1]
        ):
            contour = make_contour_overlay(view_info.image, img.copy(), (0, 1.0, 0))
            visualizations["refined"].append(contour["img"])

        else:
            img = img / 255
            visualizations["refined"].append(img)

    # -----------------------
    # Store visualizations
    # -----------------------
    # Sort visualizations by view ID for consistent ordering
    visualizations["initial"] = [
        visualizations["initial"][visualizations["id"].index(i)]
        for i in sorted(visualizations["id"])
    ]
    visualizations["refined"] = [
        visualizations["refined"][visualizations["id"].index(i)]
        for i in sorted(visualizations["id"])
    ]
    visualizations["id"] = sorted(visualizations["id"])

    len_imgs = len(visualizations["id"])
    fig, axs = plt.subplots(2, len_imgs, figsize=(len_imgs * 5, 8), squeeze=False)
    for i in range(len_imgs):
        axs[0, i].imshow(visualizations["initial"][i])
        axs[0, i].set_title(f"Initial pose {visualizations['id'][i]}")
        axs[0, i].axis("off")
        axs[1, i].imshow(visualizations["refined"][i])
        axs[1, i].set_title(f"Refined pose {visualizations['id'][i]}")
        axs[1, i].axis("off")
    plt.tight_layout()

    os.makedirs(path_testing, exist_ok=True)
    plt.savefig(f"{path_testing}/refinement_results.png")
    plt.close(fig)