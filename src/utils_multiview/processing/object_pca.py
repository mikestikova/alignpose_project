#!/usr/bin/env python3

"""Fits the per-object PCA used by the multi-view refinement, directly from a mesh.

The multi-view refinement renders templates online and needs only the PCA projector
from the object representation. This module renders template views in memory, extracts
features and fits the PCA, without writing template images to disk.
"""

import inspect
import json
import os
from typing import Any, Dict, Iterator, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from foundpose.utils import (
    feature_util,
    geometry,
    misc,
    projector_util,
    renderer_builder,
    repre_util,
    structs,
)
from foundpose.utils.misc import array_to_tensor
from foundpose.utils.renderer_base import RenderType
from foundpose.utils.structs import AlignedBox2f, PinholePlaneCameraModel

from src.utils_multiview.opts import DEFAULT_EXTRACTOR_NAME


def render_template_views(
    model_path: str,
    renderer=None,
    min_num_viewpoints: int = 57,
    num_inplane_rotations: int = 14,
    render_res: int = 1024,
    crop_rel_pad: float = 0.3,
    crop_size: Tuple[int, int] = (420, 420),
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Renders the object from views sampled on a sphere around it.

    Uses the same view sampling, rendering and cropping as `src/scripts/gen_templates.py`.

    Args:
        model_path: Path to the object mesh in millimeters (BOP convention).
        renderer: Optional renderer to reuse across objects.
        min_num_viewpoints: Minimum number of viewpoints sampled on the sphere.
        num_inplane_rotations: Number of in-plane rotations per viewpoint.
        render_res: Resolution of the render camera.
        crop_rel_pad: Relative padding of the crop around the object.
        crop_size: Size of the cropped templates (width, height).
    Yields:
        RGB template (H, W, 3, uint8) and binary mask (H, W, uint8, values 0/255).
    """

    # Prepare a render camera model.
    render_camera_model = misc.get_default_camera_model(render_res=render_res)

    # Build a renderer and add the model to it.
    if renderer is None:
        renderer = renderer_builder.build(
            renderer_type=renderer_builder.RendererType.PYRENDER_RASTERIZER
        )
    obj_id = model_path
    renderer.add_object_model(obj_id=obj_id, model_path=model_path, center_model=True)

    # Generate viewpoints from which the object model will be rendered.
    trimesh_obj_model = renderer.object_meshes[obj_id]
    obj_radius = trimesh_obj_model.bounding_sphere.primitive.radius * 1000  # to mm
    dist = misc.distance_for_rendering(
        obj_radius=obj_radius,
        camera_model=render_camera_model,
    )
    views_sphere = misc.sample_views(
        min_n_views=min_num_viewpoints,
        radius=dist,
        mode="fibonacci",
    )[0]

    # Add in-plane rotations.
    if num_inplane_rotations == 1:
        views = views_sphere
    else:
        inplane_angle = 2 * np.pi / num_inplane_rotations
        views = []
        for view_sphere in views_sphere:
            for inplane_id in range(num_inplane_rotations):
                R_inplane = geometry.rotation_matrix_numpy(
                    inplane_angle * inplane_id, np.array([0, 0, 1])
                )[:3, :3]
                views.append(
                    {
                        "R": R_inplane.dot(view_sphere["R"]),
                        "t": R_inplane.dot(view_sphere["t"]),
                    }
                )

    for view in tqdm(views, desc="Templates"):
        # Camera model for rendering (transformation from camera to model).
        trans_m2c = structs.RigidTransform(R=view["R"], t=view["t"])
        R_c2m = trans_m2c.R.T
        trans_c2m = structs.RigidTransform(R=R_c2m, t=-R_c2m.dot(trans_m2c.t))
        render_camera_model_c2w = PinholePlaneCameraModel(
            width=render_camera_model.width,
            height=render_camera_model.height,
            f=render_camera_model.f,
            c=render_camera_model.c,
            T_world_from_eye=misc.get_rigid_matrix(trans_c2m),
        )

        # Rendering.
        output = renderer.render_object_model(
            obj_id=obj_id,
            camera_model_c2w=render_camera_model_c2w,
            render_types=[RenderType.COLOR, RenderType.MASK],
            return_tensors=False,
            debug=False,
        )
        mask = (255 * output[RenderType.MASK]).astype(np.uint8)

        # Calculate 2D bounding box of the object and make sure it is within the image.
        ys, xs = mask.nonzero()
        box = np.array(misc.calc_2d_box(xs, ys))
        object_box = AlignedBox2f(left=box[0], top=box[1], right=box[2], bottom=box[3])
        if (
            object_box.left == 0
            or object_box.top == 0
            or object_box.right == render_camera_model_c2w.width - 1
            or object_box.bottom == render_camera_model_c2w.height - 1
        ):
            raise ValueError("The model does not fit the viewport.")

        # Crop and resize the color image and the mask.
        crop_box = misc.calc_crop_box(
            box=object_box,
            box_scaling_factor=1.0 + 2 * crop_rel_pad,
            make_square=True,
        )
        rgb_cropped = misc.crop_and_resize_image(
            image=output[RenderType.COLOR],
            crop_box=crop_box,
            output_size=crop_size,
            interpolation=cv2.INTER_LINEAR,
        )
        mask_cropped = misc.crop_and_resize_image(
            image=mask,
            crop_box=crop_box,
            output_size=crop_size,
            interpolation=cv2.INTER_NEAREST,
        )

        yield np.asarray(255.0 * rgb_cropped, np.uint8), mask_cropped


def fit_object_pca(
    model_path: str,
    extractor_name: str = DEFAULT_EXTRACTOR_NAME,
    extractor: Optional[torch.nn.Module] = None,
    renderer=None,
    device: Optional[str] = None,
    grid_cell_size: float = 14.0,
    pca_components: int = 256,
    pca_whiten: bool = False,
    pca_max_samples: int = 100000,
    seed: int = 0,
    **render_kwargs,
) -> Tuple[repre_util.FeatureBasedObjectRepre, Dict[str, Any]]:
    """Fits the PCA of features extracted from rendered templates of one object.

    Args:
        model_path: Path to the object mesh in millimeters (BOP convention).
        extractor_name: Name of the feature extractor. Stored in the representation and
            must match `extractor_name` used for the multi-view refinement.
        extractor: Optional already loaded extractor (created by
            `feature_util.make_feature_extractor(extractor_name)`) to reuse across objects.
        renderer: Optional renderer to reuse across objects.
        device: Device for feature extraction. Defaults to CUDA if available.
        grid_cell_size: Spacing of the grid at which features are sampled.
        pca_components: Number of PCA components.
        pca_whiten: Whether to whiten the PCA.
        pca_max_samples: Maximum number of feature vectors used for fitting the PCA.
        seed: Random seed for subsampling the feature vectors and fitting the PCA.
        **render_kwargs: Options passed to `render_template_views`.
    Returns:
        Object representation containing the fitted PCA projector, and the configuration
        used to fit it (including default rendering options), e.g. for `save_object_pca`.
    """

    # Record all options used for fitting, including defaults of `render_template_views`.
    render_defaults = {
        name: param.default
        for name, param in inspect.signature(render_template_views).parameters.items()
        if param.default is not inspect.Parameter.empty and name != "renderer"
    }
    config = {
        "model_path": model_path,
        "extractor_name": extractor_name,
        "grid_cell_size": grid_cell_size,
        "pca_components": pca_components,
        "pca_whiten": pca_whiten,
        "pca_max_samples": pca_max_samples,
        "seed": seed,
        **render_defaults,
        **render_kwargs,
    }

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if extractor is None:
        extractor = feature_util.make_feature_extractor(extractor_name)
    extractor.to(device)

    # Extract features from the masked object in each template.
    feat_vectors_list = []
    for rgb, mask in render_template_views(model_path, renderer=renderer, **render_kwargs):
        image_chw = (
            array_to_tensor(rgb).to(torch.float32).permute(2, 0, 1).to(device) / 255.0
        )
        object_mask = array_to_tensor(mask).to(torch.float32).to(device)
        feat_vectors_list.append(
            feature_util.get_masked_features(
                image_chw=image_chw,
                object_mask=object_mask,
                extractor=extractor,
                grid_cell_size=grid_cell_size,
            )
        )
    feat_vectors = torch.cat(feat_vectors_list, dim=0)

    # Fit the PCA. Torch seeds the subsampling of feature vectors, numpy the randomized
    # SVD solver which sklearn picks for this data size.
    torch.manual_seed(seed)
    np.random.seed(seed)
    pca_projector = projector_util.PCAProjector(
        n_components=pca_components, whiten=pca_whiten
    )
    pca_projector.fit(feat_vectors, max_samples=pca_max_samples)

    repre = repre_util.FeatureBasedObjectRepre(
        feat_opts=repre_util.FeatureOpts(extractor_name=extractor_name),
    )
    repre.feat_raw_projectors.append(pca_projector)
    repre.feat_vis_projectors = [pca_projector]
    return repre, config


def save_object_pca(
    repre: repre_util.FeatureBasedObjectRepre,
    output_dir: str,
    object_id,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Saves the representation to `<output_dir>/<object_id>/repre.pth`.

    If `config` (returned by `fit_object_pca`) is given, it is saved next to it as
    `config.json`. To use the representation in `src/scripts/refine_multiview.py`, set
    `output_dir` to `<OUTPUT_DIR>/object_repre/<object_dataset>/<repre_version>`.

    Returns:
        Directory of the saved representation.
    """
    repre_dir = os.path.join(output_dir, str(object_id))
    os.makedirs(repre_dir, exist_ok=True)
    repre_util.save_simple_object_repre(repre, repre_dir)
    if config is not None:
        with open(os.path.join(repre_dir, "config.json"), "w") as f:
            json.dump({"object_id": object_id, **config}, f, indent=2)
    return repre_dir
