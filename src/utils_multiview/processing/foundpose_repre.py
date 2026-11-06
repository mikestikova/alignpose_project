#!/usr/bin/env python3

"""Builds the full FoundPose object representation directly from a mesh.

The single-view pose estimation (FoundPose) matches query features against templates of
the object, so it needs the full representation: features registered in 3D, their PCA,
visual words and tf-idf template descriptors. This module does what
`foundpose/scripts/gen_templates.py` + `gen_repre.py` do, but without a BOP dataset:
templates are rendered in memory from the mesh with a given camera and camera-object
distance, and are written to disk only if requested with `save_templates`.

The representation also contains the PCA used by the multi-view refinement.
"""

import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from foundpose.utils import (
    cluster_util,
    feature_util,
    geometry,
    misc,
    projector_util,
    renderer_builder,
    repre_util,
    structs,
    template_util,
)
from foundpose.utils.misc import array_to_tensor, warp_depth_image, warp_image
from foundpose.utils.renderer_base import RenderType
from foundpose.utils.structs import AlignedBox2f, PinholePlaneCameraModel

from src.utils_multiview.opts import DEFAULT_EXTRACTOR_NAME


def load_intrinsics(path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Reads camera intrinsics from a `.json`, `.yml` or `.yaml` file.

    Accepts `fx`, `fy`, `cx`, `cy` or a row-major 3x3 `cam_K`, and the image size as
    `width` / `height` (or `w` / `h`). Other keys are ignored.

    Returns:
        The 3x3 intrinsic matrix and the image size as (width, height).
    """
    path = Path(path).expanduser()
    if path.suffix in (".yml", ".yaml"):
        import yaml

        c = yaml.safe_load(path.read_text())
    else:
        c = json.loads(path.read_text())

    def get(*names):
        for name in names:
            if name in c:
                return c[name]
        raise ValueError(f"None of {names} found in camera file {path}")

    if "cam_K" in c:
        K = np.asarray(c["cam_K"], float).reshape(3, 3)
    else:
        K = np.array(
            [
                [float(get("fx")), 0.0, float(get("cx"))],
                [0.0, float(get("fy")), float(get("cy"))],
                [0.0, 0.0, 1.0],
            ]
        )
    return K, (int(get("width", "w")), int(get("height", "h")))


def render_foundpose_templates(
    model_path: str,
    K: np.ndarray,
    image_size: Tuple[int, int],
    depth_range: Tuple[float, float],
    renderer=None,
    num_viewspheres: int = 1,
    min_num_viewpoints: int = 57,
    num_inplane_rotations: int = 14,
    ssaa_factor: float = 4.0,
    crop_rel_pad: float = 0.2,
    crop_size: Tuple[int, int] = (420, 420),
    features_patch_size: int = 14,
) -> List[Dict[str, Any]]:
    """Renders the templates of an object, as `foundpose/scripts/gen_templates.py` does.

    Views are sampled on `num_viewspheres` spheres around the object, with radii spread
    over `depth_range`, and rendered with the camera given by `K` and `image_size`. Views
    in which the object is not fully visible are skipped.

    Args:
        model_path: Path to the object mesh in millimeters. The object is rendered in its
            own model frame, so the estimated poses refer to this frame.
        K: 3x3 intrinsics of the camera the templates should resemble (usually the camera
            the object will be seen with).
        image_size: (width, height) of that camera.
        depth_range: (min, max) camera-object distance in mm. Use the same value twice for
            a single distance.
        renderer: Optional renderer to reuse across objects. A renderer renders at one
            size only, so reuse it only with the same `image_size` and `ssaa_factor`.
        num_viewspheres, min_num_viewpoints, num_inplane_rotations: View sampling.
        ssaa_factor: Supersampling factor for anti-aliasing (1 to disable; renders are
            `ssaa_factor` times larger than the camera image).
        crop_rel_pad, crop_size: Cropping of the templates around the object.
        features_patch_size: Patch size of the feature extractor.
    Returns:
        One dict per template with `rgb` (HxWx3 uint8), `depth` (HxW float32, mm),
        `mask` (HxW uint8, 0/255) and `camera` (the template camera, with
        `T_world_from_eye` the camera-to-model transformation).
    """

    # Prepare a camera for the template (square viewport of a size divisible by the patch size).
    width, height = image_size
    image_side = features_patch_size * int(max(width, height) / features_patch_size)
    camera_model = PinholePlaneCameraModel(
        width=image_side,
        height=image_side,
        f=(K[0, 0], K[1, 1]),
        c=(K[0, 2] - 0.5 * (width - image_side), K[1, 2] - 0.5 * (height - image_side)),
    )
    # Prepare a camera for rendering, upsampled for SSAA (supersampling anti-aliasing).
    render_camera_model = PinholePlaneCameraModel(
        width=int(camera_model.width * ssaa_factor),
        height=int(camera_model.height * ssaa_factor),
        f=(camera_model.f[0] * ssaa_factor, camera_model.f[1] * ssaa_factor),
        c=(camera_model.c[0] * ssaa_factor, camera_model.c[1] * ssaa_factor),
    )

    # Build a renderer and add the model in its own frame.
    if renderer is None:
        renderer = renderer_builder.build(
            renderer_type=renderer_builder.RendererType.PYRENDER_RASTERIZER
        )
    obj_id = str(model_path)
    renderer.add_object_model(obj_id=obj_id, model_path=model_path)

    # Radii of the view spheres, sampled uniformly in the range of camera-object distances.
    min_depth, max_depth = np.min(depth_range), np.max(depth_range)
    depth_cell_size = (max_depth - min_depth) / float(num_viewspheres)
    viewsphere_radii = [
        min_depth + (depth_cell_id + 0.5) * depth_cell_size
        for depth_cell_id in range(num_viewspheres)
    ]

    # Generate viewpoints from which the object model will be rendered.
    views_sphere = []
    for radius in viewsphere_radii:
        views_sphere += misc.sample_views(
            min_n_views=min_num_viewpoints, radius=radius, mode="fibonacci"
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

    templates = []
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
            render_types=[RenderType.COLOR, RenderType.DEPTH, RenderType.MASK],
            return_tensors=False,
            debug=False,
        )
        output[RenderType.MASK] = (255 * output[RenderType.MASK]).astype(np.uint8)

        # Skip views in which the object is not fully visible.
        ys, xs = output[RenderType.MASK].nonzero()
        box = np.array(misc.calc_2d_box(xs, ys))
        object_box = AlignedBox2f(left=box[0], top=box[1], right=box[2], bottom=box[3])
        if (
            object_box.left == 0
            or object_box.top == 0
            or object_box.right == render_camera_model_c2w.width
            or object_box.bottom == render_camera_model_c2w.height
        ):
            continue

        # Construct a virtual camera focused on the object.
        crop_box = misc.calc_crop_box(box=object_box, make_square=True)
        crop_camera_model_c2w = misc.construct_crop_camera(
            box=crop_box,
            camera_model_c2w=render_camera_model_c2w,
            viewport_size=(int(crop_size[0] * ssaa_factor), int(crop_size[1] * ssaa_factor)),
            viewport_rel_pad=crop_rel_pad,
        )

        # Map the images to the virtual camera.
        for key in output.keys():
            if key == RenderType.DEPTH:
                output[key] = warp_depth_image(
                    src_camera=render_camera_model_c2w,
                    dst_camera=crop_camera_model_c2w,
                    src_depth_image=output[key],
                )
            elif key == RenderType.COLOR:
                interpolation = (
                    cv2.INTER_AREA
                    if crop_box.width >= crop_camera_model_c2w.width
                    else cv2.INTER_LINEAR
                )
                output[key] = warp_image(
                    src_camera=render_camera_model_c2w,
                    dst_camera=crop_camera_model_c2w,
                    src_image=output[key],
                    interpolation=interpolation,
                )
            else:
                output[key] = warp_image(
                    src_camera=render_camera_model_c2w,
                    dst_camera=crop_camera_model_c2w,
                    src_image=output[key],
                    interpolation=cv2.INTER_NEAREST,
                )

        # The virtual camera, scaled to the template size, is the template camera.
        camera_model_c2w = crop_camera_model_c2w.copy()
        scale_factor = crop_size[0] / float(crop_camera_model_c2w.width)
        camera_model_c2w.width = crop_size[0]
        camera_model_c2w.height = crop_size[1]
        camera_model_c2w.c = (camera_model_c2w.c[0] * scale_factor, camera_model_c2w.c[1] * scale_factor)
        camera_model_c2w.f = (camera_model_c2w.f[0] * scale_factor, camera_model_c2w.f[1] * scale_factor)

        # Downsample the renderings to the target size in case of SSAA.
        if ssaa_factor != 1.0:
            target_size = (camera_model_c2w.width, camera_model_c2w.height)
            for key in output.keys():
                interpolation = cv2.INTER_AREA if key == RenderType.COLOR else cv2.INTER_NEAREST
                output[key] = misc.resize_image(
                    image=output[key], size=target_size, interpolation=interpolation
                )

        templates.append(
            {
                "rgb": np.asarray(255.0 * output[RenderType.COLOR], np.uint8),
                # Rounded to mm, as when gen_templates.py saves the depth as a 16-bit PNG.
                "depth": np.round(output[RenderType.DEPTH]).astype(np.uint16).astype(np.float32),
                "mask": output[RenderType.MASK],
                "camera": camera_model_c2w,
            }
        )
    return templates


def save_templates(templates: List[Dict[str, Any]], templates_dir: str) -> None:
    """Saves templates, e.g. for debugging.

    Writes `rgb/`, `depth/` (16-bit PNG, mm) and `mask/` images named
    `template_<id>.png`, and `cameras.json` with the camera of each template
    (`T_WorldFromCamera` is the camera-to-model transformation). Template ids match the
    template ids in the representation.

    Args:
        templates: Output of `render_foundpose_templates`.
        templates_dir: Output directory.
    """
    for name in ("rgb", "depth", "mask"):
        os.makedirs(os.path.join(templates_dir, name), exist_ok=True)

    cameras = []
    for template_id, template in enumerate(templates):
        file_name = f"template_{template_id:04d}.png"
        cv2.imwrite(os.path.join(templates_dir, "rgb", file_name), template["rgb"][:, :, ::-1])
        cv2.imwrite(os.path.join(templates_dir, "depth", file_name), template["depth"].astype(np.uint16))
        cv2.imwrite(os.path.join(templates_dir, "mask", file_name), template["mask"])
        cameras.append({"template_id": template_id, **template["camera"].to_json()})

    with open(os.path.join(templates_dir, "cameras.json"), "w") as f:
        json.dump(cameras, f, indent=2)


def build_foundpose_repre(
    templates: List[Dict[str, Any]],
    extractor_name: str = DEFAULT_EXTRACTOR_NAME,
    extractor: Optional[torch.nn.Module] = None,
    device: Optional[str] = None,
    grid_cell_size: float = 14.0,
    pca_components: int = 256,
    pca_whiten: bool = False,
    pca_max_samples: int = 100000,
    cluster_num: int = 2048,
    template_desc_opts: repre_util.TemplateDescOpts = repre_util.TemplateDescOpts(),
    seed: int = 0,
) -> repre_util.FeatureBasedObjectRepre:
    """Builds the representation from templates, as `foundpose/scripts/gen_repre.py` does.

    Args:
        templates: Output of `render_foundpose_templates`.
        extractor_name: Feature extractor. Must match the one used for pose estimation.
        extractor: Optional already loaded extractor, to reuse across objects.
        device: Defaults to CUDA if available.
        grid_cell_size: Spacing of the grid at which features are sampled.
        pca_components, pca_whiten, pca_max_samples: PCA of the features.
        cluster_num: Number of visual words (k-means clusters of the features).
        template_desc_opts: Options of the template descriptors (tf-idf).
        seed: Random seed for subsampling the features for the PCA and fitting it.
    Returns:
        The full representation: features registered in 3D (PCA-projected), the PCA,
        visual words, tf-idf template descriptors, and the templates and their cameras.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if extractor is None:
        extractor = feature_util.make_feature_extractor(extractor_name)
    extractor.to(device)

    # Extract features from each template and register them in 3D.
    feat_vectors_full_list = []
    feat_vectors_list = []
    query_pts_list = []
    feat_to_vertex_ids_list = []
    vertices_in_model_list = []
    feat_to_template_ids_list = []
    templates_list = []
    template_cameras_cam_from_model_list = []
    for template_id, template in enumerate(templates):
        camera = template["camera"]
        image_chw = (
            array_to_tensor(template["rgb"]).to(torch.float32).permute(2, 0, 1).to(device) / 255.0
        )
        depth_image_hw = array_to_tensor(template["depth"]).to(torch.float32).to(device)
        object_mask_modal = array_to_tensor(template["mask"]).to(torch.float32).to(device)

        # Model and world frames are aligned, so camera-to-world is camera-to-model.
        T_model_from_camera = (
            array_to_tensor(camera.T_world_from_eye).to(torch.float32).to(device)
        )

        (
            feat_vectors_full,
            feat_vectors,
            query_pts,
            feat_to_vertex_ids,
            vertices_in_model,
        ) = feature_util.get_visual_features_registered_in_3d(
            image_chw=image_chw,
            depth_image_hw=depth_image_hw,
            object_mask=object_mask_modal,
            camera=camera,
            T_model_from_camera=T_model_from_camera,
            extractor=extractor,
            grid_cell_size=grid_cell_size,
            debug=False,
        )

        # Full feature maps are not stored, to save space.
        feat_vectors_full_list.append(
            torch.empty((0, feat_vectors_full.shape[1]), device=device)
        )
        feat_vectors_list.append(feat_vectors)
        query_pts_list.append(query_pts)
        feat_to_vertex_ids_list.append(feat_to_vertex_ids)
        vertices_in_model_list.append(vertices_in_model)
        feat_to_template_ids_list.append(
            template_id * torch.ones(feat_vectors.shape[0], dtype=torch.int32, device=device)
        )
        templates_list.append((image_chw * 255).to(torch.uint8))
        camera_model = camera.copy()
        camera_model.extrinsics = torch.linalg.inv(T_model_from_camera)
        template_cameras_cam_from_model_list.append(camera_model)

    repre = repre_util.FeatureBasedObjectRepre(
        vertices=torch.cat(vertices_in_model_list),
        feat_vectors=torch.cat(feat_vectors_list),
        feat_vectors_full=torch.stack(feat_vectors_full_list),
        query_pts=torch.cat(query_pts_list),
        feat_opts=repre_util.FeatureOpts(extractor_name=extractor_name),
        feat_to_vertex_ids=torch.cat(feat_to_vertex_ids_list),
        feat_to_template_ids=torch.cat(feat_to_template_ids_list),
        templates=torch.stack(templates_list),
        template_cameras_cam_from_model=template_cameras_cam_from_model_list,
    )

    # Transform the features to a PCA space. Torch seeds the subsampling of the
    # features, numpy the randomized SVD solver which sklearn picks for this data size.
    torch.manual_seed(seed)
    np.random.seed(seed)
    pca_projector = projector_util.PCAProjector(n_components=pca_components, whiten=pca_whiten)
    pca_projector.fit(repre.feat_vectors, max_samples=pca_max_samples)
    repre.feat_raw_projectors.append(pca_projector)
    feat_vectors = pca_projector.transform(repre.feat_vectors)

    # Cluster the features into visual words.
    centroids, cluster_ids, _ = cluster_util.kmeans(
        samples=feat_vectors, num_centroids=cluster_num, verbose=True
    )
    repre.feat_cluster_centroids = centroids
    repre.feat_to_cluster_ids = cluster_ids

    # Template descriptors.
    repre.template_desc_opts = template_desc_opts
    if template_desc_opts.desc_type != "tfidf":
        raise ValueError(f"Unknown template descriptor type: {template_desc_opts.desc_type}")
    repre.template_descs, repre.feat_cluster_idfs = template_util.calc_tfidf_descriptors(
        feat_vectors=feat_vectors,
        feat_words=repre.feat_cluster_centroids,
        feat_to_word_ids=repre.feat_to_cluster_ids,
        feat_to_template_ids=repre.feat_to_template_ids,
        num_templates=len(repre.templates),
        tfidf_knn_k=template_desc_opts.tfidf_knn_k,
        tfidf_soft_assign=template_desc_opts.tfidf_soft_assign,
        tfidf_soft_sigma_squared=template_desc_opts.tfidf_soft_sigma_squared,
    )

    # The PCA projector is also used for visualizations.
    repre.feat_vis_projectors = [pca_projector]
    repre.feat_vectors = feat_vectors
    return repre


def foundpose_repre_config(
    model_path: str,
    K: np.ndarray,
    image_size: Tuple[int, int],
    depth_range: Tuple[float, float],
    extractor_name: str = DEFAULT_EXTRACTOR_NAME,
    render_opts: Optional[Dict[str, Any]] = None,
    repre_opts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """All options used to build a representation, including the defaults, e.g. for
    `save_foundpose_repre`.

    Args:
        model_path, K, image_size, depth_range: See `render_foundpose_templates`.
        extractor_name: See `build_foundpose_repre`.
        render_opts: Other options passed to `render_foundpose_templates`.
        repre_opts: Other options passed to `build_foundpose_repre`.
    """

    def defaults(fn, skip):
        return {
            name: param.default
            for name, param in inspect.signature(fn).parameters.items()
            if param.default is not inspect.Parameter.empty and name not in skip
        }

    config = {
        "model_path": str(model_path),
        "K": np.asarray(K).tolist(),
        "image_size": list(image_size),
        "depth_range": [float(d) for d in depth_range],
        "extractor_name": extractor_name,
        **defaults(render_foundpose_templates, {"renderer"}),
        **(render_opts or {}),
        **defaults(build_foundpose_repre, {"extractor_name", "extractor", "device"}),
        **(repre_opts or {}),
    }
    config["template_desc_opts"] = dict(config["template_desc_opts"]._asdict())
    return config


def save_foundpose_repre(
    repre: repre_util.FeatureBasedObjectRepre,
    output_dir: str,
    object_id,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Saves the representation to `<output_dir>/<object_id>/repre.pth`.

    Args:
        repre: Output of `build_foundpose_repre`.
        output_dir: Directory with one subdirectory per object.
        object_id: ID of the object, the name of its subdirectory.
        config: Optional output of `foundpose_repre_config`, saved as `config.json`.
    Returns:
        Directory of the saved representation.
    """
    repre_dir = os.path.join(output_dir, str(object_id))
    os.makedirs(repre_dir, exist_ok=True)
    repre_util.save_object_repre(repre, repre_dir)
    if config is not None:
        with open(os.path.join(repre_dir, "config.json"), "w") as f:
            json.dump({"object_id": object_id, **config}, f, indent=2)
    return repre_dir
