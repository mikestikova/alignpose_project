#!/usr/bin/env python3

"""Single-image pose estimation core, shared by the inference scripts.

The body of `estimate_pose_in_image` was moved verbatim out of the inner loop of
`scripts/infer_without_targets.py` so that both the BOP inference script and
`scripts/infer_single.py` run exactly the same algorithm.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.utils_multiview.datasets.object_dataset import ObjectDataset
import cv2
import numpy as np
import torch

from foundpose.utils import (
    corresp_util,
    feature_util,
    featuremetric_refiner,
    knn_util,
    logging,
    misc as misc_util,
    pnp_util,
    projector_util,
    repre_util,
    structs,
)
from foundpose.utils.misc import array_to_tensor, warp_image
from foundpose.utils.structs import AlignedBox2f, PinholePlaneCameraModel

_LOGGER: logging.Logger = logging.get_logger()


@dataclass
class ObjectContext:
    """Per-object data needed for pose estimation.

    Building the per-template KNN indices dominates the setup cost, so this is
    built once per object and reused for every image and detection.
    """

    repre: repre_util.FeatureBasedObjectRepre
    visual_words_knn_index: Optional[knn_util.KNN] = None
    template_knn_indices: List[knn_util.KNN] = field(default_factory=list)


@dataclass
class PoseEstimate:
    """Result of `estimate_pose_in_image`.

    `R_m2c` / `t_m2c` in `final_pose` are expressed in `camera_c2w`, which is the
    virtual crop camera when `opts.crop` is set (not the camera passed in). Compose
    through `camera_c2w.T_world_from_eye` to get the pose in the original frame.
    """

    final_pose: Dict[str, Any]
    coarse_pose: Dict[str, Any]
    camera_c2w: PinholePlaneCameraModel
    corresp: List[Dict[str, Any]]
    best_coarse_pose_id: int
    image_np_hwc: np.ndarray
    mask_modal: np.ndarray
    box_amodal: AlignedBox2f
    feature_map_chw: torch.Tensor
    feature_map_chw_proj: torch.Tensor
    times: Dict[str, Optional[float]]


def build_object_context(
    repre: repre_util.FeatureBasedObjectRepre,
    opts: Any,
    logger: Optional[logging.Logger] = None,
) -> ObjectContext:
    """Builds the KNN indices for one object.

    Moved from `scripts/infer_without_targets.py` (the visual-words and
    per-template index blocks that followed loading the representation).
    """

    logger = logger or _LOGGER

    # Build a kNN index from object feature vectors.
    visual_words_knn_index = None
    if opts.match_template_type == "tfidf":
        visual_words_knn_index = knn_util.KNN(
            k=repre.template_desc_opts.tfidf_knn_k,
            metric=repre.template_desc_opts.tfidf_knn_metric,
        )
        visual_words_knn_index.fit(repre.feat_cluster_centroids)

    # Build per-template KNN index with features from that template.
    template_knn_indices = []
    if opts.match_feat_matching_type == "cyclic_buddies":
        logger.info("Building per-template KNN indices...")
        for template_id in range(len(repre.template_cameras_cam_from_model)):
            logger.info(f"Building KNN index for template {template_id}...")
            tpl_feat_mask = repre.feat_to_template_ids == template_id
            tpl_feat_ids = torch.nonzero(tpl_feat_mask).flatten()

            template_feats = repre.feat_vectors[tpl_feat_ids]

            # Build knn index for object features.
            template_knn_index = knn_util.KNN(k=1, metric="l2")
            template_knn_index.fit(template_feats.cpu())
            template_knn_indices.append(template_knn_index)
        logger.info("Per-template KNN indices built.")

    return ObjectContext(
        repre=repre,
        visual_words_knn_index=visual_words_knn_index,
        template_knn_indices=template_knn_indices,
    )


def make_grid_points(
    opts: Any, image_size: Tuple[int, int], device: str
) -> torch.Tensor:
    """Generates the grid of points at which query features are sampled.

    `image_size` is (width, height) of the original image; it is only used when
    cropping is disabled.
    """

    if opts.crop:
        grid_size = opts.crop_size
    else:
        grid_size = image_size
    grid_points = feature_util.generate_grid_points(
        grid_size=grid_size,
        cell_size=opts.grid_cell_size,
    )
    return grid_points.to(device)


def estimate_pose_in_image(
    orig_image_np_hwc: np.ndarray,
    orig_camera_c2w: PinholePlaneCameraModel,
    orig_mask_modal: np.ndarray,
    orig_box_amodal: AlignedBox2f,
    ctx: ObjectContext,
    extractor: torch.nn.Module,
    grid_points: torch.Tensor,
    opts: Any,
    device: str = "cuda",
    timer: Optional[misc_util.Timer] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[PoseEstimate]:
    """Estimates the pose of one object instance in one image.

    Args:
        orig_image_np_hwc: RGB image as float32 in [0, 1].
        orig_camera_c2w: Camera of `orig_image_np_hwc`.
        orig_mask_modal: Modal mask of the instance, same size as the image.
        orig_box_amodal: Amodal 2D box of the instance.
        ctx: Per-object data from `build_object_context`.
        extractor: Feature extractor, already on `device`.
        grid_points: Output of `make_grid_points`.
        opts: Inference options (see `InferOpts` in the inference scripts).
    Returns:
        The estimate, or None if no pose could be established.
    """

    logger = logger or _LOGGER
    timer = timer or misc_util.Timer(enabled=False)

    # Unpack the object context (these names are used by the moved code below).
    repre = ctx.repre
    visual_words_knn_index = ctx.visual_words_knn_index
    template_knn_indices = ctx.template_knn_indices

    times: Dict[str, Optional[float]] = {}

    # Default on: a pose behind the camera is never physically valid.
    require_positive_depth = getattr(opts, "require_positive_depth", True)

    timer.start()

    # Optional cropping.
    if not opts.crop:
        camera_c2w = orig_camera_c2w
        image_np_hwc = orig_image_np_hwc
        mask_modal = orig_mask_modal
        box_amodal = orig_box_amodal
    else:
        # Get box for cropping.
        crop_box = misc_util.calc_crop_box(
            box=orig_box_amodal,
            make_square=True,
        )

        # Construct a virtual camera focused on the crop.
        crop_camera_model_c2w = misc_util.construct_crop_camera(
            box=crop_box,
            camera_model_c2w=orig_camera_c2w,
            viewport_size=opts.crop_size,
            viewport_rel_pad=opts.crop_rel_pad,
        )

        # Map images to the virtual camera.
        interpolation = (
            cv2.INTER_AREA
            if crop_box.width >= crop_camera_model_c2w.width
            else cv2.INTER_LINEAR
        )
        image_np_hwc = warp_image(
            src_camera=orig_camera_c2w,
            dst_camera=crop_camera_model_c2w,
            src_image=orig_image_np_hwc,
            interpolation=interpolation,
        )
        mask_modal = warp_image(
            src_camera=orig_camera_c2w,
            dst_camera=crop_camera_model_c2w,
            src_image=orig_mask_modal,
            interpolation=cv2.INTER_NEAREST,
        )

        # Recalculate the object bounding box (it changed if we constructed the virtual camera).
        ys, xs = mask_modal.nonzero()
        box = np.array(misc_util.calc_2d_box(xs, ys))
        box_amodal = AlignedBox2f(
            left=box[0],
            top=box[1],
            right=box[2],
            bottom=box[3],
        )

        # The virtual camera is becoming the main camera.
        camera_c2w = crop_camera_model_c2w

    times["prep"] = timer.elapsed("Time for preparation")
    timer.start()

    # Extract feature map from the crop.
    image_tensor_chw = (
        array_to_tensor(image_np_hwc)
        .to(torch.float32)
        .permute(2, 0, 1)
        .to(device)
    )
    image_tensor_bchw = image_tensor_chw.unsqueeze(0)
    extractor_output = extractor(image_tensor_bchw)
    feature_map_chw = extractor_output["feature_maps"][0]

    times["feat_extract"] = timer.elapsed("Time for feature extraction")
    timer.start()

    # Keep only points inside the object mask.
    mask_modal_tensor = array_to_tensor(mask_modal).to(device)
    query_points = feature_util.filter_points_by_mask(
        grid_points, mask_modal_tensor
    )

    # Subsample query points if we have too many.
    if query_points.shape[0] > opts.max_num_queries:
        perm = torch.randperm(query_points.shape[0])
        query_points = query_points[perm[: opts.max_num_queries]]
        msg = (
            "Randomly sumbsampled queries "
            f"({perm.shape[0]} -> {query_points.shape[0]}))"
        )
        logging.log_heading(logger, msg, style=logging.RED_BOLD)

    # Extract features at the selected points, of shape (num_points, feat_dims).
    timer.start()
    query_features = feature_util.sample_feature_map_at_points(
        feature_map_chw=feature_map_chw,
        points=query_points,
        image_size=(image_np_hwc.shape[1], image_np_hwc.shape[0]),
    ).contiguous()

    times["grid_sample"] = timer.elapsed("Time for grid sample")
    timer.start()
    # Potentially project features to a PCA space.
    if (
        query_features.shape[1] != repre.feat_vectors.shape[1]
        and len(repre.feat_raw_projectors) != 0
    ):
        try:
            query_features_proj = projector_util.project_features(
                feat_vectors=query_features,
                projectors=repre.feat_raw_projectors,
            ).contiguous()
        except Exception as e:
            logger.warning(
                f"Projecting features failed with error {e}, using raw features."
            )
            return None
        _c, _h, _w = feature_map_chw.shape
        feature_map_chw_proj = (
            projector_util.project_features(
                feat_vectors=feature_map_chw.permute(1, 2, 0).view(-1, _c),
                projectors=repre.feat_raw_projectors,
            )
            .view(_h, _w, -1)
            .permute(2, 0, 1)
        )
    else:
        query_features_proj = query_features
        feature_map_chw_proj = feature_map_chw

    times["proj"] = timer.elapsed("Time for projection")
    timer.start()

    # Establish 2D-3D correspondences.
    corresp = []
    if len(query_points) != 0:
        corresp = corresp_util.establish_correspondences(
            query_points=query_points,
            query_features=query_features_proj,
            object_repre=repre,
            template_matching_type=opts.match_template_type,
            template_knn_indices=template_knn_indices,
            feat_matching_type=opts.match_feat_matching_type,
            top_n_templates=opts.match_top_n_templates,
            top_k_buddies=opts.match_top_k_buddies,
            visual_words_knn_index=visual_words_knn_index,
            debug=opts.debug,
        )

    times["corresp"] = timer.elapsed("Time for corresp")
    timer.start()

    logger.info(
        f"Number of corresp: {[len(c['coord_2d']) for c in corresp]}"
    )

    # Estimate coarse poses from corespondences.
    coarse_poses = []
    for corresp_id, corresp_curr in enumerate(corresp):
        # We need at least 3 correspondences for P3P.
        num_corresp = len(corresp_curr["coord_2d"])
        if num_corresp < 6:
            logger.info(f"Only {num_corresp} correspondences, skipping.")
            continue
        (
            coarse_pose_success,
            R_m2c_coarse,
            t_m2c_coarse,
            inliers_coarse,
            quality_coarse,
        ) = pnp_util.estimate_pose(
            corresp=corresp_curr,
            camera_c2w=camera_c2w,
            pnp_type=opts.pnp_type,
            pnp_ransac_iter=opts.pnp_ransac_iter,
            pnp_inlier_thresh=opts.pnp_inlier_thresh,
            pnp_required_ransac_conf=opts.pnp_required_ransac_conf,
            pnp_refine_lm=opts.pnp_refine_lm,
        )

        logger.info(
            f"Quality of coarse pose {corresp_id}: {quality_coarse}"
        )

        # A pose with z <= 0 puts the object behind the camera, which no camera
        # can see. PnP admits it because reprojection error cannot separate the
        # two solutions of a near-planar point set, so reject it here -- left in,
        # it can win the argmax below on inlier count alone.
        if coarse_pose_success and require_positive_depth:
            z = float(np.asarray(t_m2c_coarse).ravel()[2])
            if z <= 0.0:
                logger.info(
                    f"Coarse pose {corresp_id} is behind the camera "
                    f"(z = {z:.1f} mm), rejected."
                )
                continue

        if coarse_pose_success:
            coarse_poses.append(
                {
                    "type": "coarse",
                    "R_m2c": R_m2c_coarse,
                    "t_m2c": t_m2c_coarse,
                    "corresp_id": corresp_id,
                    "quality": quality_coarse,
                    "inliers": inliers_coarse,
                }
            )

    # Find the best coarse pose.
    best_coarse_quality = None
    best_coarse_pose_id = 0
    for coarse_pose_id, pose in enumerate(coarse_poses):
        if (
            best_coarse_quality is None
            or pose["quality"] > best_coarse_quality
        ):
            best_coarse_pose_id = coarse_pose_id
            best_coarse_quality = pose["quality"]

    times["pose_coarse"] = timer.elapsed("Time for coarse pose")

    timer.start()

    # Select the final pose estimate.
    final_poses = []

    if opts.final_pose_type in ["best_coarse", "refined"]:
        # If no successful coarse pose, continue.
        if len(coarse_poses) == 0:
            return None

        # Select the refined pose corresponding to the best coarse pose as the final pose.
        final_pose = None

        if opts.final_pose_type in ["best_coarse", "refined"]:
            final_pose = coarse_poses[best_coarse_pose_id]

        if final_pose is not None:
            final_poses.append(final_pose)

    else:
        raise ValueError(f"Unknown final pose type {opts.final_pose_type}")

    times["final_select"] = timer.elapsed("Time for selecting final pose")

    # Keep best coarse pose
    coarse = deepcopy(final_poses[0])

    if opts.final_pose_type == "refined":
        timer.start()

        top_coarse_pred = final_poses[0]

        # Get coarse pose as initial pose.
        initial_pose = structs.ObjectPose(
            R=top_coarse_pred["R_m2c"], t=top_coarse_pred["t_m2c"]
        )

        # Get the best template id from correspondences
        top_corresp = corresp[top_coarse_pred["corresp_id"]]
        best_template_id = top_corresp["template_id"].item()

        # Get template data from repre
        templ_mask = torch.where(
            repre.feat_to_template_ids == best_template_id
        )[0]
        template_masked_features_ref = repre.feat_vectors[
            templ_mask
        ].unsqueeze(0)
        template_vertices_ref = repre.vertices[templ_mask].unsqueeze(0)

        # Get the feature map for the query
        feature_map_chw_proj_ref = feature_map_chw_proj.unsqueeze(0)
        feature_map_chw_proj_ref = torch.nn.functional.interpolate(
            feature_map_chw_proj_ref,
            (opts.crop_size[0], opts.crop_size[1]),
            mode="bilinear",
        )

        # Run the refinement
        optimized_pose, failed = featuremetric_refiner.refine(
            initial_pose_m2c=initial_pose,
            template_vertices_ref=template_vertices_ref,
            template_masked_features_ref=template_masked_features_ref,
            feature_map_chw_proj_ref=feature_map_chw_proj_ref,
            camera_c2w=camera_c2w,
            num_iters=30,
        )
        if failed:
            logger.info(
                f"Featuremetric refinement failed, keeping coarse pose"
            )
        # Update final pose with the refined pose
        final_poses[0]["R_m2c"] = optimized_pose.R
        final_poses[0]["t_m2c"] = optimized_pose.t.reshape(3, 1)

        times["featuremetric_refine"] = timer.elapsed(
            "Time for featuremetric refine"
        )

    return PoseEstimate(
        final_pose=final_poses[0],
        coarse_pose=coarse,
        camera_c2w=camera_c2w,
        corresp=corresp,
        best_coarse_pose_id=best_coarse_pose_id,
        image_np_hwc=image_np_hwc,
        mask_modal=mask_modal,
        box_amodal=box_amodal,
        feature_map_chw=feature_map_chw,
        feature_map_chw_proj=feature_map_chw_proj,
        times=times,
    )


@dataclass
class ObjectPoseResult:
    """One estimated pose, expressed in the frame of the camera that was passed in."""

    object_id: int
    det_id: Any
    T_cam_obj: np.ndarray  # 4x4 model-to-camera, translation in mm
    score: float
    num_inliers: int
    template_id: int
    times: Dict[str, Optional[float]]
    # The raw result, for visualization or debugging. Its `camera_c2w` is the
    # virtual crop camera, and its poses are relative to that camera, not to the
    # one passed to `estimate_image`.
    estimate: PoseEstimate

    @property
    def R(self) -> np.ndarray:
        return self.T_cam_obj[:3, :3]

    @property
    def t(self) -> np.ndarray:
        """Translation as a 3x1 column, in mm."""
        return self.T_cam_obj[:3, 3:]


class PoseEstimator:
    """Estimates object poses in images, holding the expensive state between calls.

    The feature extractor, the object representations and their KNN indices are
    loaded once in `__init__`; `estimate_image` can then be called repeatedly:

        est = PoseEstimator(
            opts=opts, object_database=object_database, device=device, logger=logger
        )
        for image, detections in stream:
            for res in est.estimate_image(image, camera, detections):
                print(res.object_id, res.T_cam_obj)

    `opts` is duck-typed -- anything exposing the fields `estimate_pose_in_image`
    reads (crop, crop_rel_pad, crop_size, grid_cell_size, max_num_queries,
    match_*, pnp_*, final_pose_type, debug) works.
    """

    def __init__(
        self,
        opts: Any,
        object_database: ObjectDataset,
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        timer: Optional[misc_util.Timer] = None,
    ):
        """
        Args:
            opts: Inference options (see the class docstring).
            object_database: An already-loaded object dataset, as
                `MultiviewPoseRefiner` takes. Anything exposing `object_ids` and
                `objects[object_id].representation` works, so this module stays
                independent of `src/`. Its representations must have been loaded
                with `full=True` -- the projector-only subset the refiner uses
                lacks the template cameras and tf-idf options needed here.
            device: Defaults to cuda when available.
            timer: Default timer for the estimate_* calls.
        """

        self.opts = opts
        self.object_database = object_database

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger or logging.get_logger(level=logging.WARNING)
        self.timer = timer or misc_util.Timer(enabled=getattr(opts, "debug", False))

        self._contexts: Dict[int, ObjectContext] = {}
        # Keyed by image size, since the grid only depends on the image when
        # cropping is disabled.
        self._grid_points: Dict[Tuple[int, int], torch.Tensor] = {}

        self.logger.info(f"PoseEstimator on {self.device}")

        # The extractor is the same for every object, so build it once.
        self.extractor = feature_util.make_feature_extractor(opts.extractor_name)
        self.extractor.to(self.device)

        object_ids = list(object_database.object_ids)
        for object_id in object_ids:
            self.add_object(object_id)

    @property
    def object_ids(self) -> List[int]:
        return sorted(self._contexts)

    def add_object(self, object_id: int) -> None:
        """Makes one object available, building its KNN indices."""

        objects = self.object_database.objects
        if object_id not in objects:
            raise ValueError(
                f"Object {object_id} is not in the object database "
                f"(has {sorted(objects)})."
            )

        repre = objects[object_id].representation
        if repre is None:
            raise ValueError(
                f"Object {object_id} has no representation loaded. "
                f"Call object_database.load_representations(full=True) first."
            )

        # `load_representations` defaults to reading only the PCA projectors, which
        # is all the multi-view refiner needs. Both fields below are left at their
        # dataclass defaults by that path, and both are read while building the KNN
        # indices -- the empty camera list silently yields no template indices at
        # all, so check rather than let it through.
        needed = []
        if self.opts.match_template_type == "tfidf" and repre.template_desc_opts is None:
            needed.append("template_desc_opts")
        if (
            self.opts.match_feat_matching_type == "cyclic_buddies"
            and not repre.template_cameras_cam_from_model
        ):
            needed.append("template_cameras_cam_from_model")
        if needed:
            raise ValueError(
                f"Object {object_id}'s representation is missing "
                f"{' and '.join(needed)}. Reload the object database with "
                f"load_representations(repre_version, device, full=True)."
            )

        self._contexts[object_id] = build_object_context(
            repre=repre, opts=self.opts, logger=self.logger
        )

    def _grid_for(self, image_size: Tuple[int, int]) -> torch.Tensor:
        if image_size not in self._grid_points:
            self._grid_points[image_size] = make_grid_points(
                opts=self.opts, image_size=image_size, device=self.device
            )
        return self._grid_points[image_size]

    def _resolve_object_id(self, detection: Dict[str, Any]) -> int:
        object_id = detection.get("object_id", detection.get("obj_id"))
        if object_id is None:
            if len(self._contexts) != 1:
                raise ValueError(
                    f"Detection has no 'object_id' and {len(self._contexts)} objects "
                    f"are loaded ({self.object_ids}), so it is ambiguous."
                )
            return self.object_ids[0]
        if object_id not in self._contexts:
            raise ValueError(
                f"Object {object_id} is not loaded (have {self.object_ids}). "
                f"Call add_object({object_id}) first."
            )
        return int(object_id)

    @staticmethod
    def _resolve_box(detection: Dict[str, Any], mask: np.ndarray) -> AlignedBox2f:
        """Takes the box from the detection, or derives it from the mask."""

        if "bbox_xyxy" in detection:
            x1, y1, x2, y2 = detection["bbox_xyxy"]
        elif "bbox_xywh" in detection:
            x, y, w, h = detection["bbox_xywh"]
            x1, y1, x2, y2 = x, y, x + w, y + h
        else:
            ys, xs = mask.nonzero()
            if len(xs) == 0:
                raise ValueError("Detection mask is empty and no bbox was given.")
            x1, y1, x2, y2 = misc_util.calc_2d_box(xs, ys)
        return AlignedBox2f(left=x1, top=y1, right=x2, bottom=y2)

    def estimate_detection(
        self,
        image_np_hwc: np.ndarray,
        camera_c2w: PinholePlaneCameraModel,
        mask_modal: np.ndarray,
        object_id: int,
        box_amodal: Optional[AlignedBox2f] = None,
        det_id: Any = None,
        timer: Optional[misc_util.Timer] = None,
    ) -> Optional[ObjectPoseResult]:
        """Estimates the pose of one detection. Returns None if no pose was found."""

        if box_amodal is None:
            box_amodal = self._resolve_box({}, mask_modal)

        result = estimate_pose_in_image(
            orig_image_np_hwc=image_np_hwc,
            orig_camera_c2w=camera_c2w,
            orig_mask_modal=mask_modal,
            orig_box_amodal=box_amodal,
            ctx=self._contexts[object_id],
            extractor=self.extractor,
            grid_points=self._grid_for((camera_c2w.width, camera_c2w.height)),
            opts=self.opts,
            device=self.device,
            timer=timer or self.timer,
            logger=self.logger,
        )
        if result is None:
            return None

        # The pose is relative to the virtual crop camera. Take it to the world
        # frame the crop camera was built in, then into the input camera's frame.
        # (When the input camera has identity extrinsics these are the same frame.)
        pose_m2crop = structs.ObjectPose(
            R=result.final_pose["R_m2c"], t=result.final_pose["t_m2c"]
        )
        T_world_obj = result.camera_c2w.T_world_from_eye.dot(
            misc_util.get_rigid_matrix(pose_m2crop)
        )
        T_cam_obj = np.linalg.inv(camera_c2w.T_world_from_eye).dot(T_world_obj)

        return ObjectPoseResult(
            object_id=object_id,
            det_id=det_id,
            T_cam_obj=T_cam_obj,
            score=float(result.final_pose["quality"]),
            num_inliers=int(len(result.final_pose["inliers"])),
            template_id=int(
                result.corresp[result.final_pose["corresp_id"]]["template_id"]
            ),
            times=result.times,
            estimate=result,
        )

    def estimate_image(
        self,
        image_np_hwc: np.ndarray,
        camera_c2w: PinholePlaneCameraModel,
        detections: List[Dict[str, Any]],
        timer: Optional[misc_util.Timer] = None,
    ) -> List[ObjectPoseResult]:
        """Estimates a pose for every detection in one image.

        Args:
            image_np_hwc: RGB image, float32 in [0, 1] (uint8 is accepted and
                divided by 255).
            camera_c2w: Camera of `image_np_hwc`. Poses come back in this frame.
            detections: One dict per instance, with
                `mask` (or `mask_modal`): HxW array, non-zero on the object,
                `object_id`: optional when a single object is loaded,
                `bbox_xywh` or `bbox_xyxy`: optional, derived from the mask if absent,
                `id`: optional, echoed back as `det_id`.
        Returns:
            One result per detection that yielded a pose, in input order.
        """

        image_np_hwc = np.asarray(image_np_hwc)
        if image_np_hwc.dtype == np.uint8:
            image_np_hwc = image_np_hwc.astype(np.float32) / 255.0
        image_np_hwc = misc_util.ensure_three_channels(image_np_hwc)

        if image_np_hwc.shape[:2] != (camera_c2w.height, camera_c2w.width):
            raise ValueError(
                f"Image is {image_np_hwc.shape[1]}x{image_np_hwc.shape[0]} but the "
                f"camera says {camera_c2w.width}x{camera_c2w.height}."
            )

        results = []
        for index, detection in enumerate(detections):
            mask = detection.get("mask", detection.get("mask_modal"))
            if mask is None:
                raise ValueError(
                    f"Detection {index} has no 'mask'. Query points are filtered by "
                    f"the modal mask, so a bounding box alone is not enough."
                )
            mask = np.asarray(mask)
            if mask.shape[:2] != image_np_hwc.shape[:2]:
                raise ValueError(
                    f"Detection {index}: mask is {mask.shape[1]}x{mask.shape[0]}, "
                    f"image is {image_np_hwc.shape[1]}x{image_np_hwc.shape[0]}."
                )
            mask = (mask > 0).astype(np.uint8)

            object_id = self._resolve_object_id(detection)
            result = self.estimate_detection(
                image_np_hwc=image_np_hwc,
                camera_c2w=camera_c2w,
                mask_modal=mask,
                object_id=object_id,
                box_amodal=self._resolve_box(detection, mask),
                det_id=detection.get("id", index),
                timer=timer,
            )
            if result is None:
                self.logger.warning(
                    f"Detection {detection.get('id', index)} "
                    f"(object {object_id}): no pose found."
                )
                continue
            results.append(result)

        return results
