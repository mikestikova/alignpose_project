from typing import List, Tuple
import torch
from foundpose.utils import misc
import pandas as pd
import trimesh
from src.utils_multiview.datasets.object_dataset import ObjectDataset

def nms_3d(
    prediction_boxes: List[Tuple[float, trimesh.Trimesh]], iou_threshold: float = 0.5
) -> torch.Tensor:
    """
    Source: https://github.com/GiulioRusso/NMS-3D/blob/main/nms_3d/nms_3d.py

    Perform 3D Non-Maximum Suppression on a set of bounding boxes.
    Args:
        prediction_boxes: A list of tuples (confidence score, bbox (trimesh.Trimesh)).
        iou_threshold: Intersection over union threshold between 0 and 1. Default is 0.5.

    Returns:
        Tensor containing the selected bounding boxes after applying NMS.
        Tensor of indices of the selected boxes in the original order.

    Example:
        # list of tuples: (score, box)
        prediction_boxes = [
            (0.95, box1),  # high-score box
            (0.90, box2),  # overlapping box (should be suppressed)
            (0.85, box3),  # distant box (should be kept)
            (0.80, box4),  # overlapping with previous (should be suppressed)
            (0.75, box5)  # another distant box (should be kept)
        ]

        # define the IoU threshold
        iou_threshold = 0.5

        # run the 3D Non-Maximum Suppression
        filtered_boxes, selected_indices = nms_3d(prediction_boxes=prediction_boxes, iou_threshold=iou_threshold)

    """

    # check iou value
    if not (0 <= iou_threshold <= 1):
        raise ValueError(
            f"iou_threshold must be between 0 and 1. Got {iou_threshold} instead"
        )

    best_boxes_hist = []
    selected_indices = []

    # Keep track of original indices
    orig_indices = torch.arange(len(prediction_boxes))

    while len(prediction_boxes) > 0:
        # sort predictions list by score in descending order
        sorted_indices = [
            i
            for i, _ in sorted(
                enumerate(prediction_boxes), key=lambda x: x[1][0], reverse=True
            )
        ]
        prediction_boxes = [prediction_boxes[i] for i in sorted_indices]
        orig_indices = orig_indices[sorted_indices]

        highest_score_box = prediction_boxes[0]
        highest_score_idx = orig_indices[0].item()

        # calculate intersection volume
        box1 = highest_score_box[1]  # box part of the tuple

        def safe_intersection_volume(box1, box2):
            mesh = trimesh.boolean.intersection([box1, box2])
            return mesh.volume if not mesh.is_empty and mesh.volume != 0 else 0.0

        intersection_volume = torch.tensor(
            [safe_intersection_volume(box1, box2) for (_, box2) in prediction_boxes]
        )

        # if all boxes have zero intersection volume, remove this box and continue
        if torch.all(intersection_volume == 0):
            prediction_boxes = prediction_boxes[1:]
            orig_indices = orig_indices[1:]
            continue

        # calculate volumes
        highest_score_box_volume = torch.tensor(highest_score_box[1].volume)

        row_volume = torch.tensor(
            [box.volume for (_, box) in prediction_boxes]
        )  # box part of the tuple

        # calculate iou values
        iou_values = intersection_volume / (
            highest_score_box_volume + row_volume - intersection_volume
        )

        # create a mask to threshold over the boxes that need to be suppressed
        iou_threshold_mask = iou_values > iou_threshold

        # select the rows with iou > threshold
        iou_threshold_boxes = [
            prediction_boxes[i]
            for i in range(len(prediction_boxes))
            if iou_threshold_mask[i]
        ]

        # save the highest score box into the best boxes list
        best_boxes_hist.append(iou_threshold_boxes[0])
        selected_indices.append(highest_score_idx)

        # remove threshold boxes
        keep_mask = ~iou_threshold_mask
        prediction_boxes = [
            prediction_boxes[i] for i in range(len(prediction_boxes)) if keep_mask[i]
        ]
        orig_indices = orig_indices[keep_mask]

    return best_boxes_hist, torch.tensor(selected_indices)


def get_bbox(
    model: trimesh.Trimesh, box_type: str = "oriented"
) -> trimesh.primitives.Box:
    """
    Get the bounding box of a mesh based on the specified box type.
    
    Args:
        model (trimesh.Trimesh): The input mesh for which to compute the bounding box.
        box_type (str): The type of bounding box to compute. Supported types are "axis-aligned", "oriented", "convex-hull", and "sphere". Default is "oriented".
    Returns:
        trimesh.primitives.Box: The computed bounding box of the mesh.
    """
    if box_type == "axis-aligned":
        return model.bounding_box
    elif box_type == "oriented":
        return model.bounding_box_oriented
    elif box_type == "convex-hull":
        return model.convex_hull
    elif box_type == "sphere":
        return model.bounding_sphere
    else:
        raise ValueError(
            f"Unknown box_type: {box_type}. Supported types are 'oriented', 'axis-aligned', 'convex-hull', and 'sphere'."
        )


def nms_candidates_per_object(
    candidates: pd.DataFrame,
    object_dataset: ObjectDataset,
    iou_threshold: float = 0.4,
    box_type: str = "oriented"
) -> pd.DataFrame:
    """
    Perform NMS on candidate poses for each object separately.
    Args:
        candidates (pd.DataFrame): DataFrame of candidate poses to filter, must contain columns "obj_id", "pose", and "score".
        object_dataset (ObjectDataset): The dataset containing object meshes.
        iou_threshold (float): IoU threshold for NMS.
        box_type (str): Type of bounding box to use for NMS
    Returns:
        pd.DataFrame: DataFrame of filtered candidate poses after NMS.
    """
    filtered_candidates = []
    per_obj_bboxes = {}

    # gather all bounding boxes per object
    for i in range(len(candidates)):
        object_id = candidates["obj_id"][i]
        score = candidates["score"][i]
        candidate_pose_wm = candidates["pose"][i]
        candidate_pose_wm = misc.get_rigid_matrix(candidate_pose_wm)

        # get 3D oriented bounding box
        mesh2 = object_dataset.objects[object_id].mesh.copy()
        mesh2.apply_transform(candidate_pose_wm)

        bbox = get_bbox(model=mesh2, box_type=box_type)

        per_obj_bboxes.setdefault(object_id, []).append([score, bbox])

    # perform nms per object
    for object_id, all_bboxes in per_obj_bboxes.items():
        if len(all_bboxes) == 0:
            continue

        # call nms_3d on the bounding boxes
        _, nms_indices = nms_3d(
            prediction_boxes=all_bboxes,
            iou_threshold=iou_threshold,
        )

        # update the candidates with the nms results
        obj_candidates = candidates[candidates["obj_id"] == object_id].reset_index(
            drop=True
        )
        filtered_obj_candidates = obj_candidates.loc[nms_indices].reset_index(drop=True)
        filtered_candidates.append(filtered_obj_candidates)

    filtered_candidates = pd.concat(filtered_candidates).reset_index(drop=True)
    return filtered_candidates

def nms_candidates_inter_object(
    candidates: pd.DataFrame,
    object_dataset: ObjectDataset,
    iou_threshold: float = 0.4,
    box_type: str = "axis-aligned"
) -> pd.DataFrame:
    """Perform NMS on candidate poses across all objects.
    Args:        
        candidates (pd.DataFrame): DataFrame of candidate poses to filter, must contain columns "obj_id", "pose", and "score".
        object_dataset (ObjectDataset): The dataset containing object meshes.
        iou_threshold (float): IoU threshold for NMS.
        box_type (str): Type of bounding box to use for NMS
    Returns:        
        pd.DataFrame: DataFrame of filtered candidate poses after NMS.
    """
    # gather all bounding boxes per object
    all_bboxes = []
    for i in range(len(candidates)):
        object_id = candidates["obj_id"][i]
        score = candidates["score"][i]
        candidate_pose_wm = candidates["pose"][i]
        candidate_pose_wm = misc.get_rigid_matrix(candidate_pose_wm)

        # get 3D oriented bounding box
        mesh2 = object_dataset.objects[object_id].mesh.copy()
        mesh2.apply_transform(candidate_pose_wm)

        bbox = get_bbox(model=mesh2, box_type=box_type)

        all_bboxes.append([score, bbox])

    # perform nms per object
    if len(all_bboxes) == 0:
        return candidates

    # call nms_3d on the bounding boxes
    _, nms_indices = nms_3d(
        prediction_boxes=all_bboxes,
        iou_threshold=iou_threshold,
    )

    # update the candidates with the nms results
    filtered_candidates = candidates.loc[nms_indices].reset_index(drop=True)

    return filtered_candidates
        

def nms_candidates(
    candidates: pd.DataFrame,
    object_dataset: ObjectDataset,
    iou_threshold: float = 0.4,
    box_type: str = "axis-aligned",
    nms_mode: str = "inter-object"
) -> pd.DataFrame:
    """
    Perform NMS on candidate poses across all objects.
    Args:
        candidates (pd.DataFrame): DataFrame of candidate poses to filter, must contain columns "obj_id", "pose", and "score".
        object_dataset (ObjectDataset): The dataset containing object meshes.
        iou_threshold (float): IoU threshold for NMS.
        box_type (str): Type of bounding box to use for NMS.
    Returns:
        pd.DataFrame: DataFrame of filtered candidate poses after NMS.
    """
    if nms_mode == "per-object":
        return nms_candidates_per_object(
            candidates, object_dataset, iou_threshold, box_type
        )

    elif nms_mode == "inter-object":
        return nms_candidates_inter_object(
            candidates, object_dataset, iou_threshold, box_type
        )
    else:
        raise ValueError(f"Unknown nms_mode: {nms_mode}. Supported modes are 'per-object' and 'inter-object'.")
    
        
