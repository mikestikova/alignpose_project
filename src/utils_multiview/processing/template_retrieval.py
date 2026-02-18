import json
import numpy as np
from src.utils_multiview.constants import OUTPUT_DIR

def se3_geodesic_distance(pose1, pose2, separate_distances=False):
    """
    Calculates an approximation of the Geodesic distance between two SE(3) poses.
    This is a simplified version using separate translation and rotation distances.

    Args:
        pose1: A 4x4 NumPy array representing the first SE(3) pose.
        pose2: A 4x4 NumPy array representing the second SE(3) pose.

    Returns:
        An approximate geodesic distance between the two poses.
    """

    # 1. Extract the rotation matrices and translation vectors
    rotation_matrix1 = pose1[:3, :3]
    rotation_matrix2 = pose2[:3, :3]
    translation_vector1 = pose1[:3, 3] / 1000 # convert to meters
    translation_vector2 = pose2[:3, 3] / 1000 # convert to meters

    # 2. Calculate the translation distance (Euclidean)
    translation_distance = np.linalg.norm(translation_vector1 - translation_vector2)

    # 3. Calculate the rotation distance (Frobenius norm of the log of relative rotation)
    relative_rotation = (
        rotation_matrix1 @ rotation_matrix2.T
    )  # equivalent to R1 @ inv(R2)
    trace = np.trace(relative_rotation)
    rotation_distance = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))
    # rotation_log = logm(relative_rotation)
    # rotation_distance = np.linalg.norm(rotation_log, ord='fro')
    # 4. Combine the distances
    distance = np.sqrt(translation_distance**2 + rotation_distance**2)
    if separate_distances:
        return translation_distance, rotation_distance
    return distance

import os
def get_template_by_pose(pose_cm, obj_id, dataset):
    template_version = "v1"
    # Load the templates
    dataset_torch_relpath = os.path.join(
        "templates",
        template_version,
        dataset,
        str(obj_id),
    )

    template_metadata_path = (
        f"{OUTPUT_DIR}/{dataset_torch_relpath}/metadata.json"
    )
    template_metadata = json.load(open(template_metadata_path))
    template_poses = np.zeros((len(template_metadata), 4, 4))
    for i, template in enumerate(template_metadata):
        template_pose_cm = np.array(template["cameras"]["ModelViewMatrix"])
        template_poses[i] = template_pose_cm

    # Find the closest template
    distances = [
        se3_geodesic_distance(pose_cm, p) for p in template_poses
    ]
    ind_smallest_distances = np.argmin(distances)

    # return the closest template
    return template_metadata[ind_smallest_distances]
