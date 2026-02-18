import numpy as np
from bop_toolkit_lib import inout
from pathlib import Path

import pandas as pd
from foundpose.utils import (
    misc,
    structs,
)

from typing import Dict, Iterable, List, Tuple
from src.utils_multiview.structs.data_structures import View

def load_bop_predictions_csv(path: Path) -> pd.DataFrame:
    """
    Load the poses from a csv file in BOP format and return them in a DataFrame.

    Args:
        path (Path): The path to the csv file.

    Returns:
        det_df (pd.DataFrame): Predictions loaded as a df.
    """
    if not path.is_file():
        return pd.DataFrame()

    # Load the poses
    dets = inout.load_bop_results(path)
    df = list_to_df_predictions(dets)

    # Create pose column
    df = df_convert_rt_to_pose(df)
    return df

def save_bop_predictions_csv(
    predictions: pd.DataFrame, out_csv_path: Path, normalize: bool = False,
    ):
    """
    Save the predictions in BOP format to a CSV file.
    Args:
        predictions: DataFrame of predictions, each row with keys:
            'scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time'.
        out_csv_path: Path to the output CSV file.
        normalize: Whether to normalize the scores (per scene).
    Returns:
        out_csv_path: Path to the saved CSV file.
    """
    # Convert to bop format
    predictions = df_convert_pose_to_rt(predictions)
    predictions = df_predictions_to_list(predictions)

    # Normalize scores
    if normalize:
        predictions = normalize_scores(predictions)

    print("Number of predictions: ", len(predictions))
    print("Wrote:", out_csv_path)
    Path(out_csv_path).parent.mkdir(exist_ok=True, parents=True)
    inout.save_bop_results(out_csv_path, predictions)
    return out_csv_path


def load_bop_predictions_scene(
    camera_names: Iterable[str],
    predictions_path: Path,
    score_threshold: float = None,
    logger=None,
) -> pd.DataFrame:
    """
    Load all predictions from the given BOP csv path for all cameras

    Args:
        camera_names (Iterable[str]): The names of the cameras to load predictions for.
        predictions_path (Path): The path to the csv file or to directory containing the predictions.
        score_threshold (float, optional): The minimum score for predictions to be included. Defaults to None.
        logger (optional): Logger for logging information. Defaults to None.

    Returns:
        pd.DataFrame: All predictions for all cameras concatenated.
    """
    predictions = []

    for camera_id in camera_names:
        csv_path = None
        if predictions_path.is_file():
            csv_path = predictions_path
        elif camera_id is not None and camera_id != "":
            matches = list(predictions_path.glob(f"*{camera_id}*.csv"))
            csv_path = matches[0] if matches else None

        if csv_path is None:
            continue

        df = load_bop_predictions_csv(csv_path)
        if df is None:
            continue

        df["camera_id"] = camera_id

        if score_threshold is not None:
            df = df[df["score"] >= score_threshold].reset_index(drop=True)

        predictions.append(df)

        if logger:
            logger.info(f"Loaded {len(df)} predictions for camera {camera_id}")

    if not predictions:
        raise ValueError("No predictions found for any camera.")

    return pd.concat(predictions, ignore_index=True)


def df_camera_to_world_poses(
    predictions: pd.DataFrame, views: List[View]
) -> pd.DataFrame:
    """Transform all predictions from camera to world coordinates.

    Args:
        predictions (pd.DataFrame): The predictions to transform. Must contain 'camera_id', 'im_id' and 'pose' columns.
        views (List[View]): The views to use for the transformation.

    Returns:
        predictions (pd.DataFrame): The transformed predictions.
    """
    assert "camera_id" in predictions.columns, (
        "predictions must contain 'camera_id' column"
    )
    assert "im_id" in predictions.columns, "predictions must contain 'im_id' column"
    assert "pose" in predictions.columns, "predictions must contain 'pose' column"

    predictions = predictions.copy()
    view_map = {(v.camera_id, v.image_id): v for v in views}

    for i, row in predictions.iterrows():
        view = view_map[(row.camera_id, row.im_id)]

        # Camera pose in world coordinates
        T_wc = view.camera.T_world_from_eye

        # get object pose in camera coordinates
        T_cm = misc.get_rigid_matrix(row.pose)

        # Transform object pose to world coordinates
        T_wm = T_wc @ T_cm
        predictions.at[i, "pose"] = structs.ObjectPose(
            R=T_wm[:3, :3],
            t=T_wm[:3, 3],
        )

    # Drop the camera_id column, poses are now in world coordinates
    return predictions.drop(columns=["camera_id"])


def df_world_to_camera_poses(
    predictions: pd.DataFrame, views: List[View]
) -> pd.DataFrame:
    """Transform all predictions from world to camera coordinates.

    Args:
        predictions (pd.DataFrame): The predictions to transform.
        views (List[View]): The views to use for the transformation.
    Returns:
        transformed_predictions (pd.DataFrame): The transformed predictions. Contains 'camera_id' and 'im_id' columns.
    """
    output = []

    for _, row in predictions.iterrows():
        for view in views:
            # Camera pose in world coordinates
            T_wc = view.camera.T_world_from_eye

            # Object pose in world coordinates
            T_wm = misc.get_rigid_matrix(row.pose)

            # Transform object pose to camera coordinates
            T_cm = np.linalg.inv(T_wc) @ T_wm

            new_row = row.copy()
            new_row["pose"] = structs.ObjectPose(
                R=T_cm[:3, :3],
                t=T_cm[:3, 3],
            )
            new_row["camera_id"] = view.camera_id
            new_row["im_id"] = view.image_id

            output.append(new_row)

    return pd.DataFrame(output)


def df_predictions_to_list(predictions: pd.DataFrame) -> List[Dict]:
    """Transform all predictions from a DataFrame to a list of dictionaries.

    Args:
        predictions (pd.DataFrame): The predictions to transform.
    Returns:
        predictions (List[Dict]): The transformed predictions.
    """
    final_predictions = []
    for _, row in predictions.iterrows():
        new_prediction = row.copy().to_dict()
        final_predictions.append(new_prediction)
    return final_predictions

def list_to_df_predictions(predictions: List[Dict]) -> pd.DataFrame:
    """Transform all predictions from a list of dictionaries to a DataFrame.

    Args:
        predictions (List[Dict]): The predictions to transform.
    Returns:
        predictions (pd.DataFrame): The transformed predictions.
    """
    df = pd.DataFrame(predictions)
    return df

def normalize_scores(predictions: List[Dict]) -> List[Dict]:
    """
    Normalize the scores of the predictions to be in the range [0, 1].
    """
    if len(predictions) == 0:
        return predictions
    
    # Group predictions by scene_id and im_id
    predictions = sorted(predictions, key=lambda x: (x["scene_id"], x["im_id"]))

    # Extract scores and find min/max
    # for each group of predictions
    grouped_predictions = {}
    for pred in predictions:
        key = (pred["scene_id"], pred["im_id"])
        if key not in grouped_predictions:
            grouped_predictions[key] = []
        grouped_predictions[key].append(pred)
    
    # Normalize scores within each group
    for key, group in grouped_predictions.items():
        scores = [pred["score"] for pred in group]
        min_score = min(scores)
        max_score = max(scores)

        if min_score == max_score:
            # Avoid division by zero
            for pred in group:
                pred["score"] = 1.0
        else:
            a = 0.1
            for pred in group:
                pred["score"] = a + (pred["score"] - min_score) * (1-a) / (max_score - min_score)

    # Flatten the grouped predictions back to a list
    normalized_predictions = []
    for group in grouped_predictions.values():
        normalized_predictions.extend(group)
    return normalized_predictions

def df_convert_pose_to_rt(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Convert pose in RigidTransform format to separate R and t.
    Args:
        predictions (pd.DataFrame): DataFrame of predictions, each row with keys:
            'scene_id', 'im_id', 'obj_id', 'score', 'pose', 'time'.
    Returns:
        predictions_rt (pd.DataFrame): DataFrame of predictions with 'R' and 't' keys instead of 'pose'.
    """
    assert "pose" in predictions.columns, "DataFrame must contain 'pose' column"
    assert predictions["pose"].apply(lambda x: isinstance(x, structs.RigidTransform)).all(), "All 'pose' entries must be of type RigidTransform"

    predictions["R"] = predictions.apply(
        lambda row: row["pose"].R, axis=1
    )
    predictions["t"] = predictions.apply(
        lambda row: row["pose"].t, axis=1
    )
    return predictions.drop(columns=["pose"])

def df_convert_rt_to_pose(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Convert pose from separate R and t to RigidTransform format.
    Args:
        predictions (pd.DataFrame): DataFrame of predictions, each row with keys:
            'scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time'.
    Returns:
        predictions_pose (pd.DataFrame): DataFrame of predictions with 'pose' key instead of 'R' and 't'.
    """
    predictions["pose"] = predictions.apply(
        lambda row: structs.RigidTransform(R=row["R"], t=row["t"]), axis=1
    )
    return predictions.drop(columns=["R", "t"])
