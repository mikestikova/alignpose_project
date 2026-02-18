from typing import Any, Dict, List, NamedTuple, Optional, Tuple


class GenTemplatesOpts(NamedTuple):
    """Options that can be specified via the command line."""

    version: str
    object_dataset: str
    path_to_models: Optional[str] = None  # Path template to the object models, e.g. `path/to/models/{obj_id}.ply`
    object_lids: Optional[List[int]] = None

    # Viewpoint options.
    min_num_viewpoints: int = 57
    num_inplane_rotations: int = 14

    # Rendering options.
    render_res: int = 1024

    # Cropping options.
    crop: bool = True
    crop_rel_pad: float = 0.3
    crop_size: Tuple[int, int] = (420, 420)

    # Other options.
    save_templates: bool = True
    overwrite: bool = True
    debug: bool = True


class GenRepreOpts(NamedTuple):
    """Options that can be specified via the command line."""

    version: str
    templates_version: str
    object_dataset: str
    path_to_models: Optional[str] = None  # Path template to the object models, e.g. `path/to/models/{obj_id}.ply`
    object_lids: Optional[List[int]] = None

    # Feature extraction options.
    extractor_name: str = "dinov2_vits14_reg"
    grid_cell_size: float = 14.0

    # Feature PCA options.
    apply_pca: bool = True
    pca_components: int = 256
    pca_whiten: bool = False
    pca_max_samples_for_fitting: int = 100000

    # Other options.
    overwrite: bool = True
    debug: bool = True


class RefineOpts(NamedTuple):
    """Options that can be specified via the command line."""

    input_path: str
    version: str
    repre_version: str
    object_dataset: str
    path_to_models: Optional[str] = None  # Path template to the object models, e.g. `path/to/models/{obj_id}.ply`

    # Feature extraction & cropping.
    extractor_name: str = "dinov2_vitl14"
    grid_cell_size: float = 14.0
    crop_size: Tuple[int, int] = (420, 420)
    crop_rel_pad: float = 0.2

    # Refinement.
    loss_fn: str = "scaled_barron(-5, 0.5)"
    num_iters: int = 30
    use_offline_templates: bool = False

    # NMS.
    nms: bool = True
    nms_threshold: float = 0.5
    nms_box_type: str = "axis-aligned"  # 'axis-aligned' or 'oriented' or 'convex-hull' or 'sphere'
    nms_mode: str = "inter-object"  # 'inter-object' or 'per-object'

    # Output.
    vis_results: bool = True
    debug: bool = True
