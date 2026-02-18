#!/usr/bin/env python3

"""Synthesizes object templates."""

from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import os

# set up visible gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import cv2

import numpy as np

from bop_toolkit_lib import inout, dataset_params

import bop_toolkit_lib.config as bop_config

from foundpose.utils import (
    misc as foundpose_misc,
    json_util,
    config_util,
    logging,
    misc,
    structs,
)

from foundpose.utils.structs import AlignedBox2f, PinholePlaneCameraModel

from foundpose.utils import geometry, renderer_builder
from foundpose.utils.renderer_base import RenderType

from pathlib import Path


class GenTemplatesOpts(NamedTuple):
    """Options that can be specified via the command line."""

    version: str
    object_dataset: str
    object_lids: Optional[List[int]] = None

    # Viewpoint options.
    num_viewspheres: int = 1
    min_num_viewpoints: int = 57
    num_inplane_rotations: int = 14
    images_per_view: int = 1

    # Mesh pre-processing options.
    max_num_triangles: int = 20000
    back_face_culling: bool = False
    texture_size: Tuple[int, int] = (1024, 1024)

    # Rendering options.
    ssaa_factor: float = 4.0
    background_type: str = "black"
    light_type: str = "multi_directional"

    # Cropping options.
    crop: bool = True
    crop_rel_pad: float = 0.2
    crop_size: Tuple[int, int] = (420, 420)

    # Other options.
    features_patch_size: int = 14
    save_templates: bool = True
    overwrite: bool = True
    debug: bool = True


def synthesize_templates(opts: GenTemplatesOpts) -> None:
    datasets_path = bop_config.datasets_path

    # Fix the random seed for reproducibility.
    np.random.seed(0)

    # Prepare a logger and a timer.
    logger = logging.get_logger(level=logging.INFO if opts.debug else logging.WARNING)
    timer = misc.Timer(enabled=opts.debug)
    timer.start()

    # Get IDs of objects to process.
    object_lids = opts.object_lids
    bop_model_props = dataset_params.get_model_params(
        datasets_path=datasets_path, dataset_name=opts.object_dataset
    )
    if object_lids is None:
        # If local (object) IDs are not specified, synthesize templates for all objects
        # in the specified dataset.
        object_lids = bop_model_props["obj_ids"]

    # Get properties of the test split of the specified dataset.
    bop_test_split_props = dataset_params.get_split_params(
        datasets_path=datasets_path, dataset_name=opts.object_dataset, split="test"
    )
    # Get properties of the default camera for the specified dataset.
    # bop_camera = dataset_params.get_camera_params(
    #     datasets_path=datasets_path, dataset_name=opts.object_dataset
    # )
    bop_camera = inout.load_cam_params(
        os.path.join(datasets_path, opts.object_dataset, "camera_xyz.json")
    )
    logger.info(f"Bop camera details are read ")

    print("Object lids: ", object_lids)

    # print("Bop camera params: \n", bop_camera)

    # print("Bop test split props: \n", bop_test_split_props)

    # Prepare a camera for the template (square viewport of a size divisible by the patch size).
    bop_camera_width = bop_camera["im_size"][0]
    bop_camera_height = bop_camera["im_size"][1]
    max_image_side = max(bop_camera_width, bop_camera_height)
    image_side = opts.features_patch_size * int(
        max_image_side / opts.features_patch_size
    )
    camera_model = PinholePlaneCameraModel(
        width=image_side,
        height=image_side,
        f=(bop_camera["K"][0, 0], bop_camera["K"][1, 1]),
        c=(
            bop_camera["K"][0, 2] - 0.5 * (bop_camera_width - image_side),
            bop_camera["K"][1, 2] - 0.5 * (bop_camera_height - image_side),
        ),
    )
    # Prepare a camera for rendering, upsampled for SSAA (supersampling anti-aliasing).
    render_camera_model = PinholePlaneCameraModel(
        width=int(camera_model.width * opts.ssaa_factor),
        height=int(camera_model.height * opts.ssaa_factor),
        f=(
            camera_model.f[0] * opts.ssaa_factor,
            camera_model.f[1] * opts.ssaa_factor,
        ),
        c=(
            camera_model.c[0] * opts.ssaa_factor,
            camera_model.c[1] * opts.ssaa_factor,
        ),
    )
    print("camera model created")

    # Build a renderer.
    render_types = [RenderType.COLOR, RenderType.MASK]
    renderer_type = renderer_builder.RendererType.PYRENDER_RASTERIZER
    renderer = renderer_builder.build(renderer_type=renderer_type)

    # Define radii of the view spheres on which we will sample viewpoints.
    # The specified number of radii is sampled uniformly in the range of
    # camera-object distances from the test split of the specified dataset.
    depth_range = bop_test_split_props["depth_range"]
    min_depth = np.min(depth_range)
    max_depth = np.max(depth_range)
    depth_range_size = max_depth - min_depth
    depth_cell_size = depth_range_size / float(opts.num_viewspheres)
    viewsphere_radii = []
    for depth_cell_id in range(opts.num_viewspheres):
        viewsphere_radii.append(min_depth + (depth_cell_id + 0.5) * depth_cell_size)

    # Generate viewpoints from which the object model will be rendered.
    views_sphere = []
    for radius in viewsphere_radii:
        views_sphere += foundpose_misc.sample_views(
            min_n_views=opts.min_num_viewpoints,
            radius=radius,
            mode="fibonacci",
        )[0]
    logger.info(f"Sampled points on the sphere: {len(views_sphere)}")

    # Add in-plane rotations.
    if opts.num_inplane_rotations == 1:
        views = views_sphere
    else:
        inplane_angle = 2 * np.pi / opts.num_inplane_rotations
        views = []
        for view_sphere in views_sphere:
            for inplane_id in range(opts.num_inplane_rotations):
                R_inplane = geometry.rotation_matrix_numpy(
                    inplane_angle * inplane_id, np.array([0, 0, 1])
                )[:3, :3]
                views.append(
                    {
                        "R": R_inplane.dot(view_sphere["R"]),
                        "t": R_inplane.dot(view_sphere["t"]),
                    }
                )
    logger.info(f"Number of views: {len(views)}")

    timer.elapsed("Time for setting up the stage")

    # Generate templates for each specified object.
    for object_lid in object_lids:
        logging.log_heading(logger, f"Object {object_lid} from {opts.object_dataset}")
        timer.start()

        # Prepare output folder.
        dataset_torch_relpath = os.path.join(
            "templates",
            opts.version,
            opts.object_dataset,
            str(object_lid),
        )
        output_dir = os.path.join(
            bop_config.output_path,
            dataset_torch_relpath,
        )
        print("output_dir: ", bop_config.output_path)
        if os.path.exists(output_dir) and not opts.overwrite:
            raise ValueError(f"Output directory already exists: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output will be saved to: {output_dir}")

        # Save parameters to a file.
        config_path = os.path.join(output_dir, "config.json")
        json_util.save_json(config_path, opts)

        # Prepare folder for saving templates.
        templates_rgb_dir = os.path.join(output_dir, "rgb")
        if opts.save_templates:
            os.makedirs(templates_rgb_dir, exist_ok=True)

        templates_depth_dir = os.path.join(output_dir, "depth")
        if opts.save_templates:
            os.makedirs(templates_depth_dir, exist_ok=True)

        templates_mask_dir = os.path.join(output_dir, "mask")
        if opts.save_templates:
            os.makedirs(templates_mask_dir, exist_ok=True)

        # Add the model to the renderer.
        model_path = bop_model_props["model_tpath"].format(obj_id=object_lid)
        renderer.add_object_model(obj_id=object_lid, model_path=model_path, debug=True)

        # Prepare a metadata list.
        metadata_list = []

        timer.elapsed("Time for preparing object data")

        template_list = []
        template_counter = 0
        for view_id, view in enumerate(views):
            logger.info(
                f"Rendering object {object_lid} from {opts.object_dataset}, view {view_id}/{len(views)}..."
            )
            for _ in range(opts.images_per_view):
                # if template_counter == 5:
                #     break

                timer.start()

                # Transformation from model to camera.
                trans_m2c = structs.RigidTransform(R=view["R"], t=view["t"])

                # Transformation from camera to model.
                R_c2m = trans_m2c.R.T
                trans_c2m = structs.RigidTransform(R=R_c2m, t=-R_c2m.dot(trans_m2c.t))

                # Camera model for rendering.
                trans_c2m_matrix = misc.get_rigid_matrix(trans_c2m)
                render_camera_model_c2w = PinholePlaneCameraModel(
                    width=render_camera_model.width,
                    height=render_camera_model.height,
                    f=render_camera_model.f,
                    c=render_camera_model.c,
                    T_world_from_eye=trans_c2m_matrix,
                )

                # Rendering.
                output = renderer.render_object_model(
                    obj_id=object_lid,
                    camera_model_c2w=render_camera_model_c2w,
                    render_types=render_types,
                    return_tensors=False,
                    debug=False,
                )

                timer.elapsed("Time for rendering")
                timer.start()

                # Convert rendered mask.
                if RenderType.MASK in output:
                    output[RenderType.MASK] = (255 * output[RenderType.MASK]).astype(
                        np.uint8
                    )

                # Calculate 2D bounding box of the object and make sure
                # it is within the image.
                ys, xs = output[RenderType.MASK].nonzero()
                box = np.array(foundpose_misc.calc_2d_box(xs, ys))
                object_box = AlignedBox2f(
                    left=box[0],
                    top=box[1],
                    right=box[2],
                    bottom=box[3],
                )

                if (
                    object_box.left == 0
                    or object_box.top == 0
                    or object_box.right == render_camera_model_c2w.width - 1
                    or object_box.bottom == render_camera_model_c2w.height - 1
                ):
                    raise ValueError("The model does not fit the viewport.")

                timer.elapsed("Time for mask processing")
                # Optionally crop the object region.
                if opts.crop:
                    timer.start()
                    # Get box for cropping.
                    crop_box = foundpose_misc.calc_crop_box(
                        box=object_box,
                        make_square=True,
                    )

                    # Manually increase the size of the crop box around the object.
                    box_width = crop_box.width
                    box_height = crop_box.height
                    pad_x = opts.crop_rel_pad * box_width
                    pad_y = opts.crop_rel_pad * box_height

                    crop_box_padded = AlignedBox2f(
                        left=max(0, crop_box.left - pad_x),
                        top=max(0, crop_box.top - pad_y),
                        right=min(
                            render_camera_model_c2w.width - 1, crop_box.right + pad_x
                        ),
                        bottom=min(
                            render_camera_model_c2w.height - 1, crop_box.bottom + pad_y
                        ),
                    )
                    crop_box = crop_box_padded

                    for output_key in output.keys():
                        if output_key in [RenderType.COLOR]:
                            # Crop and resize the color image.
                            output[output_key] = misc.crop_and_resize_image(
                                image=output[output_key],
                                crop_box=crop_box,
                                output_size=opts.crop_size,
                                interpolation=cv2.INTER_LINEAR,
                            )
                        elif output_key in [RenderType.MASK]:
                            # Crop and resize the binary mask.
                            output[output_key] = misc.crop_and_resize_image(
                                image=output[output_key],
                                crop_box=crop_box,
                                output_size=opts.crop_size,
                                interpolation=cv2.INTER_NEAREST,
                            )

                # Record the template in the template list.
                template_list.append(
                    {
                        "seq_id": template_counter,
                    }
                )

                # Save template rgb, depth and mask.
                timer.start()
                rgb_image = np.asarray(255.0 * output[RenderType.COLOR], np.uint8)

                rgb_path = os.path.join(
                    templates_rgb_dir, f"template_{template_counter:04d}.png"
                )
                logger.info(f"Saving template RGB {template_counter} to: {rgb_path}")
                inout.save_im(rgb_path, rgb_image)

                # Save template mask.
                mask_path = os.path.join(
                    templates_mask_dir, f"template_{template_counter:04d}.png"
                )
                logger.info(
                    f"Saving template binary mask {template_counter} to: {mask_path}"
                )
                inout.save_im(mask_path, output[RenderType.MASK])

                data = {
                    "dataset": opts.object_dataset,
                    "lid": object_lid,
                    "template_id": template_counter,
                    "rgb_image_path": rgb_path,
                    "binary_mask_path": mask_path,
                }
                timer.elapsed("Time for template saving")

                metadata_list.append(data)

                template_counter += 1

        # Save the metadata to be read from object repre.
        metadata_path = os.path.join(output_dir, "metadata.json")
        json_util.save_json(metadata_path, metadata_list)


def main() -> None:
    opts = config_util.load_opts_from_json_or_command_line(GenTemplatesOpts)[0]

    synthesize_templates(opts)


if __name__ == "__main__":
    main()
