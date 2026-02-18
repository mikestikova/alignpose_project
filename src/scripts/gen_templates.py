#!/usr/bin/env python3

"""Synthesizes object templates."""

import os
import numpy as np
import cv2
from tqdm import tqdm


os.environ["PYOPENGL_PLATFORM"] = "egl"

from foundpose.utils import (
    config_util,
    renderer_builder,
    logging,
    json_util,
    misc,
    geometry,
    structs,
)
from foundpose.utils.renderer_base import RenderType
from foundpose.utils.structs import (
    AlignedBox2f,
    PinholePlaneCameraModel,
)

from src.utils_multiview.opts import GenTemplatesOpts
from src.utils_multiview.datasets.object_dataset import make_object_dataset
from src.utils_multiview.constants import OUTPUT_DIR

from bop_toolkit_lib import inout

RENDER_TYPES = [RenderType.COLOR, RenderType.MASK]


def synthesize_templates(
    opts: GenTemplatesOpts,
):
    """
    Generates object templates by rendering the object model from multiple viewpoints.

    Args:
        opts: Options for template generation.
    """

    # Fix the random seed for reproducibility.
    np.random.seed(0)

    # Prepare a logger.
    logger = logging.get_logger(level=logging.INFO if opts.debug else logging.WARNING)

    # Create object dataset
    logger.info(f"Creating object dataset for {opts.object_dataset}...")
    object_dataset = make_object_dataset(
        dataset_name=opts.object_dataset,
        model_tpath=opts.path_to_models,
    )
    logger.info(f"Object dataset created with {len(object_dataset.object_ids)} objects.")

    # Prepare objects
    object_lids = opts.object_lids
    if object_lids is None:
        # If local (object) IDs are not specified, synthesize templates for all objects
        # in the specified dataset.
        object_lids = object_dataset.object_ids

    # Prepare a render camera model.
    render_camera_model = misc.get_default_camera_model(render_res=opts.render_res)

    # Build a renderer.
    renderer_type = renderer_builder.RendererType.PYRENDER_RASTERIZER
    renderer = renderer_builder.build(
        renderer_type=renderer_type, model_path=object_dataset.model_tpath
    )

    # Generate templates for each specified object.
    for num_obj, object_lid in enumerate(object_lids):
        logger.info(f"Object {object_lid},  [{num_obj + 1}/{len(object_lids)}]")
        if object_lid not in object_dataset.object_ids:
            logger.warning(
                f"Object ID {object_lid} not found in the dataset. Skipping."
            )
            continue

        # Prepare output folder.
        dataset_torch_relpath = os.path.join(
            "templates",
            opts.version,
            object_dataset.dataset_name,
            str(object_lid),
        )
        output_dir = os.path.join(
            OUTPUT_DIR,
            dataset_torch_relpath,
        )
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

        templates_mask_dir = os.path.join(output_dir, "mask")
        if opts.save_templates:
            os.makedirs(templates_mask_dir, exist_ok=True)

        # Add the model to the renderer.
        model_path = object_dataset.objects[object_lid].model_path
        renderer.add_object_model(
            obj_id=object_lid, model_path=model_path, center_model=True
        )

        # Generate viewpoints from which the object model will be rendered.
        trimesh_obj_model = renderer.get_object_model(
            obj_id=object_lid, center_model=True
        )
        obj_radius = trimesh_obj_model.bounding_sphere.primitive.radius * 1000  # to mm
        dist = misc.distance_for_rendering(
            obj_radius=obj_radius,
            camera_model=render_camera_model,
        )

        views_sphere = misc.sample_views(
            min_n_views=opts.min_num_viewpoints,
            radius=dist,
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

        # Prepare a metadata list.
        metadata_list = []

        template_counter = 0

        for view in tqdm(views):
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
                render_types=RENDER_TYPES,
                return_tensors=False,
                debug=False,
            )

            # Convert rendered mask.
            if RenderType.MASK in output:
                output[RenderType.MASK] = (255 * output[RenderType.MASK]).astype(
                    np.uint8
                )

            # Calculate 2D bounding box of the object and make sure
            # it is within the image.
            ys, xs = output[RenderType.MASK].nonzero()
            box = np.array(misc.calc_2d_box(xs, ys))
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

            # Optional cropping of the rendered images.
            if opts.crop:
                # Get box for cropping.
                crop_box = misc.calc_crop_box(
                    box=object_box,
                    box_scaling_factor=1.0 + 2 * opts.crop_rel_pad,
                    make_square=True,
                )

                # Crop and resize the color image.
                output["cropped_img"] = misc.crop_and_resize_image(
                    image=output[RenderType.COLOR],
                    crop_box=crop_box,
                    output_size=opts.crop_size,
                    interpolation=cv2.INTER_LINEAR,
                )
                rgb_image = np.asarray(255.0 * output["cropped_img"], np.uint8)

                # Crop and resize the binary mask.
                mask_image = misc.crop_and_resize_image(
                    image=output[RenderType.MASK],
                    crop_box=crop_box,
                    output_size=opts.crop_size,
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                rgb_image = np.asarray(255.0 * output[RenderType.COLOR], np.uint8)
                mask_image = output[RenderType.MASK]

            # Save template rgb and mask.
            rgb_path = os.path.join(
                templates_rgb_dir, f"template_{template_counter:04d}.png"
            )
            inout.save_im(rgb_path, rgb_image)

            # Save template mask.
            mask_path = os.path.join(
                templates_mask_dir, f"template_{template_counter:04d}.png"
            )
            inout.save_im(mask_path, mask_image)

            data = {
                "dataset": object_dataset.dataset_name,
                "lid": object_lid,
                "template_id": template_counter,
                "rgb_image_path": rgb_path,
                "binary_mask_path": mask_path,
            }

            metadata_list.append(data)

            template_counter += 1

        # Save the metadata to be read from object repre.
        metadata_path = os.path.join(output_dir, "metadata.json")
        json_util.save_json(metadata_path, metadata_list)

    return True


def main() -> None:
    # Load options
    opts = config_util.load_opts_from_json_or_command_line(GenTemplatesOpts)[0]

    # Synthesize templates for the objects in the dataset
    synthesize_templates(opts)


if __name__ == "__main__":
    main()

