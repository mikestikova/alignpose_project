
<p align="center">
  <h1 align="center">AlignPose: Generalizable 6D Pose Estimation via <br> Multi-view Feature-metric Alignment</h1>
  <p align="center">
    <a href="">Anna Šárová Mikeštíková</a>
    ·
    <a href="https://medericfourmy.github.io">Médéric Fourmy</a>
    ·
    <a href="https://cifkam.github.io">Martin Cífka</a>
    ·
    <a href="http://people.ciirc.cvut.cz/~sivic/">Josef Sivic</a>
    .
    <a href="https://petrikvladimir.github.io">Vladimir Petrik</a>
  </p>
  <h3 align="center">
    <a href="https://arxiv.org/abs/2512.20538" align="center">Arxiv</a>
    ·
    <a href="https://mikestikova.github.io/alignpose/" align="center">Project Page</a>
  </h3>
</p>

<p align="center">
  <img src="images/alignpose_pipeline.png" width=80% alt="AlignPose pipeline overview" max-width=20px>
</p>

This repository contains code for AlignPose, our method for 6 object pose estimation from multiple views. This branch is for running AlignPose on your own objects, cameras and images. To reproduce the results on [BOP](https://bop.felk.cvut.cz/) datasets from the paper, see the `master` branch.

## Table of Contents

- [Environment setup](#setup)
- [Running Alignpose for custom objects](#alignpose_custom)
   - [Full pose estimation](#custom_full)
   - [Multi-view refinement only](#custom_refine)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Environment setup <a name="setup"></a>

Download the code with the git submodules and navigate to the folder:

```bash
git clone --recurse-submodules https://github.com/mikestikova/alignpose_project.git
cd alignpose_project
```

Setup the [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) environment for CUDA:
```bash
conda env create -f conda_alignpose_gpu.yaml
```

Next, create (or update) the conda environment activation script to set the necessary environment variables. This script will run automatically when you activate the environment.

The activation script is typically located at ```$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh```. You can find ```$CONDA_PREFIX``` by running:
```bash
conda info --envs
```
If the ```env_vars.sh``` file does not exist, create it. 

Edit the ```env_vars.sh``` file as follows:

```bash
#!/bin/sh

export ALIGNPOSE=/path/to/alignpose/repository  # Replace with the path to the AlignPose repository.
export BOP_PATH=/path/to/bop/datasets  # Replace with the path to BOP datasets (https://bop.felk.cvut.cz/datasets).

export PYTHONPATH=$ALIGNPOSE:$ALIGNPOSE/src/external/bop_toolkit:$ALIGNPOSE/src/external/dinov2:$ALIGNPOSE/src/external/pixloc
```

Activate the conda environment:
```bash
conda activate alignpose_gpu
```

## Running Alignpose for custom objects <a name="alignpose_custom"></a>
This section describes how to run AlignPose on your own objects, cameras and images, without depending on BOP datasets. There are two ways to use it:
- [Full pose estimation](#custom_full): you have images, calibrated cameras and detection masks of the objects. FoundPose estimates a pose in each view, and the multi-view refinement combines the views.
- [Multi-view refinement only](#custom_refine): you already have initial poses (e.g. from another single-view method) and want to refine them using all views.

### Full pose estimation (FoundPose + multi-view refinement) <a name="custom_full"></a>

#### 1. Build the object representations
Each object needs preprocessing: we render templates of the object from multiple viewpoints, then extract features from them to build the object representation. Render the templates with your camera intrinsics at your working distance from the object:
```bash
python src/scripts/preprocess_objects_foundpose.py \
    --model-paths path/to/models/obj_000001.ply path/to/models/obj_000002.ply \
    --object-ids 1 2 \
    --camera path/to/camera.json \
    --distance 600 \
    --output-dir object_repre/custom_dataset/foundpose_v1
```
- `--camera` is a `.json` or `.yml` file with `fx`, `fy`, `cx`, `cy`, `width` and `height` (or a BOP-style `cam_K`).
- `--distance` is the camera-object distance in mm.
- `--ssaa-factor` (default 4) renders the templates at a higher resolution for anti-aliasing. For large camera images (e.g. 5 MP), use `--ssaa-factor 1` to keep the rendering size manageable.
- `--save-templates` also saves the rendered templates to `<output-dir>/<object-id>/templates/` (`rgb/`, `depth/` in mm, `mask/` and `cameras.json`), e.g. to check what the templates look like. The template numbers match the template ids in the representation.

Each representation is saved to `<output-dir>/<object-id>/repre.pth`, with the options used in `config.json`. With the default settings (798 templates), building one takes about 15–30 minutes on a GPU and the file is about 0.5 GB.

To debug the template generation, you can run the two steps separately from Python and inspect the templates before extracting the features:
```python
from src.utils_multiview.processing.foundpose_repre import (
    build_foundpose_repre, foundpose_repre_config, load_intrinsics,
    render_foundpose_templates, save_foundpose_repre, save_templates,
)

model_path = "path/to/my_object.ply"
K, image_size = load_intrinsics("path/to/camera.json")
depth_range = (600, 600)  # camera-object distance in mm

templates = render_foundpose_templates(model_path, K, image_size, depth_range)
save_templates(templates, "debug/templates")  # optional, for debugging
repre = build_foundpose_repre(templates)

config = foundpose_repre_config(model_path, K, image_size, depth_range)
save_foundpose_repre(repre, "object_repre/custom_dataset/foundpose_v1", object_id=1, config=config)
```

#### 2. Run pose estimation
Coming soon!

### Multi-view refinement only <a name="custom_refine"></a>

#### 1. Preprocess the objects
The multi-view refinement needs only the PCA of each object's features, which is faster to compute than the full representation. (A full FoundPose representation from the section above works as well.) Object IDs must be unique.
```bash
python src/scripts/preprocess_objects.py \
    --model-paths path/to/models/obj_000001.ply path/to/models/obj_000002.ply \
    --object-ids 1 2 \
    --output-dir object_repre/custom_dataset/repre_v1
```
Each representation is saved to `<output-dir>/<object-id>/repre.pth`. To use the representations in `refine_multiview.py`, save them to `object_repre/<object_dataset>/<repre_version>` and set `repre_version` in the refinement config to match.

You can do the same from Python:
```python
from src.utils_multiview.processing.object_pca import fit_object_pca, save_object_pca

repre, config = fit_object_pca("path/to/my_object.ply")
save_object_pca(repre, "object_repre/custom_dataset/repre_v1", object_id=1, config=config)
```

#### 2. Run multi-view refinement
Coming soon!


## Citation
If you find this code useful in your research, please cite:

```
@InProceedings{mikestikova2026alignpose,
    author    = {Mike\v{s}t{\'\i}kov\'a, Anna \v{S}\'arov\'a and Fourmy, M\'ed\'eric and Cifka, Martin and Sivic, Josef and Petrik, Vladimir},
    title     = {AlignPose: Generalizable 6D Pose Estimation via Multi-view Feature-metric Alignment},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026}
}
```

## Acknowledgements <a name="acknowledgements"></a>

This repository relies heavily on the following works:
- [BOP Toolkit](https://github.com/thodan/bop_toolkit) -  Evaluation scripts, standard utils
- [FoundPose](https://github.com/facebookresearch/foundpose) - Single-view pose estimation, utils
- [Pixloc](https://github.com/cvg/pixloc) - Featuremetric refinement

## License <a name="license"></a>

The code in this repository is released under the [MIT License](LICENSE), with the following exceptions:
- `foundpose/` and the files adapted from [FoundPose](https://github.com/facebookresearch/foundpose) (`src/scripts/gen_repre.py`, `src/scripts/gen_templates.py`, `src/utils_multiview/opts.py`) are licensed under [CC BY-NC 4.0](foundpose/LICENSE).
- The submodules in `src/external/` ([BOP Toolkit](https://github.com/thodan/bop_toolkit), [DINOv2](https://github.com/facebookresearch/dinov2), [Pixloc](https://github.com/cvg/pixloc)) are distributed under their own licenses.

Note that the full pipeline depends on the FoundPose code, so running it is subject to the non-commercial terms of CC BY-NC 4.0.
