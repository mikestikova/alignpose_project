# AlignPose
Featuremetric Multi-View Refinement for BOP Datasets

This repository provides a pipeline to run multi-view featuremetric refinement on BOP (Benchmark for 6D Object Pose Estimation) datasets. It is designed for 6DoF object pose estimation from multiple views.

## Table of Contents

- [Setup](#setup)
   - [Environment](#environment)
   - [Dataset](#dataset)
- [Using Alignpose](#alignpose)
- [Using Foundpose](#foundpose)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Setup <a name="setup"></a>

### Environment <a name="environment"></a>

Download the code with the git submodules and navigate to the folder:

```bash
git clone --recurse-submodules https://github.com/mikestikova/alignpose.git
cd alignpose
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

export REPO_PATH=/path/to/foundpose/repository  # Replace with the path to the FoundPose repository.
export BOP_PATH=/path/to/bop/datasets  # Replace with the path to BOP datasets (https://bop.felk.cvut.cz/datasets).

export PYTHONPATH=$REPO_PATH:$REPO_PATH/external/bop_toolkit:$REPO_PATH/external/dinov2:$REPO_PATH/external/pixloc
```

Activate the conda environment:
```bash
conda activate foundpose_gpu
```

## Running Alignpose<a name="alignpose"></a>
Currently we support BOP datasets from BOP-Industrial track (ITODD-MV, XYZ-IBD, IPD) and selected datasets from BOP-Classic (YCBV, T-LESS).

### Dataset <a name="dataset"></a>
Download the BOP datasets from [here](https://bop.felk.cvut.cz/datasets/). 
Note that we only need the `base` archive with the dataset info, `models` folder and `test` folder. Extract dataset to this [format.](https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md)

In addition to the test images, this method needs a specification of which views shoud be used together in the multi-view setup:
- BOP Industrial (ipd, xyzibd, itoddmv) contain `test_targets_multiview_bop25.json` folder.
- BOP Classic Core (YCBV, TLESS) do not contain this folder so we provide it in `bop_test_targets/{dataset}/test_targets_multiview_bop25.json`. These files were generated for 4-view setup.  

For custom dataset follow the setup in 

### Generate templates:
Specify your configs in `configs/gen_templates/ycbv.json`
```bash
python src/scripts/run_bop_gen_templates.py --opts-path configs/gen_templates/ycbv.json
```

### Generate obejct representation
`configs/gen_repre/ycbv.json`
```bash
# Generates PCA object representation
python src/scripts/run_bop_gen_repre.py --opts-path configs/gen_repre/ycbv.json
```

### Prepare input pose estimates
We provide some sample inputs in the `data/inputs` folder. Alternatively, you may download input pose estimate `.csv` files from [BOP Leaderboard](https://bop.felk.cvut.cz/leaderboards/pose-estimation-unseen-bop23/ycb-v/) or generate them with any single-view pose estimation method.

Format of input:
- BOP Classic Core (YCB-V, T-LESS): One `.csv` file with predictions in BOP format.
- BOP Industrial (IPD, XYZIBD, ITODDMV): Folder containing a `.csv` files with predictions for all of the cameras named `{camera_id}-xxxx_{dataset-name}-test.csv`.


### Run multi-view refinement 
Run multi-view refinement using the following script and configuration:
`configs/refine/ycbv.json`
```bash
python src/scripts/run_refine_multi_view.py --opts-path src/configs/refine/ycbv.json
```


### BOP Evaluation <a name="bop-evaluation"></a>
Evaluate the BOP submission file by running this script. First specify the path to the csv and the eval_dir within the script:
```
python scripts/results_bop_evaluate.py --csv_path  path/to/csv/results.csv
```

## Runnign Alingpose Custom Datasets: 
# Create config files as needed:
configs/gen_templates/dataset_name.json
configs/gen_repre/dataset_name.json
```bash
# Generate templates
python src/scripts/gen_templates.py --opts-path configs/gen_templates/dataset_name.json 
# Create object representation for all objects
python src/scripts/gen_repre.py --opts-path configs/gen_repre/dataset_name.json
```

### Inference
configs/refine/dataset_name.json
python refine_multi_view.py


## Citation
If you find this code useful in your research, please cite:

```
@misc{mikestikova2025alignpose,
      title={AlignPose: Generalizable 6D Pose Estimation via Multi-view Feature-metric Alignment}, 
      author={Anna Šárová Mikeštíková and Médéric Fourmy and Martin Cífka and Josef Sivic and Vladimir Petrik},
      year={2025},
      eprint={2512.20538},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.20538}, 
}
```

## Acknowledgements <a name="acknowledgements"></a>

This repository relies heavily on the following works:
- [BOP Toolkit](https://github.com/thodan/bop_toolkit) -  Evaluation scripts, standard utils
- [FoundPose](https://github.com/facebookresearch/foundpose) - Single-view pose estimation, utils
- [Pixloc](https://github.com/cvg/pixloc) - Featuremetric refinement
