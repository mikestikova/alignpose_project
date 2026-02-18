import os

import bop_toolkit_lib
import subprocess

os.environ["PYOPENGL_PLATFORM"] = "egl"

csv_paths = ["dinov3_vitb11_pca256_no_norm_ycbv-test.csv"]
eval_dir = "/home/mikesann/data/inference/layer_experiment/ycbv/dinov3/"
os.makedirs(eval_dir, exist_ok=True)

for csv_path in csv_paths:
    bop_path = os.path.dirname(bop_toolkit_lib.__file__).split("/bop_toolkit_lib")[0]
    script_path = os.path.join(bop_path, "scripts", "eval_bop19_pose.py")
    command = [
        "python",
        script_path,
        "--renderer_type=vispy",
        f"--result_filenames={csv_path}",
        f"--results_path={eval_dir}",
        f"--eval_path={eval_dir}",
        f"--targets_filename=test_targets_bop19.json",
        f"--num_workers=1",
    ]

    subprocess.run(command)

    bop_path = os.path.dirname(bop_toolkit_lib.__file__).split("/bop_toolkit_lib")[0]
    script_path = os.path.join(bop_path, "scripts", "eval_bop24_pose.py")
    command = [
        "python",
        script_path,
        "--renderer_type=vispy",
        f"--result_filenames={csv_path}",
        f"--results_path={eval_dir}",
        f"--eval_path={eval_dir}",
        f"--targets_filename=test_targets_bop19.json",
        f"--num_workers=1",
    ]

    subprocess.run(command)
