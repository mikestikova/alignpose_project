import argparse
import os
from pathlib import Path
import bop_toolkit_lib
import subprocess


def main(args):
    csv_path = args.csv_path

    if not os.path.isfile(csv_path):
        print(f"Error: File {csv_path} does not exist.")
        return
    
    eval_dir = Path(csv_path).parent

    bop_path = os.path.dirname(bop_toolkit_lib.__file__).split("/bop_toolkit_lib")[0]
    script_ar = os.path.join(bop_path, "scripts", "eval_bop19_pose.py")
    command = [
        "python", 
        script_ar, 
        "--renderer_type=vispy", 
        f"--result_filenames={csv_path}",
        f"--results_path={eval_dir}",
        f"--eval_path={eval_dir}",
        f"--targets_filename=test_targets_bop19.json",
        f"--num_workers=1"
    ]

    subprocess.run(command)

    script_ap = os.path.join(bop_path, "scripts", "eval_bop24_pose.py")
    command = [
        "python", 
        script_ap, 
        "--renderer_type=vispy", 
        f"--result_filenames={csv_path}",
        f"--results_path={eval_dir}",
        f"--eval_path={eval_dir}",
        f"--targets_filename=test_targets_bop19.json",
        f"--num_workers=1"
    ]

    subprocess.run(command)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Evaluate BOP results.")
    arg_parser.add_argument("--csv_path", type=str, required=True, help="Path to the predictions csv file.")
    args = arg_parser.parse_args()
    main(args)