from bop_toolkit_lib import config as bop_config

from pathlib import Path
OUTPUT_DIR = Path(__file__).resolve().parents[2] # TODO: specify output dir
BOP_DATASETS = ["ycbv", "tless", "xyzibd", "itoddmv", "ipd"]
BOP_DATASETS_PATH = bop_config.datasets_path