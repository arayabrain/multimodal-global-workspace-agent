import os
import cv2
import uuid
import datetime
import numpy as np
import compress_pickle as cpkl

from ss_baselines.av_nav.config import get_config
from ss_baselines.savi.config.default import get_config as get_savi_config
from ss_baselines.common.env_utils import construct_envs
from ss_baselines.common.environments import get_env_class
from ss_baselines.common.utils import plot_top_down_map

# Helper / tools
from soundspaces.mp3d_utils import CATEGORY_INDEX_MAPPING
def get_category_name(idx):
    assert idx >= 0 and idx <=20, f"Invalid category index number: {idx}"

    for k, v in CATEGORY_INDEX_MAPPING.items():
        if v == idx:
            return k

def get_current_ep_category_label(obs_dict):
    return get_category_name(obs_dict["category"].argmax())

DATASET_DIR_PATH = f"SAVI_Oracle_Dataset_v0"
# DATASET_DIR_PATH = f"SAVI_Oracle_Dataset_v0_10K" # Smaller scale dataset for tests

# Read the dataset statistics file.
dataset_stats_filepath = f"{DATASET_DIR_PATH}/dataset_statistics.bz2"
with open(dataset_stats_filepath, "rb") as f:
    r__dataset_stats = cpkl.load(f)

from pprint import pprint
pprint(r__dataset_stats)


# Start byreading all the episodes in 
M = 5 # number fo scenes / rooms, for one category
N = 5 # number of trajs. per scenes / rooms, for one category
CATEGORIES_OF_INTEREST = [
    "chair",
    "picture",
    "table",
    "cushion",
    "cabinet",
    "plant"
]
C = len(CATEGORIES_OF_INTEREST)

trajs_scenes_cat = {
    k: {} for k in CATEGORIES_OF_INTEREST
}

n_selected_trajs = 0
n_selected_trajs_cat_counts = {
    k: 0 for k in CATEGORIES_OF_INTEREST
}

ep_filenames = os.listdir(DATASET_DIR_PATH)
if "dataset_statistics.bz2" in ep_filenames:
    ep_filenames.remove("dataset_statistics.bz2")
ep_filenames_iterator = iter(ep_filenames)

scenes_of_interest = [] # To make sure we have the same scenes for each category

while n_selected_trajs < C * N * M:
    ep_filename = next(ep_filenames_iterator)

    ep_filepath = f"{DATASET_DIR_PATH}/{ep_filename}"
    with open(ep_filepath, "rb") as f:
        edd = cpkl.load(f)

    ep_length = edd["ep_length"]
    ep_category = edd["category_name"]
    ep_scene = edd["scene_id"]

    # Skip if the category does not match
    if ep_category not in CATEGORIES_OF_INTEREST:
        continue

    # Track which scenes' trajectories will be saved.
    # We want the same scenes for each category
    if len(scenes_of_interest) < M and (ep_scene not in scenes_of_interest):
        scenes_of_interest.append(ep_scene)
    
    if ep_scene not in scenes_of_interest:
        continue

    if ep_scene not in trajs_scenes_cat[ep_category].keys():
        # First time seeing the scene: add it to the dict, along with the new traj.
        if len(trajs_scenes_cat[ep_category]) < M:
            # Only add it if we don't have enough scenes yet.
            trajs_scenes_cat[ep_category][ep_scene] = [
                {
                    "ep_filename": ep_filename,
                    "edd": edd
                }
            ]
            n_selected_trajs += 1
            n_selected_trajs_cat_counts[ep_category] += 1
    else:
        # The scene was already seen once; check if we need more, and append accordingly
        if len(trajs_scenes_cat[ep_category][ep_scene]) < N:
            trajs_scenes_cat[ep_category][ep_scene].append({
                "ep_filename": ep_filename,
                "edd": edd
            })
            n_selected_trajs += 1
            n_selected_trajs_cat_counts[ep_category] += 1
    
    print("### --------------------------------------------------- ###")
    print(f"### # selected traj: {n_selected_trajs_cat_counts[ep_category]} for \"{ep_category}\"")
    for k, v in trajs_scenes_cat[ep_category].items():
        print(f"\t{k}: {len(v)}")
    print("### --------------------------------------------------- ###")
    print("")

# Saving the filtered trajectories data
# trajs_scenes_cat["chair"] # Check the content
C = len(CATEGORIES_OF_INTEREST)
analysis_trajs_filename = f"analysis_trajs_C_{C}_M_{M}_N_{N}.bz2"; print(analysis_trajs_filename)
# Uncomment for actual saving
with open(analysis_trajs_filename, "wb") as f:
    cpkl.dump(trajs_scenes_cat, f)
