import os
import cv2
import uuid
import time
import datetime
import numpy as np
import compress_pickle as cpkl

# General config related
from configurator import get_arg_dict, generate_args

from ss_baselines.av_nav.config import get_config
from ss_baselines.savi.config.default import get_config as get_savi_config
from ss_baselines.common.env_utils import construct_envs
from ss_baselines.common.environments import get_env_class

# Helper to show expected training time in human readable form:
# Credits: https://github.com/hevalhazalkurt/codewars_python_solutions/blob/master/4kyuKatas/Human_readable_duration_format.md
def hrd(seconds): # Human readable duration
    words = ["year", "day", "hr", "min", "sec"]
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    y, d = divmod(d, 365)

    time = [y, d, h, m, s]
    duration = []

    for x, i in enumerate(time):
        if i == 1:
            duration.append(f"{i} {words[x]}")
        elif i > 1:
            duration.append(f"{i} {words[x]}s")

    if len(duration) == 1:
        return duration[0]
    elif len(duration) == 2:
        return f"{duration[0]}, {duration[1]}"
    else:
        return ", ".join(duration[:-1]) + ", " + duration[-1]
    
# Helper / tools
from soundspaces.mp3d_utils import CATEGORY_INDEX_MAPPING
def get_category_name(idx):
    assert idx >= 0 and idx <=20, f"Invalid category index number: {idx}"

    for k, v in CATEGORY_INDEX_MAPPING.items():
        if v == idx:
            return k

def get_env_scene_id(envs, env_idx):
    # For VECENV
    return envs.call_at(env_idx, "get_scene_id")

    # FOr SyncEnv
    return envs.workers[env_idx]._env._env.current_episode.scene_id.split("/")[3]

def get_current_ep_category_label(obs_dict):
    return get_category_name(obs_dict["category"].argmax())

def save_episode_to_dataset(ep_data_dict, dataset_path):
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    epid = str(uuid.uuid4().hex)
    ep_length = ep_data_dict["ep_length"]

    # TODO: consider the downside of compression if this has to be used for 
    ep_scene_id = ep_data_dict["scene_id"]
    ep_category_name = ep_data_dict["category_name"]

    ep_data_filename = f"{ep_scene_id}-{ep_category_name}-{ep_length}-{timestamp}-{epid}.bz2"
    ep_data_fullpath = os.path.join(dataset_path, ep_data_filename)
    with open(ep_data_fullpath, "wb") as f:
        cpkl.dump(ep_data_dict, f)
    
    return ep_data_filename

def dict_without_keys(d, keys_to_ignore):
    return {x: d[x] for x in d if x not in keys_to_ignore}


def main():

    # region: Generating additional hyparams
    CUSTOM_ARGS = [
        ## General configs
        get_arg_dict("seed", int, 111),
        get_arg_dict("total-steps", int, 500_000),
        get_arg_dict("num-envs", int, 10), # Number of parallel envs. 10 by default

        ## Dataset related ocnfigs
        get_arg_dict("dataset-path", str, None),

    ]
    args = generate_args(CUSTOM_ARGS)
    # endregion: Generating additional hyparams


    # Configure the environments for data collection
    config = get_savi_config(config_paths="env_configs/savi/savi_ss1.yaml")
    config.defrost()
    config.SEED = config.TASK_CONFIG.SEED = config.TASK_CONFIG.SIMULATOR.SEED = args.seed
    config.NUM_PROCESSES = args.num_envs
    config.USE_SYNC_VECENV = False # Maybe not as fast as VecEnv, but can at least access more info in each environments.
    config.USE_VECENV = True # Allegedly best perfs., but did not find access to oracles actions
    
    ## Override semantic object sensor sizes: does RGB / Depth sensor's shape increase ?
    # config.DISPLAY_RESOLUTION = 512
    # config.TASK_CONFIG.TASK.SEMANTIC_OBJECT_SENSOR.HEIGHT = 512
    # config.TASK_CONFIG.TASK.SEMANTIC_OBJECT_SENSOR.WIDTH = 512

    # For custom resolution, disable the use of pre-rendered observations
    config.TASK_CONFIG.SIMULATOR.USE_RENDERED_OBSERVATIONS = False
    # For smoother video, set CONTINUOUS_VIEW_CHANGE to True, and get the additional frames in obs_dict["intermediate"]
    config.TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE = False

    config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = 256
    config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = 256
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH = 256
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT = 256

    # Add support for TOP_DOWN_MAP
    # TODO: this seems to trigger the "DummySimulator" object has no attribute 'pathfinder'
    # error. Possible fix would be not use the pre-rendered observations
    # with the caveat of slower performance overall
    config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.freeze()
    # print(config)

    print(config)

    # Instantiate the environments
    envs = construct_envs(config, get_env_class(config.ENV_NAME))

    # Dataset parameterization
    # TODO: add argparse support ?
    DATASET_TOTAL_STEPS = args.total_steps
    # DATASET_DIR_PATH = f"SAVI_Oracle_Dataset_2023_05_17__{DATASET_TOTAL_STEPS}__STEPS"
    DATASET_DIR_PATH = "SAVI_Oracle_Dataset_v0" # This assumes the directory is already created, on a drive that have enouhg room for all the data.
    
    if args.dataset_path == None:
        raise NotImplementedError(f"TODO: add support for dataset generation wihtout giving the path; use timestamp as suffix ?")
    else:
        DATASET_DIR_PATH = args.dataset_path

    ## Compute action coefficient for CEL of BC
    dataset_stats_filename = "dataset_statistics.bz2"
    dataset_stats_filepath = f"{DATASET_DIR_PATH}/{dataset_stats_filename}"

    NUM_ENVS = config.NUM_PROCESSES

    # Data collection
    # Placeholders for episode data
    obs_list, \
    reward_list, \
    done_list, \
    info_list, \
    action_list, \
    ep_scene_id_list = \
        [[] for _ in range(NUM_ENVS)], \
        [[] for _ in range(NUM_ENVS)], \
        [[] for _ in range(NUM_ENVS)], \
        [[] for _ in range(NUM_ENVS)], \
        [[] for _ in range(NUM_ENVS)], \
        [[] for _ in range(NUM_ENVS)]
    # Plaecholders for various statistics about the training data
    dataset_statistics = {
        "action_counts": {i: 0 for i in range(4)},
        "total_steps": 0,
        "total_episodes": 0,
        "scene_counts": {},
        "category_counts": {get_category_name(i): 0 for i in range(21)}, # 21 categories in SAVi.
        "episode_lengths": [],
        "category_counts": {get_category_name(i): 0 for i in range(21)}, # 21 categories in SAVi.
        "cat_scene_filenames": {}, # category -> { scene_id -> [{ep_filename: "", "ep_length": X}]} for easier filtering later
        "scene_cat_filenames": {}, # scene_id -> {categary -> [{ep_filename: "", "ep_length": X}]} for easier filtering later
    }
    # Override dataset statistics if the file already exists
    if os.path.exists(dataset_stats_filepath):
        with open(dataset_stats_filepath, "rb") as f:
            dataset_statistics = cpkl.load(f)

    obs, done = envs.reset(), [False for _ in range(NUM_ENVS)]

    # Track data collection time
    start_time = time.time()
    step = 0
    ep_returns = []
    envs_current_step = [0 for _ in range(NUM_ENVS)]

    while step < DATASET_TOTAL_STEPS:
        # Recover the optimal action for each parallel env, for SYNC_ENV
        # actions = [envs.workers[i]._env._env._sim._oracle_actions[envs_current_step[i]] for i in range(NUM_ENVS)]

        # For VecEnv support:
        actions = [envs.call_at(i, "get_oracle_action_at_step", {"step": envs_current_step[i]}) for i in range(NUM_ENVS)]

        # Get current step's meta data
        envs_scene_ids = [get_env_scene_id(envs, i) for i in range(NUM_ENVS)]

        # Step the environment
        outputs = envs.step(actions)
        next_obs, reward, next_done, info = [list(x) for x in zip(*outputs)]

        # Recorder episode trajectoreis
        for i in range(NUM_ENVS):
            obs_list[i].append(obs[i])
            done_list[i].append(done[i])
            action_list[i].append(actions[i])
            reward_list[i].append(reward[i])
            info_list[i].append(info[i])
            ep_scene_id_list[i].append(envs_scene_ids[i])

        # When one or more episode end is detected, write to disk,
        # then reset the placeholders for the finished env. idx
        if np.sum(next_done) >= 1.:
            finished_envs_idxs = np.where(next_done)[0]
        
            for i in finished_envs_idxs:
                if not info_list[i][-1]["success"] == 1:
                    continue
            
                ep_length = len(obs_list[i])
                ep_returns = []
                ep_success = []
                ep_norm_dist_to_goal = []

                # Pre-process the obs_dict to have lists of "rgb", "depth", etc..
                obs_dict = {k: [] for k in obs_list[i][0].keys()} # TODO: to also store intermediate, maaybe use obs_list[i][1].keys() instead

                # Stores the high resolution rgb and depth data
                highres_obs_list = {k: [] for k in ["rgb", "depth"]}

                for t in range(ep_length):
                    for k, v in obs_list[i][t].items():
                        if k in ["rgb", "depth"]:
                            highres_obs_list[k].append(v.copy()) # TODO: make sure the high res stuff is not resized later
                            obs_dict[k].append(cv2.resize(v, dsize=(128, 128)))
                        else:
                            obs_dict[k].append(v)
                    
                    # Count actions for overall dataset statistics
                    dataset_statistics["action_counts"][action_list[i][t]] += 1
                
                # Additional episode metadata
                # TODO: episode's scene SPLIT
                ep_scene_id = get_env_scene_id(envs, i)
                ep_category_idx = obs_dict["category"][0].argmax()
                ep_category_name = get_category_name(ep_category_idx)

                ep_data_dict = {
                    "obs_list": obs_dict, # RGB and DEPTH shape (128, 128, 3)
                    "highres_obs_list": highres_obs_list, # RGB and DEPTH shape (512, 512, 3)
                    "action_list": action_list[i],
                    "done_list": done_list[i],
                    "reward_list": reward_list[i],
                    "info_list": info_list[i], # This can arguably be skipped ?,
                    "ep_scene_id_list": ep_scene_id_list[i],
                    # Other metadata
                    "ep_length": ep_length,
                    "scene_id": ep_scene_id,
                    "category_idx": ep_category_idx,
                    "category_name": ep_category_name
                }

                ep_returns.append(np.sum(reward_list[i]))
                # TODO: double check why the last info list is not the final one
                ep_success.append(info_list[i][-1]["success"])
                ep_norm_dist_to_goal.append(info_list[i][-1]["normalized_distance_to_goal"])

                # Saves to disk
                ep_filename = save_episode_to_dataset(ep_data_dict, DATASET_DIR_PATH)

                step += ep_length

                # Track overall statistics of the dataset
                dataset_statistics["total_episodes"] += 1
                dataset_statistics["total_steps"] += ep_length
                dataset_statistics["category_counts"][ep_category_name] += 1
                if ep_scene_id not in dataset_statistics["scene_counts"].keys():
                    dataset_statistics["scene_counts"][ep_scene_id] = 1
                else:
                    dataset_statistics["scene_counts"][ep_scene_id] += 1
                # Track the lengths of all episodes
                dataset_statistics["episode_lengths"].append(ep_length)

                # Add metadata about episode grouped by categories, then scenes
                if ep_category_name not in dataset_statistics["cat_scene_filenames"].keys():
                    dataset_statistics["cat_scene_filenames"][ep_category_name] = {}
                if ep_scene_id not in dataset_statistics["cat_scene_filenames"][ep_category_name].keys():
                    dataset_statistics["cat_scene_filenames"][ep_category_name][ep_scene_id] = []
                dataset_statistics["cat_scene_filenames"][ep_category_name][ep_scene_id].append(
                    {"ep_filename": ep_filename, "ep_length": ep_length})

                # Add metadata about episodes grouped by scene ids, then categories
                if ep_scene_id not in dataset_statistics["scene_cat_filenames"].keys():
                    dataset_statistics["scene_cat_filenames"][ep_scene_id] = {}
                if ep_category_name not in dataset_statistics["scene_cat_filenames"][ep_scene_id].keys():
                    dataset_statistics["scene_cat_filenames"][ep_scene_id][ep_category_name] = []
                dataset_statistics["scene_cat_filenames"][ep_scene_id][ep_category_name].append(
                    {"ep_filename": ep_filename, "ep_length": ep_length})

                # Reset the data placeholders
                obs_list[i], action_list[i], done_list[i], reward_list[i], info_list[i], ep_scene_id_list[i] = \
                    [], [], [], [], [], []

                # Save the dataset statistics to file
                ## Compute action probs
                dataset_statistics["action_probs"] = {
                    a: dataset_statistics["action_counts"][a] / dataset_statistics["total_steps"] for a in range(4)
                }
                dataset_statistics["action_cel_coefs"] = {
                    k: (0.25 / v) if v > 0 else 0. for k, v in dataset_statistics["action_probs"].items()
                }
                ## Compute action coefficient for CEL of BC
                dataset_stats_filename = "dataset_statistics.bz2"
                dataset_stats_filepath = f"{DATASET_DIR_PATH}/{dataset_stats_filename}"
                with open(dataset_stats_filepath, "wb") as f:
                    cpkl.dump(dataset_statistics, f)
                
                for k, v in dict_without_keys(dataset_statistics,
                    ["cat_scene_filenames", "scene_cat_filenames", "episode_lengths"]).items():
                    print(f"{k}: {v}")

            SPS = step / (time.time() - start_time) # Number of steps per second
            print("")
            print("#####################################################################################")
            print("#####################################################################################")
            print(f"Collected {step} / {DATASET_TOTAL_STEPS}; Avg return: {np.mean(ep_returns):0.2f}; Avg Suc.: {np.mean(ep_success)}; Avg: Norm Dist Goal: {np.mean(ep_norm_dist_to_goal)}")
            print("ETA:", hrd(int(args.total_steps - step) / SPS))
            print("#####################################################################################")
            print("#####################################################################################")
            print("")
                
        # Prepare for the next step
        obs = next_obs
        done = next_done

        for i in range(NUM_ENVS):
            envs_current_step[i] = int((1 - next_done[i]) * (envs_current_step[i] + 1))

        # Stop collection ASAP
        if step >= DATASET_TOTAL_STEPS:
            break

if __name__ == "__main__":
    main()
