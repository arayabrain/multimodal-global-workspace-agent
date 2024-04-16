import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import os
import apex
import copy
import random
import numpy as np
import matplotlib as mpl
import compress_pickle as cpkl

import rsatoolbox

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec # TODO: move to the top

from mpl_toolkits.axes_grid1 import make_axes_locatable

import umap

from ss_baselines.common.utils import plot_top_down_map

mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["axes.facecolor"] = "white"
mpl.rcParams["savefig.facecolor"] = "white" 

# General config related
from configurator import get_arg_dict, generate_args

# Env config related
from ss_baselines.av_nav.config import get_config
from ss_baselines.savi.config.default import get_config as get_savi_config
from ss_baselines.common.env_utils import construct_envs
from ss_baselines.common.environments import get_env_class

# region: Generating additional hyparams
CUSTOM_ARGS = [
    # General hyper parameters
    get_arg_dict("seed", int, 111),
    get_arg_dict("total-steps", int, 1_000_000),
    
    # Behavior cloning gexperiment config
    get_arg_dict("dataset-path", str, "SAVI_Oracle_Dataset_v0"),

    # SS env config
    get_arg_dict("config-path", str, "env_configs/savi/savi_ss1.yaml"),

    # Probing setting
    get_arg_dict("probe-depth", int, 1),
    get_arg_dict("probe-hid-size", int, 512),
    get_arg_dict("probe-bias", bool, False, metatype="bool"),
    
    get_arg_dict("probing-targets", str, ["category", "scene"], metatype="list"), # What to probe for 
    get_arg_dict("probing-inputs", str, 
        ["state_encoder", "visual_encoder.cnn.7", "audio_encoder.cnn.7"], metatype="list"), # What to base the probe on
    get_arg_dict("pretrained-model-name", str, None), # Simplified model name; required
    get_arg_dict("pretrained-model-path", str, None), # Path to the weights of the pre-trained model; required
    get_arg_dict("n-epochs", int, 1), # How many iteration over the whole dataset (* with caveat)
    
    # PPO Hyper parameters
    get_arg_dict("num-envs", int, 1), # Number of parallel envs. 10 by default
    get_arg_dict("num-steps", int, 150), # For each env, how many steps are collected to form PPO Agent rollout.
    get_arg_dict("num-minibatches", int, 1), # Number of mini-batches the rollout data is split into to make the updates
    get_arg_dict("update-epochs", int, 4), # Number of gradient step for the policy and value networks
    get_arg_dict("gamma", float, 0.99),
    get_arg_dict("gae-lambda", float, 0.95),
    get_arg_dict("norm-adv", bool, True, metatype="bool"),
    get_arg_dict("clip-coef", float, 0.1), # Surrogate loss clipping coefficient
    get_arg_dict("clip-vloss", bool, True, metatype="bool"),
    get_arg_dict("ent-coef", float, 0.0), # Entropy loss coef; 0.2 in SS baselines
    get_arg_dict("vf-coef", float, 0.5), # Value loss coefficient
    get_arg_dict("max-grad-norm", float, 0.5),
    get_arg_dict("target-kl", float, None),
    get_arg_dict("lr", float, 2.5e-4), # Learning rate
    get_arg_dict("optim-wd", float, 0), # weight decay for adam optim
    ## Agent network params
    get_arg_dict("agent-type", str, "ss-default", metatype="choice",
        choices=["ss-default", "custom-gru", "custom-gwt", "perceiver-gwt-gwwm"]),
    get_arg_dict("use-pose", bool, False, metatype="bool"), # Use "pose" field iin observations
    get_arg_dict("hidden-size", int, 512), # Size of the visual / audio features and RNN hidden states 
    ## Perceiver / PerceiverIO params: TODO: num_latnets, latent_dim, etc...
    get_arg_dict("pgwt-latent-type", str, "randn", metatype="choice",
        choices=["randn", "zeros"]), # Depth of the Perceiver
    get_arg_dict("pgwt-latent-learned", bool, True, metatype="bool"),
    get_arg_dict("pgwt-depth", int, 1), # Depth of the Perceiver
    get_arg_dict("pgwt-num-latents", int, 8),
    get_arg_dict("pgwt-latent-dim", int, 64),
    get_arg_dict("pgwt-cross-heads", int, 1),
    get_arg_dict("pgwt-latent-heads", int, 4),
    get_arg_dict("pgwt-cross-dim-head", int, 64),
    get_arg_dict("pgwt-latent-dim-head", int, 64),
    get_arg_dict("pgwt-weight-tie-layers", bool, False, metatype="bool"),
    get_arg_dict("pgwt-ff", bool, False, metatype="bool"),
    get_arg_dict("pgwt-num-freq-bands", int, 6),
    get_arg_dict("pgwt-max-freq", int, 10.),
    get_arg_dict("pgwt-use-sa", bool, False, metatype="bool"),
    ## Peceiver Modality Embedding related
    get_arg_dict("pgwt-mod-embed", int, 0), # Learnable modality embeddings
    ## Additional modalities
    get_arg_dict("pgwt-ca-prev-latents", bool, True, metatype="bool"), # if True, passes the prev latent to CA as KV input data

    ## Special BC
    get_arg_dict("prev-actions", bool, False, metatype="bool"),
    get_arg_dict("burn-in", int, 0), # Steps used to init the latent state for RNN component
    get_arg_dict("batch-chunk-length", int, 0), # For gradient accumulation
    get_arg_dict("dataset-ce-weights", bool, True, metatype="bool"), # If True, will read CEL weights based on action dist. from the 'dataset_statistics.bz2' file.
    get_arg_dict("ce-weights", float, None, metatype="list"), # Weights for the Cross Entropy loss
    
    ## Custom GWT Agent with BU and TD attentions
    get_arg_dict("gwt-hid-size", int, 512),
    get_arg_dict("gwt-channels", int, 32),

    ## SSL Support
    get_arg_dict("obs-center", bool, False, metatype="bool"), # Centers the rgb_observations' range to [-0.5,0.5]
    get_arg_dict("ssl-tasks", str, None, metatype="list"), # Expects something like ["rec-rgb-vis", "rec-depth", "rec-spectr"]
    get_arg_dict("ssl-task-coefs", float, None, metatype="list"), # For each ssl-task, specifies the loss coeff. during computation

    # Eval protocol
    get_arg_dict("eval", bool, True, metatype="bool"),
    get_arg_dict("eval-every", int, int(1.5e4)), # Every X frames || steps sampled
    get_arg_dict("eval-n-episodes", int, 5),

    # Logging params
    # NOTE: While supported, video logging is expensive because the RGB generation in the
    # envs hogs a lot of GPU, especially with multiple envs 
    get_arg_dict("save-videos", bool, False, metatype="bool"),
    get_arg_dict("save-model", bool, True, metatype="bool"),
    get_arg_dict("log-sampling-stats-every", int, int(1.5e3)), # Every X frames || steps sampled
    get_arg_dict("log-training-stats-every", int, int(10)), # Every X model update
    get_arg_dict("logdir-prefix", str, "./logs/") # Overrides the default one
]
args = generate_args(CUSTOM_ARGS)

# Additional PPO overrides
args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.num_minibatches)

# Load environment config
is_SAVi = str.__contains__(args.config_path, "savi")
if is_SAVi:
    env_config = get_savi_config(config_paths=args.config_path)
else:
    env_config = get_config(config_paths=args.config_path)
# endregion: Generating additional hyparams

# Overriding some envs parametes from the .yaml env config
env_config.defrost()
env_config.NUM_PROCESSES = 1 # Corresponds to number of envs, makes script startup faster for debugs
env_config.USE_SYNC_VECENV = True
# env_config.USE_VECENV = False
# env_config.CONTINUOUS = args.env_continuous
## In caes video saving is enabled, make sure there is also the rgb videos
env_config.freeze()
# print(env_config)

# Environment instantiation
# envs = construct_envs(env_config, get_env_class(env_config.ENV_NAME))
# Dummy environment spaces

# TODO: add dyanmicallly set single_observation_space so that RGB and RGBD based variants
# can be evaluated at thet same time
from gym import spaces
single_action_space = spaces.Discrete(4)
single_observation_space = spaces.Dict({
    "rgb": spaces.Box(shape=[128,128,3], low=0, high=255, dtype=np.uint8),
    # "depth": spaces.Box(shape=[128,128,1], low=0, high=255, dtype=np.uint8),
    "audiogoal": spaces.Box(shape=[2,16000], low=-3.4028235e+38, high=3.4028235e+38, dtype=np.float32),
    "spectrogram": spaces.Box(shape=[65,26,2], low=-3.4028235e+38, high=3.4028235e+38, dtype=np.float32)
})
# single_observation_space = envs.observation_spaces[0]
# single_action_space = envs.action_spaces[0]

# single_observation_space, single_action_space

# Specify file name
analysis_trajs_filename = "cats_scenes_trajs_C_6_M_5_N_5__2023_06_01_10_41.bz2"

# Read the filtred trajectories data
## Default format is {cat -> { scenes -> traj: []}}
with open(analysis_trajs_filename, "rb") as f:
    cats_scenes_trajs_dict = cpkl.load(f)

## Compute the equivalent scenes cat trajs format
## {scenes -> { cat -> trajs: []}}
scenes_cats_trajs_dict = {}
for cat, cat_scenes_trajs in cats_scenes_trajs_dict.items():
    for scene, scenes_trajs in cat_scenes_trajs.items():
        if scene not in scenes_cats_trajs_dict.keys():
            scenes_cats_trajs_dict[scene] = {}
        
        scenes_cats_trajs_dict[scene][cat] = scenes_trajs

# Generic: load the dataset statistics
## Compute action coefficient for CEL of BC
dataset_stats_filepath = f"{args.dataset_path}/dataset_statistics.bz2"
# Override dataset statistics if the file already exists
if os.path.exists(dataset_stats_filepath):
    with open(dataset_stats_filepath, "rb") as f:
        dataset_statistics = cpkl.load(f)

# Extract some global metadata
# TARGET_SCENE_LIST = list(cats_scenes_trajs_dict[list(cats_scenes_trajs_dict.keys())[0]].keys())
TARGET_SCENE_LIST = list(dataset_statistics["scene_counts"].keys())
TARGET_SCENE_DICT = {scene: i for i, scene in enumerate(TARGET_SCENE_LIST)}
TARGET_CATEGORY_LIST = list(cats_scenes_trajs_dict.keys())
TARGET_CATEGORY_DICT = {cat: i for i, cat in enumerate(TARGET_CATEGORY_LIST)}

C = len(TARGET_CATEGORY_LIST) # C: total number of categories
M = len(TARGET_SCENE_LIST) # M: total number of rooms, assuming all categories has N trajs for a same set of scenes.

print(f"# of categories C: {C} | # of scenes: {M}")
print(f"TARGET_CATEGORY_DICT: {TARGET_CATEGORY_DICT}")
print(f"TARGET_SCENE_DICT: {TARGET_SCENE_DICT}")
print("")

# for catname, cat_scenes_trajs in cats_scenes_trajs_dict.items():
#     print(f"Cat: {catname}; Scenes: {[k for k in cat_scenes_trajs.keys()]}")

# Basic check of the scene -> categories fileted trajectories
# for scene, scenes_cat_trajs in scenes_cats_trajs_dict.items():
#     print(f"Scene: {scene}; Cats: {[k for k in scenes_cat_trajs.keys()]}")

# More detailed breakdown of the trajectories per categories then scenes
for catname, cat_scenes_trajs in cats_scenes_trajs_dict.items():
    print(f"{catname}:")
    for scene, scene_trajs in cat_scenes_trajs.items():
        traj_lengths = [len(traj_data["edd"]["done_list"]) for traj_data in scene_trajs]
        print(f"\t{scene}: {traj_lengths}")
    print("")

# More detailed breakdown of the trajectories per categories then scenes
for scene, scene_cats_trajs in scenes_cats_trajs_dict.items():
    print(f"{scene}")
    for cat, cat_trajs in scene_cats_trajs.items():
        traj_lengths = [len(traj_data["edd"]["done_list"]) for traj_data in cat_trajs]
        print(f"\t{cat}: {traj_lengths}")
    print("")

print("#######################################")
print("### Dataset statistics              ###")
print("#######################################")

print(dataset_statistics["category_counts"])

# region: Categories -> Scenes
## cats_scenes_trajs_dict: dictionary structured as: {category: {scene: [traj_data]}}
# TODO: add support for the device in case tensors are returned
def get_traj_data_by_category_scene_trajIdx(trajs_dicts, category, scene, trajIdx=0, tensorize=False, device="cpu"):
    # Get a single trajectory specified by idx, for a specificed category and scene
    # TODO: maybe fix the "depth" dimension here directly ?
    obs_list_dict = trajs_dicts[category][scene][trajIdx]["edd"]["obs_list"]
    done_list = trajs_dicts[category][scene][trajIdx]["edd"]["done_list"]

    obs_dict_list = []
    target_scene_idx_list, target_category_idx_list = [], []

    T = len(obs_list_dict["rgb"])
    for t in range(T):
        obs_dict_list.append({k: v[t] for k, v in obs_list_dict.items()})
        target_scene_idx_list.append(TARGET_SCENE_DICT[scene])
        target_category_idx_list.append(TARGET_CATEGORY_DICT[category])

    # Tensorize if required
    if tensorize:
        done_list__th = []
        obs_dict_list__th = []

        for t, (obs_dict, done) in enumerate(zip(obs_dict_list, done_list)):
            # done_list__th.append(th.Tensor(np.array([done])[None, :]))
            done_list__th.append(th.Tensor(np.array([done])).to(device)) # TODO: make sure that the deprecation warning stops showing up. Or always stay on current Torch version.
            tmp_dict = {}
            for k, v in obs_dict.items():
                if k == "depth":
                    v = np.array(v)[:, :, None] # From (H, W) -> (H, W, 1)
                tmp_dict[k] = th.Tensor(v)[None, :].to(device)
            
            obs_dict_list__th.append(tmp_dict)
        
        return obs_dict_list__th, done_list__th, target_scene_idx_list, target_category_idx_list

    return obs_dict_list, done_list, target_scene_idx_list, target_category_idx_list

def get_traj_data_by_category_scene(trajs_dicts, category, scene, max_scenes=0, tensorize=False, device="cpu"):
    # Get all trajectories for a specific category and scene
    obs_dict_list, done_list = [], []
    target_scene_idx_list, target_category_idx_list = [], []

    N_SCENES = len(trajs_dicts[category][scene])
    res_n_scenes = N_SCENES if max_scenes <= 0 else max_scenes

    for i in range(N_SCENES):
        traj_obs_dict_list, traj_done_list, target_scene_idxes, target_category_idxes = \
            get_traj_data_by_category_scene_trajIdx(trajs_dicts, category, scene, i, tensorize=tensorize, device=device)

        obs_dict_list.extend(traj_obs_dict_list)
        done_list.extend(traj_done_list)
        target_scene_idx_list.extend(target_scene_idxes)
        target_category_idx_list.extend(target_category_idxes)

        traj_length = len(traj_done_list)
        # print(f"Selected traj of length: {traj_length}")
        if i >= res_n_scenes - 1:
            break

    return obs_dict_list, done_list, target_scene_idx_list, target_category_idx_list

def get_traj_data_by_category(trajs_dicts, category, max_scenes=0, tensorize=False, device="cpu"):
    # Get all trajectories for a specific category, across all scenes and all trajectories
    obs_dict_list, done_list =[], []
    target_scene_idx_list, target_category_idx_list = [], []

    for scene in trajs_dicts[category].keys():
        scene_obs_dict_list, scene_done_list, target_scene_idxes, target_category_idxes = \
            get_traj_data_by_category_scene(trajs_dicts, category, scene, max_scenes=max_scenes, tensorize=tensorize, device=device)

        obs_dict_list.extend(scene_obs_dict_list)
        done_list.extend(scene_done_list)
        target_scene_idx_list.extend(target_scene_idxes)
        target_category_idx_list.extend(target_category_idxes)
    
    return obs_dict_list, done_list, target_scene_idx_list, target_category_idx_list
# endregion: Categories -> Scenes

# region: Scenes -> Categories
# TODO: add "return" for target categories and scenes label
## scenes_cats_trajs_dict: dictionary structured as: {scene: {category: [traj-data]}}
def get_traj_data_by_scene_category_trajIdx(trajs_dicts, scene, category, trajIdx=0, tensorize=False, device="cpu"):
    # Get a single trajectory specified by idx, for a specificed category and scene
    # TODO: maybe fix the "depth" dimension here directly ?
    obs_list_dict = trajs_dicts[scene][category][trajIdx]["edd"]["obs_list"]
    done_list = trajs_dicts[scene][category][trajIdx]["edd"]["done_list"]
    target_scene_idx_list, target_category_idx_list = [], []

    obs_dict_list = []
    T = len(obs_list_dict["rgb"])
    for t in range(T):
        obs_dict_list.append({k: v[t] for k, v in obs_list_dict.items()})
        target_scene_idx_list.append(TARGET_SCENE_DICT[scene])
        target_category_idx_list.append(TARGET_CATEGORY_DICT[category])

    # Tensorize if required
    if tensorize:
        done_list__th = []
        obs_dict_list__th = []

        for t, (obs_dict, done) in enumerate(zip(obs_dict_list, done_list)):
            # done_list__th.append(th.Tensor(np.array([done])[None, :]))
            done_list__th.append(th.Tensor(np.array([done])).to(device)) # TODO: make sure that the deprecation warning stops showing up. Or always stay on current Torch version.
            tmp_dict = {}
            for k, v in obs_dict.items():
                if k == "depth":
                    v = np.array(v)[:, :, None] # From (H, W) -> (H, W, 1)
                tmp_dict[k] = th.Tensor(v)[None, :].to(device)
            
            obs_dict_list__th.append(tmp_dict)
        
        return obs_dict_list__th, done_list__th, target_scene_idx_list, target_category_idx_list
        
    return obs_dict_list, done_list, target_scene_idx_list, target_category_idx_list

def get_traj_data_by_scene_category(trajs_dicts, scene, category, tensorize=False, device="cpu"):
    # Get all trajectories for a specific category and scene
    obs_dict_list, done_list = [], []
    target_scene_idx_list, target_category_idx_list = [], []

    for i in range(len(trajs_dicts[scene][category])):
        traj_obs_dict_list, traj_done_list, target_scene_idxes, target_category_idxes = \
            get_traj_data_by_scene_category_trajIdx(trajs_dicts, scene, category, i, tensorize=tensorize, device=device)

        obs_dict_list.extend(traj_obs_dict_list)
        done_list.extend(traj_done_list)
        target_scene_idx_list.extend(target_scene_idxes)
        target_category_idx_list.extend(target_category_idxes)

        traj_length = len(traj_done_list)
        # print(f"Selected traj of length: {traj_length}")

    return obs_dict_list, done_list, target_scene_idx_list, target_category_idx_list

def get_traj_data_by_scene(trajs_dicts, scene, tensorize=False, device="cpu"):
    # Get all trajectories for a specific category, across all scenes and all trajectories
    obs_dict_list, done_list =[], []
    target_scene_idx_list, target_category_idx_list = [], []
    
    for cat in trajs_dicts[scene].keys():
        cat_obs_dict_list, cat_done_list, target_scene_idxes, target_category_idxes = \
            get_traj_data_by_scene_category(trajs_dicts, scene, cat, tensorize=tensorize, device=device)

        obs_dict_list.extend(cat_obs_dict_list)
        done_list.extend(cat_done_list)
        target_scene_idx_list.extend(target_scene_idxes)
        target_category_idx_list.extend(target_category_idxes)
    
    return obs_dict_list, done_list, target_scene_idx_list, target_category_idx_list
# endregion: Scenes -> Categories


# TODO seeding for reproducibility ? Make sure that we can control the generated episode trajs ?

# Loading pretrained agent
import models
import models2
from models import ActorCritic, ActorCritic2, Perceiver_GWT_GWWM_ActorCritic
from models2 import GWTAgent, GWTAgent_BU, GWTAgent_TD

MODEL_VARIANTS_TO_STATEDICT_PATH = {

    # TODO: rename the random baseline to show SAVi or AvNav ?
    # region: Random GRU Baseline
    # "ppo_gru__random": {
    #     "pretty_name": "GRU Random",
    #     "state_dict_path": ""
    # },
    # # Random PGWT Baseline
    # "ppo_pgwt__random": {
    #     "pretty_name": "TransRNN Random",
    #     "state_dict_path": ""
    # },
    # endregion: Random GRU Baseline
    
    # region: SAVi BC variants: trained using RGB + Depth + Spectrogram to 5M steps
    # "ppo_gru__bc__SAVi": {
    #     "pretty_name": "[SAVi] PPO GRU (BC)",
    #     "state_dict_path": "/home/rousslan/random/rl/exp-logs/ss-hab-bc/"
    #         "ppo_bc__savi_ss1__rgb_depth_spectro__gru_seed_111__2023_05_23_23_17_03_387659.musashi"
    #         "/models/ppo_agent.4995001.ckpt.pth"
    # },
    # "ppo_pgwt__bc__SAVi": {
    #     "pretty_name": "[SAVi] PPO PGWT (BC)",
    #     "state_dict_path": "/home/rousslan/random/rl/exp-logs/ss-hab-bc/"
    #         "ppo_bc__savi_ss1__rgb_depth_spectro__pgwt__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats_seed_111__2023_05_23_23_17_04_044443.musashi"
    #         "/models/ppo_agent.4995001.ckpt.pth"
    # },
    # endregion: SAVi BC variants: trained using RGB + Depth + Spectrogram to 5M steps

    # region: SAVi BC variants; trained using RGB + Spectrogram to 2.5M steps
    # "ppo_gru__bc__SAVi": {
    #     "pretty_name": "[SAVi] PPO GRU (BC)",
    #     "state_dict_path": "/home/rousslan/random/rl/exp-logs/ss-hab-bc/"
    #         "ppo_bc__savi_ss1_rgb_spectro__gru_seed_222__2023_06_08_18_10_08_906803.musashi"
    #         "/models/ppo_agent.2490001.ckpt.pth"
    # },
    # "ppo_pgwt__bc__SAVi": {
    #     "pretty_name": "[SAVi] PPO PGWT (BC)",
    #     "state_dict_path": "/home/rousslan/random/rl/exp-logs/ss-hab-bc/"
    #         "ppo_bc__savi_ss1_rgb__spectro__pgwt__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats_seed_111__2023_06_08_18_01_21_322731.musashi"
    #         "/models/ppo_agent.2490001.ckpt.pth"
    # },
    # endregion: SAVi BC variants; trained using RGB + Spectrogram to 2.5M steps

    # region: SAVi BC variants; trained using RGBD + Spectrogram ; trained up to 5M steps
    # "ppo_bc__rgbd_spectro__gru__SAVi": {
    #     "pretty_name": "[SAVi BC] PPO GRU | RGB Spectro",
    #     "state_dict_path": "/home/rousslan/random/rl/exp-logs/ss-hab-bc/"
    #         "ppo_bc__savi_ss1_rgbd_spectro__gru_seed_111__2023_06_10_16_05_39_999286.musashi"
    #         "/models/ppo_agent.4995001.ckpt.pth"
    # },
    # "ppo_bc__rgbd_spectro__pgwt__SAVi": {
    #     "pretty_name": "[SAVi BC] PPO TransRNN | RGB Spectro",
    #     "state_dict_path": "/home/rousslan/random/rl/exp-logs/ss-hab-bc/"
    #         "ppo_bc__savi_ss1_rgbd__spectro__pgwt__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats_seed_111__2023_06_10_16_05_37_098602.musashi"
    #         "/models/ppo_agent.4995001.ckpt.pth"
    # },
    # endregion: SAVi BC variants; trained using RGBD + Spectrogram ; trained up to 5M steps

    # region: SAVi BC variants; trained using RGB + Spectrogram to 10M steps steps
    # "ppo_bc__rgb_spectro__gru__SAVi": {
    #     "pretty_name": "[SAVi BC] PPO GRU | RGB Spectro",
    #     "state_dict_path": "/home/rousslan/random/rl/exp-logs/ss-hab-bc/"
    #         "ppo_bc__savi_ss1_rgb_spectro__gru_seed_222__2023_06_17_21_24_12_718867.musashi"
    #         "/models/ppo_agent.9990001.ckpt.pth"
    # },
    # "ppo_bc__rgb_spectro__pgwt__SAVi": {
    #     "pretty_name": "[SAVi BC] PPO TransRNN | RGB Spectro",
    #     "state_dict_path": "/home/rousslan/random/rl/exp-logs/ss-hab-bc/"
    #         "ppo_bc__savi_ss1_rgb__spectro__pgwt__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats_seed_222__2023_06_17_21_24_10_884437.musashi"
    #         "/models/ppo_agent.9990001.ckpt.pth"
    # },
    # endregion: SAVi BC variants; trained using RGB + Spectrogram to 10M steps steps

    # region: SAVi BC variants GRU v2; trained using RGB + Spectrogram to 10M steps steps
    # "ppo_bc__rgb_spectro__gru2__SAVi": {
    #     "pretty_name": "[SAVi BC] PPO GRU | RGB Spectro",
    #     "state_dict_path": "/home/rousslan/random/rl/exp-logs/ss-hab-bc/"
    #         "ppo_bc__savi_ss1_rgb_spectro__gru2_seed_222__2023_07_24_13_54_07_163432.musashi"
    #         "/models/ppo_agent.9990001.ckpt.pth"
    # },
    # endregion: SAVi BC variants GRU v2; trained using RGB + Spectrogram to 10M steps steps

    # region: SAVi BC variants; Custom GWT Agent based on SAGAN; trained using RGB + Spectrogram to 10M steps steps
    # "ppo_bc__savi_ss1_rgb_spectro__gwt_bu_td": {
    #     "pretty_name": "[SAVi BC] PPO Cstm TransRNN BU-TD | RGB Spectro",
    #     "state_dict_path": "/home/rousslan/random/rl/exp-logs/ss-hab-bc/"
    #         "ppo_bc__savi_ss1_rgb_spectro__gwt_bu_td_seed_111__2023_07_21_19_27_25_674410.musashi"
    #         "/models/ppo_agent.9990001.ckpt.pth"
    # },
    # endregion: SAVi BC variants; Custom GWT Agent based on SAGAN; trained using RGB + Spectrogram to 10M steps steps

    # region: SAVi BC variants; Custom GWT Agent based on SAGAN, BU only; trained using RGB + Spectrogram to 10M steps steps
    "ppo_bc__savi_ss1_rgb_spectro__gwt_bu": {
        "pretty_name": "[SAVi BC] PPO Cstm TransRNN BU | RGB Spectro",
        "state_dict_path": "/home/rousslan/random/rl/exp-logs/ss-hab-bc/"
            "ppo_bc__savi_ss1_rgb_spectro__gwt_bu_seed_111__2023_07_25_17_47_43_676298.musashi"
            "/models/ppo_agent.9990001.ckpt.pth"
    },
    # endregion: SAVi BC variants; Custom GWT Agent based on SAGAN, BU only; trained using RGB + Spectrogram to 10M steps steps

    # region: SAVi BC variants; Custom GWT Agent based on SAGAN TD Only; trained using RGB + Spectrogram to 10M steps steps
    # "ppo_bc__savi_ss1_rgb_spectro__gwt_td": {
    #     "pretty_name": "[SAVi BC] PPO Cstm TransRNN TD | RGB Spectro",
    #     "state_dict_path": "/home/rousslan/random/rl/exp-logs/ss-hab-bc/"
    #         "ppo_bc__savi_ss1_rgb_spectro__gwt_td_seed_111__2023_07_30_11_31_48_661642.musashi"
    #         "/models/ppo_agent.9990001.ckpt.pth"
    # },
    # endregion: SAVi BC variants; Custom GWT Agent based on SAGAN TD Only; trained using RGB + Spectrogram to 10M steps steps
}

# dev = th.device("cpu")
dev = th.device("cuda") # NOTE / TODO: using GPU to be more efficient ?

# 'variant named' indexed 'torch agent'
MODEL_VARIANTS_TO_AGENTMODEL = {}

for k, v in MODEL_VARIANTS_TO_STATEDICT_PATH.items():
    args_copy = copy.copy(args)
    # Override args depending on the model in use
    if k.__contains__("gru__SAVi"):
        print(f"Loaded GRU v1")
        agent = ActorCritic(single_observation_space, single_action_space, args, extra_rgb=False,
            analysis_layers=models.GRU_ACTOR_CRITIC_DEFAULT_ANALYSIS_LAYER_NAMES)
    elif k.__contains__("gru2__SAV"):
        print(f"Loaded GRU v2")
        agent = ActorCritic2(single_observation_space, single_action_space, args, extra_rgb=False,
            analysis_layers=models.GRU_ACTOR_CRITIC_DEFAULT_ANALYSIS_LAYER_NAMES)
    elif k.__contains__("pgwt"):
        print(f"Loaded PGWT")
        agent = Perceiver_GWT_GWWM_ActorCritic(single_observation_space, single_action_space, args, extra_rgb=False,
            analysis_layers=models.PGWT_GWWM_ACTOR_CRITIC_DEFAULT_ANALYSIS_LAYER_NAMES + ["state_encoder.ca.mha"])
    elif k.__contains__("gwt_bu_td"):
        print(f"Loaded GWT v2")
        agent = GWTAgent(single_action_space, args,
            analysis_layers=models2.GWTAGENT_DEFAULT_ANALYSIS_LAYER_NAMES)
    elif k.__contains__("gwt_bu"):
        agent = GWTAgent_BU(single_action_space, args,
            analysis_layers=models2.GWTAGENT_DEFAULT_ANALYSIS_LAYER_NAMES)
    elif k.__contains__("gwt_td"):
        agent = GWTAgent_TD(single_action_space, args,
            analysis_layers=models2.GWTAGENT_DEFAULT_ANALYSIS_LAYER_NAMES)

    agent.eval()
    # Load the model weights
    # TODO: add map location device to use CPU only ?
    if v["state_dict_path"] != "":
        agent_state_dict = th.load(v["state_dict_path"], map_location=dev)
        agent.load_state_dict(agent_state_dict)
    agent = agent.to(dev)

    MODEL_VARIANTS_TO_AGENTMODEL[k] = agent


# Saliency map computation
# Precompute baseline values for non-zeros occlusion method

print("##########################################")
print("### Pre-compute sal. map. avg_baseline ###")
print("##########################################")

# TODO: consider caching this
# TODO: consider storing as numpy, then copying to torch.device only when needs be
CAT_SCENE_TRAJS_OCCLUSION_BASELINES = {}

for catname, cat_scenes_trajs in cats_scenes_trajs_dict.items():
    print(f"{catname}:")
    if catname not in CAT_SCENE_TRAJS_OCCLUSION_BASELINES.keys():
        CAT_SCENE_TRAJS_OCCLUSION_BASELINES[catname] = {}

    for scene, scene_trajs in cat_scenes_trajs.items():
        traj_lengths = [len(traj_data["edd"]["done_list"]) for traj_data in scene_trajs]
        print(f"  {scene}: {traj_lengths}")

        if scene not in CAT_SCENE_TRAJS_OCCLUSION_BASELINES[catname].keys():
            CAT_SCENE_TRAJS_OCCLUSION_BASELINES[catname][scene] = {}

        for traj_idx, traj_data in enumerate(scene_trajs):
            # Load the data, perform ablations if necessary
            obs_dict_list, done_list, target_scene_idx_list, target_category_idx_list = \
                get_traj_data_by_category_scene_trajIdx(cats_scenes_trajs_dict, catname, scene, traj_idx, tensorize=True)
            
            for t, obs_dict in enumerate(obs_dict_list):
                for k, v in obs_dict.items():
                    # if k not in ["rgb", "depth", "spectrogram"]:
                    if k not in ["rgb", "depth", "spectrogram"]:
                        continue    
                    if k not in CAT_SCENE_TRAJS_OCCLUSION_BASELINES[catname][scene].keys():
                        CAT_SCENE_TRAJS_OCCLUSION_BASELINES[catname][scene][k] = []
                    
                    CAT_SCENE_TRAJS_OCCLUSION_BASELINES[catname][scene][k].append(v)

        for k, v in CAT_SCENE_TRAJS_OCCLUSION_BASELINES[catname][scene].items():
            CAT_SCENE_TRAJS_OCCLUSION_BASELINES[catname][scene][k] = \
                th.stack(v).mean(dim=0)

## Occlusion helpers
def apply_occlusion_obs_dict_list(obs_dict_list, occ_step, occ_type="zeros", occ_target="rgb", occ_baseline_dict=None, device="cpu"):
    # Support "rgb", "depth", and "spectrogram" fields so far

    # NOTE: The following is uneeded, but just kept as reference
    if occ_target in ["rgb", "depth"]:
        H, W = 128, 128
    elif occ_target in ["spectrogram"]:
        H, W = 65, 26
    else:
        raise NotImplementedError(f"Unsupported occlusion for occ_target: {occ_target}")

    occluded_obs_dict_list = []
    occlusion_mask_dict_list = []

    ep_length = len(obs_dict_list)

    for t in range(ep_length):
        occluded_obs_dict = {k: [v] for k, v in obs_dict_list[t].items()}
        occlusion_mask_dict = {k: [th.zeros_like(v)] for k, v in obs_dict_list[t].items()}

        for k, v in obs_dict_list[t].items():
            for m in range(occ_step, H + occ_step, occ_step):
                for n in range(occ_step, W + occ_step, occ_step):

                    # TODO: improve the logic for better 
                    # if k == occ_target:
                    #   if k is either "rgb", or "spectrogram":
                    #       apply occlusion
                    # else:
                    #   copy the value of the k field as is
                    
                    if k in ["rgb", "spectrogram"] and k == occ_target:
                        # rgb_obs = copy.deepcopy(v)
                        rgb_obs = v
                        occ_mask = th.zeros_like(rgb_obs)
                        occ_mask[0, m-occ_step:m, n-occ_step:n, :] = 1

                        if occ_type == "zeros":
                            occ_data = th.zeros_like(rgb_obs)
                        elif occ_type == "avg_baseline":
                            occ_data = occ_baseline_dict[k].to(device)
                        
                        occ_rgb_obs = th.where(occ_mask.bool(), occ_data, rgb_obs)
                        
                        occluded_obs_dict[k].append(occ_rgb_obs)
                        occlusion_mask_dict[k].append(occ_mask)
                    else:
                        occ_mask = th.zeros_like(v)

                        # occluded_obs_dict[k].append(copy.deepcopy(v))
                        occluded_obs_dict[k].append(v)
                        occlusion_mask_dict[k].append(occ_mask)
        
        occluded_obs_dict_list.append({kk: th.cat(vv) for kk, vv in occluded_obs_dict.items()})
        occlusion_mask_dict_list.append({kk: th.cat(vv) for kk, vv in occlusion_mask_dict.items()})

        th.cuda.empty_cache()

    # "occluded_obs_dict_list"'s index 0 is the non occluded image
    # Length: 1 + number of occluded obs, number of occ. obs
    return occluded_obs_dict_list, occlusion_mask_dict_list

## Helper for cleaning up and preparing the recorded intermediate features
def process_analysis_feats_raw__occ_variant(raw_dict):
    result_dict = {}

    for k, v in raw_dict.items():
        if isinstance(v[0], th.Tensor):
            new_v = th.stack(v, dim=0).cpu()
        elif isinstance(v[0], tuple):
            new_v = None # TODO
            n_elements = len(v[0])
            elements = [[] for _ in range(n_elements)]
            for j in range(n_elements):
                for i in range(len(v)):
                    elements[j].append(v[i][j])
            
            new_v = [th.stack(vv, dim=0).cpu() for vv in elements]
        else:
            raise Exception(f"Unhandled type: {v[0].__class__}")
    
        result_dict[k] = new_v
    
    return result_dict

# occluded_obs_dict_list, occlusion_mask_dict_list = apply_occlusion_obs_dict_list(obs_dict_list, occ_step=32)

# Cache config
SALMAP_CACHE_DIRNAME = f"cached_data/saliency_maps/{analysis_trajs_filename.split('.')[0]}"
if not os.path.exists(SALMAP_CACHE_DIRNAME):
    os.makedirs(SALMAP_CACHE_DIRNAME, exist_ok=True)

ALL_SALIENCIES = None

# Config of the analysis
OCC_TARGETS = ["rgb", "spectrogram"] # All: ["rgb", "spectrogram"]
CATEGORIES_OF_INTEREST = ["plant", "chair", "table", "cushion", "cabinet", "picture"] # All: ["plant", "chair", "table", "cushion", "cabinet", "picture"]
SCENES_OF_INTEREST = ["gTV8FGcVJC9", "b8cTxDM8gDG", "D7N2EKCX4Sj"] # All: ["gTV8FGcVJC9", "D7N2EKCX4Sj", "vyrNrziPKCB", "Vvot9Ly1tCj", "b8cTxDM8gDG"]
TRAJS_OF_INTEREST = [0, 1, 2, 3, 4] # [0, 1, 2, 3, 4]

# LAYERS_OF_INTEREST = ["visual_encoder.cnn.7", "audio_encoder.cnn.7", "state_encoder"]
LAYERS_OF_INTEREST = ["visual_embedding", "audio_embedding", "state_encoder", "visual_encoder.cnn.7", "audio_encoder.cnn.7"]

OCCLUSION_TYPES = ["avg_baseline"] # Default: ["zeros", "avg_baseline"]

OCC_TARGET_STEPS = {
    "rgb": 8,
    "spectrogram": 2
}


import datetime
GEN_TIMESTAMP = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

ALPHA = 0.2 # For saliency maps plots

# Helpers
def scale_values(a, old_min=0, old_max=1, new_min=-1, new_max=1):
    return ((a - old_min) * (new_max - new_min)) / (old_max - old_min + 1e-8) + new_min

# Caching if not already exisitng
# If it does exist, then consider just skip the whole computation loop
allsall_filename = f"{SALMAP_CACHE_DIRNAME}/all_saliencies.{GEN_TIMESTAMP}.bz2"

# Placeholder for all saliencyes
ALL_SALIENCIES = {
    k: {
        kk: {
            kkk: {
                occ_target: {
                    occ_type: {
                        agent_variant: {} \
                            for agent_variant in MODEL_VARIANTS_TO_AGENTMODEL.keys()
                    } for occ_type in OCCLUSION_TYPES
                } for occ_target in OCC_TARGETS
            } for kkk in range(len(vv))
        } for kk, vv in v.items()
    } for k, v in cats_scenes_trajs_dict.items()
} # Holds all the saliencies computed

if not os.path.exists(allsall_filename):
    for catname, cat_scenes_trajs in cats_scenes_trajs_dict.items():
        # Skip categories that are not of interest
        if catname not in CATEGORIES_OF_INTEREST:
            continue
        print(f"{catname}:")
        
        # For plot and data caching
        cat_dirname = f"{SALMAP_CACHE_DIRNAME}/saliencies_data/{GEN_TIMESTAMP}/{catname}"

        for scene, scene_trajs in cat_scenes_trajs.items():
            # Skip scenes that are not of interest
            if scene not in SCENES_OF_INTEREST:
                continue
            
            # For plot and data caching
            scene_dirname = f"{cat_dirname}/{scene}"

            traj_lengths = [len(traj_data["edd"]["done_list"]) for traj_data in scene_trajs]
            print(f"  {scene}: {traj_lengths}")

            for traj_idx, traj_data in enumerate(scene_trajs):
                if traj_idx not in TRAJS_OF_INTEREST:
                    continue
                print(f"    Traj {traj_idx}")

                # For plot and data caching
                traj_dirname = f"{scene_dirname}/{traj_idx}"

                # Load the current trajectory's data
                obs_dict_list, done_list, target_scene_idx_list, target_category_idx_list = \
                    get_traj_data_by_category_scene_trajIdx(cats_scenes_trajs_dict, catname, scene, traj_idx, tensorize=True, device=dev)
                
                traj_length = len(obs_dict_list)

                # Iterate over the observation fields to analyze using occlusion and sal. maps
                for occ_target in OCC_TARGETS:
                    print(f"      Occlusion target: {occ_target}")

                    # For plot and data caching
                    occ_target_dirname = f"{traj_dirname}/{occ_target}"

                    # Iterate over the occlusion types
                    for occ_type in OCCLUSION_TYPES:
                        print(f"        Occlusion type: {occ_type}")
                        # For plot and data caching
                        occ_type_dirname = f"{occ_target_dirname}/{occ_type}"

                        if occ_type == "zeros":
                            occluded_obs_dict_list, occlusion_mask_dict_list = apply_occlusion_obs_dict_list(
                                obs_dict_list, 
                                occ_step=OCC_TARGET_STEPS[occ_target],
                                device=dev # size of the sliding occlusion window
                            )
                        elif occ_type == "avg_baseline":
                            occ_baseline_dict = CAT_SCENE_TRAJS_OCCLUSION_BASELINES[catname][scene]
                            occluded_obs_dict_list, occlusion_mask_dict_list = apply_occlusion_obs_dict_list(
                                obs_dict_list,
                                occ_target=occ_target,
                                occ_step=OCC_TARGET_STEPS[occ_target],
                                occ_type=occ_type,
                                occ_baseline_dict=occ_baseline_dict,
                                device=dev
                            )
                        else:
                            raise NotImplementedError(f"Unsupported occlusion type: {occ_type}")
                    
                        # Origianl image + occluded image (== H // occ_step)
                        B = occluded_obs_dict_list[0]["rgb"].shape[0]

                        for agent_variant, agent_model in MODEL_VARIANTS_TO_AGENTMODEL.items():
                            if agent_variant in ["ppo_gru__random", "ppo_pgwt__random"]:
                                # Skip irrelevant variants
                                continue
                            print(f"          Model variant: {agent_variant}")

                            # For plot and data caching
                            agent_variant_dirname = f"{occ_type_dirname}/{agent_variant}"

                            agent_raw_features = {}
                            # NOTE: Do we really need to keep this in a dictionary ?
                            if agent_variant.__contains__("gru") or \
                                agent_variant.__contains__("gwt_bu_td") or \
                                agent_variant.__contains__("gwt_bu") or \
                                agent_variant.__contains__("gwt_td"):
                                agent_rnn_state = th.zeros((1, B, args.hidden_size), device=dev)
                            elif agent_variant.__contains__("pgwt"):
                                agent_rnn_state = agent_model.state_encoder.latents.clone().repeat(B, 1, 1)

                            for t, (obs_th, done_th) in enumerate(zip(occluded_obs_dict_list, done_list)):
                                masks = 1. - done_th[:, None].repeat(B, 1)

                                with th.no_grad():
                                    _, _, _, _, _, _, agent_rnn_state, _ = \
                                        agent_model.act(obs_th, agent_rnn_state, masks)
                                    # agent_rnn_state of shape [1, 1+n_input_locs, H]
                                    # We thus need to copy the rnn_state at step 0 to steps 1:n_input_locs+1 indeed
                                    if agent_variant.__contains__("gwt_bu_td") or \
                                        agent_variant.__contains__("gwt_bu") or \
                                        agent_variant.__contains__("gwt_td"):
                                        agent_rnn_state[1:, :] = agent_rnn_state[0, :][None, :].repeat(B-1, 1)
                                    elif agent_variant.__contains__("gru"):
                                        agent_rnn_state[:, 1:, :] = agent_rnn_state[:, 0, :][:, None, :].repeat(1, B-1, 1)
                                    elif agent_variant.__contains__("pgwt"):
                                        # raise NotImplementedError(f"PGWT's hidden state vector reset not implemented yet ?")
                                        agent_rnn_state[1:, :, :] = agent_rnn_state[0, :, :].repeat(B-1, 1, 1)

                                # Collecting intermediate layers results
                                for k, v in agent_model._features.items():
                                    if k not in LAYERS_OF_INTEREST:
                                        continue # Skip irrelevant layers
                                    if k not in list(agent_raw_features.keys()):
                                        agent_raw_features[k] = []
                                    agent_raw_features[k].append((v[0].cpu(), v[1].cpu()) if isinstance(v, tuple) else v.cpu())
                            
                            agent_layers_features = process_analysis_feats_raw__occ_variant(agent_raw_features)
                            del agent_raw_features
                            # TODO: store the features: agent_layers_features ?

                            # Compute the saliency for each intermediate layer of interest of the agent variant
                            for intermediate_layer, traj_occlusions_features in agent_layers_features.items():
                                print(f"            Compute saliencies for layer: {intermediate_layer}")
                                if intermediate_layer in ["state_encoder"]:
                                    if agent_variant.__contains__("gwt_bu_td") or \
                                       agent_variant.__contains__("gru2") or \
                                       agent_variant.__contains__("gwt_bu") or \
                                       agent_variant.__contains__("gwt_td"):
                                        # Do nothing because we are using custom GRUCell instead of SS baselines'
                                        # The latter actually returns two vectors of shape [B, H], [1, B, H] for compatiblity reasons
                                        pass
                                    else:
                                        traj_occlusions_features = traj_occlusions_features[0]

                                # For plot and data caching
                                layer_dirname = f"{agent_variant_dirname}/{intermediate_layer}"

                                n_input_locs = traj_occlusions_features.shape[1] - 1 # How mnay occlusion "centers"
                                traj_layer_saliencies = []

                                for t in range(traj_length):
                                    no_occ_features = traj_occlusions_features[t][0]
                                    occ_saliencies = []

                                    for i in range(1, 1+n_input_locs):
                                        i_occ_saliency = (no_occ_features - traj_occlusions_features[t][i]).norm(2)
                                        occ_saliencies.append(i_occ_saliency.item())
                                    traj_layer_saliencies.append(occ_saliencies)
                                
                                saliencies = traj_layer_saliencies = np.array(traj_layer_saliencies)

                                # Store for later usage:
                                ALL_SALIENCIES[catname][scene][traj_idx][occ_target][occ_type][agent_variant][intermediate_layer] = \
                                    traj_layer_saliencies
                                
                                # Individual cache file
                                os.makedirs(layer_dirname, exist_ok=True)
                                layercache_filename = f"{layer_dirname}/saliencies.bz2"

                                if not os.path.exists(layercache_filename):
                                    with open(layercache_filename, "wb") as f:
                                        cpkl.dump(traj_layer_saliencies, f)
                                
                                # Generate plot, save to disk
                                # Scale the saliencies over all the values in the given trajectory
                                traj_saliency_min, traj_saliency_max = saliencies.min(), saliencies.max()
                                
                                scaled_saliencies = scale_values(saliencies, traj_saliency_min, traj_saliency_max, 0.0, 1.0)
                                n_input_locs = B - 1

                                fig, axes = plt.subplots(1, traj_length, figsize=(traj_length * 6, 6 + 1.5))

                                # Get min and max of spectrograms values across a single trajectory
                                traj_spect = np.array([occluded_obs_dict_list[t]["spectrogram"][0].cpu().numpy() for t in range(traj_length)])
                                spectr_min, spectr_max = traj_spect.min(), traj_spect.max()

                                for t, (obs_dict, obs_occ_mask) in enumerate(zip(occluded_obs_dict_list, occlusion_mask_dict_list)):
                                    # Plot the default image
                                    # NOTE: the [0] indexing in [OCC_TARGET][0] is because we store the non-occluded image at that positon
                                    # All the other indices 1: have different occlusion variants
                                    if occ_target == "rgb":
                                        img_data = obs_dict[occ_target][0].cpu().numpy().astype(np.uint8)
                                        axes[t].imshow(img_data)
                                    elif occ_target == "spectrogram":
                                        img_data = th.cat([obs_dict[occ_target][0][:, :, i] for i in range(2)], dim=1).cpu().numpy()
                                        img_data = scale_values(img_data, old_min=spectr_min, old_max=spectr_max, new_min=0., new_max=1.)
                                        axes[t].imshow(img_data, cmap="gray")
                                    else:
                                        raise NotImplementedError(f"Unsupported input data for saliency map: {occ_target}")
                                    
                                    axes[t].tick_params(axis="both", which="both", bottom=False, top=False, left=False, labelleft=False, labelbottom=False)
                                    axes[t].set_title(f"t = {t}", fontsize=20)

                                    # Generated mask based on saliencies
                                    saliency_mask = np.zeros_like(obs_dict[occ_target][0].cpu().numpy(), dtype=np.float32)
                                    
                                    for i in range(1, n_input_locs+1):
                                        i_occ_mask = obs_occ_mask[occ_target][i].cpu().numpy()
                                        saliency_mask = np.where(i_occ_mask, scaled_saliencies[t][i-1], saliency_mask)
                                    
                                    if occ_target == "rgb":
                                        saliency_mask = saliency_mask[:, :, 0]
                                    elif occ_target == "spectrogram":
                                        saliency_mask = np.concatenate([saliency_mask[:, :, i] for i in range(2)], 1)
                                    
                                    axes[t].imshow(saliency_mask, alpha=ALPHA, cmap="jet", vmin=0.0, vmax=1.0)
                                    # print(t)
                                    # print(obs_dict["rgb"].shape)
                                    # print(f"            {obs_occ_mask['rgb'].shape}")
                                
                                fig.suptitle(f"Saliency map for cat: {catname} | Scene: {scene} | Occ. target: {occ_target} | Occ. type: {occ_type} | Traj: {traj_idx} | {agent_variant} | {intermediate_layer} ", fontsize=28)
                                fig.tight_layout()

                                # Save to file
                                salmap_plot_dirname = f"{SALMAP_CACHE_DIRNAME}/saliencies_plots/{GEN_TIMESTAMP}/" + \
                                    f"{catname}/{scene}/{traj_idx}/{occ_target}/{occ_type}/{agent_variant}/{intermediate_layer}"
                                os.makedirs(salmap_plot_dirname, exist_ok=True)
                                salmap_plot_filename = f"{salmap_plot_dirname}/salmap_plot.png"
                                fig.savefig(salmap_plot_filename)
                                plt.close(fig)
                            
                            # Cleanup
                            del agent_layers_features, agent_rnn_state
                            th.cuda.empty_cache() # Clear up GPU cache that gets quite high

                        # Clean up:
                        del occluded_obs_dict_list, occlusion_mask_dict_list
                        th.cuda.empty_cache() # Clear up GPU cache that gets quite high
                
                del obs_dict_list, done_list, target_scene_idx_list, target_category_idx_list
                th.cuda.empty_cache() # Clear up GPU cache that gets quite high
                
                # break # traj_idx ; debug mostly

            # break # scene ; debug mostly

        # break # category ; debug mostly

    # Cache to disk in case the file is not detected
    if not os.path.exists(allsall_filename):
        with open(allsall_filename, "wb") as f:
            cpkl.dump(ALL_SALIENCIES, f)
    
    for catname, cat_scenes_trajs in cats_scenes_trajs_dict.items():
        # Skip categories that are not of interest
        if catname not in CATEGORIES_OF_INTEREST:
            continue
        print(f"{catname}:")
        
        # For plot and data caching
        cat_dirname = f"{SALMAP_CACHE_DIRNAME}/saliencies_data/{GEN_TIMESTAMP}/{catname}"

        for scene, scene_trajs in cat_scenes_trajs.items():
            # Skip scenes that are not of interest
            if scene not in SCENES_OF_INTEREST:
                continue
            
            # For plot and data caching
            scene_dirname = f"{cat_dirname}/{scene}"

            traj_lengths = [len(traj_data["edd"]["done_list"]) for traj_data in scene_trajs]
            print(f"  {scene}: {traj_lengths}")

            for traj_idx, traj_data in enumerate(scene_trajs):
                if traj_idx not in TRAJS_OF_INTEREST:
                    continue
                print(f"    Traj {traj_idx}")

                # For plot and data caching
                traj_dirname = f"{scene_dirname}/{traj_idx}"

                # Load the current trajectory's data
                obs_dict_list, done_list, target_scene_idx_list, target_category_idx_list = \
                    get_traj_data_by_category_scene_trajIdx(cats_scenes_trajs_dict, catname, scene, traj_idx, tensorize=True, device=dev)
                
                traj_length = len(obs_dict_list)

                # Iterate over the observation fields to analyze using occlusion and sal. maps
                for occ_target in OCC_TARGETS:
                    print(f"      Occlusion target: {occ_target}")

                    # For plot and data caching
                    occ_target_dirname = f"{traj_dirname}/{occ_target}"

                    # Iterate over the occlusion types
                    for occ_type in OCCLUSION_TYPES:
                        print(f"        Occlusion type: {occ_type}")
                        # For plot and data caching
                        occ_type_dirname = f"{occ_target_dirname}/{occ_type}"

                        if occ_type == "zeros":
                            occluded_obs_dict_list, occlusion_mask_dict_list = apply_occlusion_obs_dict_list(
                                obs_dict_list, 
                                occ_step=OCC_TARGET_STEPS[occ_target],
                                device=dev # size of the sliding occlusion window
                            )
                        elif occ_type == "avg_baseline":
                            occ_baseline_dict = CAT_SCENE_TRAJS_OCCLUSION_BASELINES[catname][scene]
                            occluded_obs_dict_list, occlusion_mask_dict_list = apply_occlusion_obs_dict_list(
                                obs_dict_list,
                                occ_target=occ_target,
                                occ_step=OCC_TARGET_STEPS[occ_target],
                                occ_type=occ_type,
                                occ_baseline_dict=occ_baseline_dict,
                                device=dev
                            )
                        else:
                            raise NotImplementedError(f"Unsupported occlusion type: {occ_type}")
                    
                        # Origianl image + occluded image (== H // occ_step)
                        B = occluded_obs_dict_list[0]["rgb"].shape[0]

                        for agent_variant, agent_model in MODEL_VARIANTS_TO_AGENTMODEL.items():
                            if agent_variant in ["ppo_gru__random", "ppo_pgwt__random"]:
                                # Skip irrelevant variants
                                continue
                            print(f"          Model variant: {agent_variant}")

                            # For plot and data caching
                            agent_variant_dirname = f"{occ_type_dirname}/{agent_variant}"

                            agent_raw_features = {}
                            # NOTE: Do we really need to keep this in a dictionary ?
                            if agent_variant.__contains__("gru") or agent_variant.__contains__("gwt_bu_td"):
                                agent_rnn_state = th.zeros((1, B, args.hidden_size), device=dev)
                            elif agent_variant.__contains__("pgwt"):
                                agent_rnn_state = agent_model.state_encoder.latents.clone().repeat(B, 1, 1)

                            for t, (obs_th, done_th) in enumerate(zip(occluded_obs_dict_list, done_list)):
                                masks = 1. - done_th[:, None].repeat(B, 1)

                                with th.no_grad():
                                    _, _, _, _, _, _, agent_rnn_state, _ = \
                                        agent_model.act(obs_th, agent_rnn_state, masks)
                                    # agent_rnn_state of shape [1, 1+n_input_locs, H]
                                    # We thus need to copy the rnn_state at step 0 to steps 1:n_input_locs+1 indeed
                                    if agent_variant.__contains__("gwt_bu_td"):
                                        agent_rnn_state[1:, :] = agent_rnn_state[0, :][None, :].repeat(B-1, 1)
                                    elif agent_variant.__contains__("gru"):
                                        agent_rnn_state[:, 1:, :] = agent_rnn_state[:, 0, :][:, None, :].repeat(1, B-1, 1)
                                    elif agent_variant.__contains__("pgwt"):
                                        # raise NotImplementedError(f"PGWT's hidden state vector reset not implemented yet ?")
                                        agent_rnn_state[1:, :, :] = agent_rnn_state[0, :, :].repeat(B-1, 1, 1)

                                # Collecting intermediate layers results
                                for k, v in agent_model._features.items():
                                    if k not in LAYERS_OF_INTEREST:
                                        continue # Skip irrelevant layers
                                    if k not in list(agent_raw_features.keys()):
                                        agent_raw_features[k] = []
                                    agent_raw_features[k].append((v[0].cpu(), v[1].cpu()) if isinstance(v, tuple) else v.cpu())
                            
                            agent_layers_features = process_analysis_feats_raw__occ_variant(agent_raw_features)
                            del agent_raw_features
                            # TODO: store the features: agent_layers_features ?

                            # Compute the saliency for each intermediate layer of interest of the agent variant
                            for intermediate_layer, traj_occlusions_features in agent_layers_features.items():
                                print(f"            Compute saliencies for layer: {intermediate_layer}")
                                if intermediate_layer in ["state_encoder"]:
                                    if agent_variant.__contains__("gwt_bu_td") or \
                                       agent_variant.__contains__("gru2") or \
                                       agent_variant.__contains__("gwt_bu") or \
                                       agent_variant.__contains__("gwt_td"):
                                        # Do nothing because we are using custom GRUCell instead of SS baselines'
                                        # The latter actually returns two vectors of shape [B, H], [1, B, H] for compatiblity reasons
                                        pass
                                    else:
                                        traj_occlusions_features = traj_occlusions_features[0]

                                # For plot and data caching
                                layer_dirname = f"{agent_variant_dirname}/{intermediate_layer}"

                                n_input_locs = traj_occlusions_features.shape[1] - 1 # How mnay occlusion "centers"
                                traj_layer_saliencies = []

                                for t in range(traj_length):
                                    no_occ_features = traj_occlusions_features[t][0]
                                    occ_saliencies = []

                                    for i in range(1, 1+n_input_locs):
                                        i_occ_saliency = (no_occ_features - traj_occlusions_features[t][i]).norm(2)
                                        occ_saliencies.append(i_occ_saliency.item())
                                    traj_layer_saliencies.append(occ_saliencies)
                                
                                saliencies = traj_layer_saliencies = np.array(traj_layer_saliencies)

                                # Store for later usage:
                                ALL_SALIENCIES[catname][scene][traj_idx][occ_target][occ_type][agent_variant][intermediate_layer] = \
                                    traj_layer_saliencies
                                
                                # Individual cache file
                                os.makedirs(layer_dirname, exist_ok=True)
                                layercache_filename = f"{layer_dirname}/saliencies.bz2"

                                if not os.path.exists(layercache_filename):
                                    with open(layercache_filename, "wb") as f:
                                        cpkl.dump(traj_layer_saliencies, f)
                                
                                # Generate plot, save to disk
                                # Scale the saliencies over all the values in the given trajectory
                                traj_saliency_min, traj_saliency_max = saliencies.min(), saliencies.max()
                                
                                scaled_saliencies = scale_values(saliencies, traj_saliency_min, traj_saliency_max, 0.0, 1.0)
                                n_input_locs = B - 1

                                fig, axes = plt.subplots(1, traj_length, figsize=(traj_length * 6, 6 + 1.5))

                                # Get min and max of spectrograms values across a single trajectory
                                traj_spect = np.array([occluded_obs_dict_list[t]["spectrogram"][0].cpu().numpy() for t in range(traj_length)])
                                spectr_min, spectr_max = traj_spect.min(), traj_spect.max()

                                for t, (obs_dict, obs_occ_mask) in enumerate(zip(occluded_obs_dict_list, occlusion_mask_dict_list)):
                                    # Plot the default image
                                    # NOTE: the [0] indexing in [OCC_TARGET][0] is because we store the non-occluded image at that positon
                                    # All the other indices 1: have different occlusion variants
                                    if occ_target == "rgb":
                                        img_data = obs_dict[occ_target][0].cpu().numpy().astype(np.uint8)
                                        axes[t].imshow(img_data)
                                    elif occ_target == "spectrogram":
                                        img_data = th.cat([obs_dict[occ_target][0][:, :, i] for i in range(2)], dim=1).cpu().numpy()
                                        img_data = scale_values(img_data, old_min=spectr_min, old_max=spectr_max, new_min=0., new_max=1.)
                                        axes[t].imshow(img_data, cmap="gray")
                                    else:
                                        raise NotImplementedError(f"Unsupported input data for saliency map: {occ_target}")
                                    
                                    axes[t].tick_params(axis="both", which="both", bottom=False, top=False, left=False, labelleft=False, labelbottom=False)
                                    axes[t].set_title(f"t = {t}", fontsize=20)

                                    # Generated mask based on saliencies
                                    saliency_mask = np.zeros_like(obs_dict[occ_target][0].cpu().numpy(), dtype=np.float32)
                                    
                                    for i in range(1, n_input_locs+1):
                                        i_occ_mask = obs_occ_mask[occ_target][i].cpu().numpy()
                                        saliency_mask = np.where(i_occ_mask, scaled_saliencies[t][i-1], saliency_mask)
                                    
                                    if occ_target == "rgb":
                                        saliency_mask = saliency_mask[:, :, 0]
                                    elif occ_target == "spectrogram":
                                        saliency_mask = np.concatenate([saliency_mask[:, :, i] for i in range(2)], 1)
                                    
                                    axes[t].imshow(saliency_mask, alpha=ALPHA, cmap="jet", vmin=0.0, vmax=1.0)
                                    # print(t)
                                    # print(obs_dict["rgb"].shape)
                                    # print(f"            {obs_occ_mask['rgb'].shape}")
                                
                                fig.suptitle(f"Saliency map for cat: {catname} | Scene: {scene} | Occ. target: {occ_target} | Occ. type: {occ_type} | Traj: {traj_idx} | {agent_variant} | {intermediate_layer} ", fontsize=28)
                                fig.tight_layout()

                                # Save to file
                                salmap_plot_dirname = f"{SALMAP_CACHE_DIRNAME}/saliencies_plots/{GEN_TIMESTAMP}" + \
                                    f"{catname}/{scene}/{traj_idx}/{occ_target}/{occ_type}/{agent_variant}/{intermediate_layer}"
                                os.makedirs(salmap_plot_dirname, exist_ok=True)
                                salmap_plot_filename = f"{salmap_plot_dirname}/salmap_plot.png"
                                fig.savefig(salmap_plot_filename)
                            
                            # Cleanup
                            del agent_layers_features, agent_rnn_state
                            th.cuda.empty_cache() # Clear up GPU cache that gets quite high

                        # Clean up:
                        del occluded_obs_dict_list, occlusion_mask_dict_list
                        th.cuda.empty_cache() # Clear up GPU cache that gets quite high
                
                del obs_dict_list, done_list, target_scene_idx_list, target_category_idx_list
                th.cuda.empty_cache() # Clear up GPU cache that gets quite high
                
                # break # traj_idx ; debug mostly

            # break # scene ; debug mostly

        # break # category ; debug mostly

    # Cache to disk in case the file is not detected
    if not os.path.exists(allsall_filename):
        with open(allsall_filename, "wb") as f:
            cpkl.dump(ALL_SALIENCIES, f)
