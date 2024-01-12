import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import os
import cv2
import apex
import copy
import random
import numpy as np
import matplotlib as mpl
import compress_pickle as cpkl
from collections import deque

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
    get_arg_dict("config-path", str, "env_configs/savi/savi_ss1_rgb_spectro.yaml"),

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
        choices=["ss-default", "perceiver-gwt-gwwm"]),
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

## Override default seed
env_config.SEED = env_config.TASK_CONFIG.SEED = env_config.TASK_CONFIG.SIMULATOR.SEED = args.seed

env_config.TASK_CONFIG.SIMULATOR.USE_RENDERED_OBSERVATIONS = False
# For smoother video, set CONTINUOUS_VIEW_CHANGE to True, and get the additional frames in obs_dict["intermediate"]
env_config.TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE = False

env_config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = 256
env_config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = 256
env_config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH = 256
env_config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT = 256

# env_config.NUM_PROCESSES = 1 # Corresponds to number of envs, makes script startup faster for debugs
# env_config.USE_SYNC_VECENV = True
# env_config.USE_VECENV = False
# env_config.CONTINUOUS = args.env_continuous
## In caes video saving is enabled, make sure there is also the rgb videos
env_config.freeze()
# print(env_config)

# Environment instantiation
envs = construct_envs(env_config, get_env_class(env_config.ENV_NAME))
# Dummy environment spaces

# TODO: add dyanmicallly set single_observation_space so that RGB and RGBD based variants
# can be evaluated at thet same time
# from gym import spaces
# single_action_space = spaces.Discrete(4)
# single_observation_space = spaces.Dict({
#     "rgb": spaces.Box(shape=[128,128,3], low=0, high=255, dtype=np.uint8),
#     # "depth": spaces.Box(shape=[128,128,1], low=0, high=255, dtype=np.uint8),
#     "audiogoal": spaces.Box(shape=[2,16000], low=-3.4028235e+38, high=3.4028235e+38, dtype=np.float32),
#     "spectrogram": spaces.Box(shape=[65,26,2], low=-3.4028235e+38, high=3.4028235e+38, dtype=np.float32)
# })

single_observation_space = envs.observation_spaces[0]
single_action_space = envs.action_spaces[0]
single_observation_space, single_action_space
# Override the observation space for "rgb" and "depth" from (256,256,C) to (128,128,C)
from gym import spaces
single_observation_space["rgb"] = spaces.Box(shape=[128,128,3], low=0, high=255, dtype=np.uint8)

# Load the agent models
# TODO seeding for reproducibility ? Make sure that we can control the generated episode trajs ?

# Loading pretrained agent
import models
from models import ActorCritic, Perceiver_GWT_GWWM_ActorCritic

MODEL_VARIANTS_TO_STATEDICT_PATH = {

    # TODO: rename the random baseline to show SAVi or AvNav ?
    # Random GRU Baseline
    # "ppo_gru__random": {
    #     "pretty_name": "GRU Random",
    #     "state_dict_path": ""
    # },
    # Random PGWT Baseline
    # "ppo_pgwt__random": {
    #     "pretty_name": "TransRNN Random",
    #     "state_dict_path": ""
    # },

    # SAVi BC variants; trained using RGB + Spectrogram to 10M steps steps
    "ppo_bc__rgb_spectro__gru__SAVi": {
        "pretty_name": "[SAVi BC] PPO GRU | RGB Spectro",
        "state_dict_path": "/home/rousslan/random/rl/exp-logs/ss-hab-bc/"
            "ppo_bc__savi_ss1_rgb_spectro__gru_seed_222__2023_06_17_21_24_12_718867.musashi"
            "/models/ppo_agent.9990001.ckpt.pth"
    },
    "ppo_bc__rgb_spectro__pgwt__SAVi": {
        "pretty_name": "[SAVi BC] PPO TransRNN | RGB Spectro",
        "state_dict_path": "/home/rousslan/random/rl/exp-logs/ss-hab-bc/"
            "ppo_bc__savi_ss1_rgb__spectro__pgwt__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats_seed_222__2023_06_17_21_24_10_884437.musashi"
            "/models/ppo_agent.9990001.ckpt.pth"
    },
}

dev = th.device("cuda") # NOTE / TODO: using GPU to be more efficient ?

# 'variant named' indexed 'torch agent'
MODEL_VARIANTS_TO_AGENTMODEL = {}

for k, v in MODEL_VARIANTS_TO_STATEDICT_PATH.items():
    args_copy = copy.copy(args)
    # Override args depending on the model in use
    if k.__contains__("gru"):
        agent = ActorCritic(single_observation_space, single_action_space, args.hidden_size, extra_rgb=False,
            analysis_layers=models.GRU_ACTOR_CRITIC_DEFAULT_ANALYSIS_LAYER_NAMES)
    elif k.__contains__("pgwt"):
        agent = Perceiver_GWT_GWWM_ActorCritic(single_observation_space, single_action_space, args, extra_rgb=False,
            analysis_layers=models.PGWT_GWWM_ACTOR_CRITIC_DEFAULT_ANALYSIS_LAYER_NAMES + ["state_encoder.ca.mha"])

    agent.eval()
    # Load the model weights
    # TODO: add map location device to use CPU only ?
    if v["state_dict_path"] != "":
        agent_state_dict = th.load(v["state_dict_path"], map_location=dev)
        agent.load_state_dict(agent_state_dict)
    
    agent = agent.to(dev)
    
    MODEL_VARIANTS_TO_AGENTMODEL[k] = agent
# Tensorize current observation, store to rollout data
def tensorize_obs_dict(obs, device, observations=None, rollout_step=None):
    obs_th = {}
    for obs_field, _ in obs[0].items():
        v_th = th.Tensor(np.array([cv2.resize(step_obs[obs_field], dsize=(128, 128)) if obs_field in ["rgb", "depth"] else step_obs[obs_field] for step_obs in obs], dtype=np.float32)).to(device)
        # in SS1.0, the depth observations comes as [B, 128, 128, 1, 1], so fix that
        if obs_field == "depth": 
            if v_th.dim() == 5:
                v_th = v_th.squeeze(-1)
            elif v_th.dim() == 3:
                v_th = v_th[:, :, :, None]
        ## Specific to the cv2resize version:
        ## resize the observation to hopefully improve consistency between
        ## oracle dataset and evaluation environments

        obs_th[obs_field] = v_th
        # Special case when doing the rollout, also stores the 
        if observations is not None:
            observations[obs_field][rollout_step] = v_th
    
    return obs_th

@th.no_grad()
def eval_agent(args, eval_envs, agent, device, tblogger, env_config, 
               current_step, n_episodes=5, save_videos=True, is_SAVi=False,
               eval_ablations=None, agent_type=""):
    obs = eval_envs.reset()
    n_eval_envs = len(obs)
    done = [False for _ in range(n_eval_envs)]
    done_th = th.Tensor(done).to(device)
    prev_acts = th.zeros([n_eval_envs, 4], device=device)

    masks = 1. - done_th[:, None]
    # if args.agent_type == "ss-default":
    if agent_type.__contains__("gru"):
        rnn_hidden_state = th.zeros((1, n_eval_envs, args.hidden_size), device=device)
    # elif args.agent_type in ["perceiver-gwt-gwwm"]:
    elif agent_type.__contains__("pgwt"):
        rnn_hidden_state = agent.state_encoder.latents.clone().repeat(n_eval_envs, 1, 1)
    else:
        raise NotImplementedError(f"Unsupported agent-type:{args.agent_type}")
    
    n_finished_episodes = 0

    window_episode_stats = {}
    # Variables to track episodic return, videos, and other SS relevant stats
    current_episode_return = th.zeros(n_eval_envs, 1).to(device)

    eval_video_data_env_0 = {
        "rgb": [], "depth": [],
        "audiogoal": [], "spectrogram": [],
        "actions": [], "top_down_map": []
    }

    while n_finished_episodes < n_episodes:
        # NOTE: the following line tensorize and also appends data to the rollout storage
        obs_th = tensorize_obs_dict(obs, device)

        if eval_ablations is None:
            pass
        elif isinstance(eval_ablations, list) and len(eval_ablations) > 0:
            for abl_obs_field in eval_ablations:
                obs_th[abl_obs_field] *= 0.0
        
        # Sample action
        action, _, _, _, _, _, rnn_hidden_state = \
            agent.act(obs_th, rnn_hidden_state, masks=masks, deterministic=True) #, prev_actions=prev_acts if args.prev_actions else None)
        outputs = eval_envs.step([a[0].item() for a in action])
        obs, reward, done, info = [list(x) for x in zip(*outputs)]
        reward_th = th.Tensor(np.array(reward, dtype=np.float32)).to(device)

        ## This is done to update the masks that will be used to track 
        # episodic return. Anyway to make this more efficient ?
        done_th = th.Tensor(done).to(device)
        masks = 1. - done_th[:, None]
        prev_acts = F.one_hot(action[:, 0], 4) * masks
        
        # Tracking episode return
        # TODO: keep this on GPU for more efficiency ? We log less than we update, so ...
        current_episode_return += reward_th[:, None]

        if save_videos:
            # Accumulate data for video + audio rendering
            eval_video_data_env_0["rgb"].append(obs[0]["rgb"])
            eval_video_data_env_0["audiogoal"].append(obs[0]["audiogoal"])
            eval_video_data_env_0["actions"].append(action[0].item())

            if done[0]:
                base_video_name = "eval_video_0"
                video_name = f"{base_video_name}_gstep_{current_step}"
                video_fullpath = os.path.join(tblogger.get_videos_savedir(), f"{video_name}.mp4")

                try:
                    images_to_video_with_audio(
                        images=eval_video_data_env_0["rgb"],
                        audios=eval_video_data_env_0["audiogoal"],
                        output_dir=tblogger.get_videos_savedir(),
                        video_name=video_name,
                        sr=env_config.TASK_CONFIG.SIMULATOR.AUDIO.RIR_SAMPLING_RATE, # 16000 for mp3d dataset
                        fps=1 # env_config.TASK_CONFIG.SIMULATOR.VIEW_CHANGE_FPS # Default is 10 it seems
                    )

                    # Upload to wandb
                    tblogger.log_wandb_video_audio(base_video_name, video_fullpath)
                except Exception as e:
                    print("Exception while writing video: ", e)

                ## Additional stas
                # How many 0 (STOP) actions are performed
                actions_histogram = {k: 0 for k in range(4)}
                for a in eval_video_data_env_0["actions"]:
                    actions_histogram[a] += 1
                tblogger.log_histogram("actions", np.array(eval_video_data_env_0["actions"]),
                                        step=current_step, prefix="eval")
                
                # tblogger.log_stats({
                #     "last_act_0": 1 if eval_video_data_env_0["actions"][-1] == 0 else 0,
                # }, step=current_step, prefix="debug")

                eval_video_data_env_0 = {
                    "rgb": [], "depth": [],
                    "audiogoal": [], "spectrogram": [],
                    "actions": [], "top_down_map": []
                }

        if True in done: # At least one env has reached terminal state
            # Extract the "success" and other SS relevant metrics from the env that are 'done'
            env_done_idxs = np.where(done)[0].tolist() # Index of the envs that are done
            for env_done_i in env_done_idxs:
                env_info_dict = info[env_done_i] # The info of the env that is done
                # Expected content: ['distance_to_goal', 'normalized_distance_to_goal', 'success', 'spl', 'softspl', 'na', 'sna', 'top_down_map']
                # - na: num_action
                # - sna: success_weight_with_num_action
                for k, v in env_info_dict.items():
                    # Make sure that the info metric is of interest / loggable.
                    k_list = ["distance_to_goal", "normalized_distance_to_goal", "success", "spl", "softspl", "na", "sna"]
                    if is_SAVi:
                        k_list.append("sws") # SAVi metric: "Success When Silent" (???)
                    if k in k_list:
                        if k not in list(window_episode_stats.keys()):
                            window_episode_stats[k] = deque(maxlen=n_episodes)
                        
                        # Append the metric of interest to the queue
                        window_episode_stats[k].append(v)
                
                # Add episodic return too
                if "episodic_return" not in list(window_episode_stats.keys()):
                    window_episode_stats["episodic_return"] = deque(maxlen=n_episodes)
                env_done_ep_returns = current_episode_return[env_done_idxs].flatten().tolist()
                # Append the episodic returns for the env that are dones to the window stats list
                window_episode_stats["episodic_return"].extend(env_done_ep_returns)

                # Tracking the last actions of an episode
                if "last_actions" not in list(window_episode_stats.keys()):
                    window_episode_stats["last_actions"] = deque(maxlen=n_episodes)
                window_episode_stats["last_actions"].extend([action[i].item() for i in env_done_idxs])
            
            # Track total number of episodes
            n_finished_episodes += len(env_done_idxs)
        
        # Resets the episodic return tracker
        current_episode_return *= masks

    return window_episode_stats


EVAL_ABLATIONS = {
    "default": None,
    "no_rgb": ["rgb"],
    "no_spectro": ["spectrogram"],
    "no_rgb_spectro": ["rgb", "spectrogram"]
}
AGENT_ABLATIONS_EVAL_STATS = {k: {} for k in MODEL_VARIANTS_TO_AGENTMODEL.keys()}
for agent_variant, agent_model in MODEL_VARIANTS_TO_AGENTMODEL.items():
    print(f"Evaluating agent: {agent_variant}")
    for ablation_type, ablation_field_list in EVAL_ABLATIONS.items():
        print(f"  Ablation type: {ablation_type}")
        AGENT_ABLATIONS_EVAL_STATS[agent_variant][ablation_type] = \
            eval_agent(args, envs, agent_model, device=dev, tblogger=None, env_config=env_config, current_step=0,
                       n_episodes=args.eval_n_episodes, save_videos=False, is_SAVi=is_SAVi, eval_ablations=ablation_field_list, agent_type=agent_variant)
        # print(f"    {np.mean(AGENT_ABLATIONS_EVAL_STATS[agent_variant][ablation_type]['success'])}")

# Avg success rate per agent variant, then ablation type
for agent_variant, ablation_eval_results in AGENT_ABLATIONS_EVAL_STATS.items():
    print(f"Agent variant: {agent_variant}:")
    for ablation_type, eval_results in ablation_eval_results.items():
        print(f"  {ablation_type}")
        for k, v in eval_results.items():
            if k not in ["success"]:
                continue
            print(f"    {k}: {np.mean(v)}")

print("")
print("")
# Avg success rate per ablation type, then agent varaint
for ablation_type in EVAL_ABLATIONS.keys():
    print(f"Ablation type: {ablation_type}")
    for agent_variant, ablation_eval_results in AGENT_ABLATIONS_EVAL_STATS.items():
        print(f"  {agent_variant}: {np.mean(ablation_eval_results[ablation_type]['success'])}")

with open("agent_eval_results.pkl", "wb") as f:
    cpkl.dump(AGENT_ABLATIONS_EVAL_STATS, f)
