import os
import cv2
import time
import random
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import apex

from collections import deque
from torchinfo import summary

import tools
from configurator import generate_args, get_arg_dict
from th_logger import TBXLogger as TBLogger

# Env deps: Soundspaces and Habitat
from habitat.datasets import make_dataset
from ss_baselines.av_nav.config import get_config
from ss_baselines.savi.config.default import get_config as get_savi_config
from ss_baselines.common.env_utils import construct_envs
from ss_baselines.common.environments import get_env_class
from ss_baselines.common.utils import images_to_video_with_audio

# Custom ActorCritic agent for PPO
from models import ActorCritic, ActorCritic2, Perceiver_GWT_GWWM_ActorCritic
from models2 import GWTAgent, GWTAgent_BU

# Dataset utils
from torch.utils.data import IterableDataset, DataLoader
import compress_pickle as cpkl

# Helpers
def dict_without_keys(d, keys_to_ignore):
    return {x: d[x] for x in d if x not in keys_to_ignore}

# This variant will fill each batch trajectory using cat.ed episode data
# There is no empty step in this batch
class BCIterableDataset3(IterableDataset):
    def __init__(self, dataset_path, batch_length, seed=111):
        self.seed = seed
        self.batch_length = batch_length
        self.dataset_path = dataset_path

        # Read episode filenames in the dataset path
        self.ep_filenames = os.listdir(dataset_path)
        if "dataset_statistics.bz2" in self.ep_filenames:
            self.ep_filenames.remove("dataset_statistics.bz2")
        
        print(f"Initialized IterDset with {len(self.ep_filenames)} episodes.")
    
    def __iter__(self):
        batch_length = self.batch_length
        while True:
            # region: Sample episode data until there is enough to fill the hole batch traj
            obs_list = {
                "depth": np.zeros([batch_length, 128, 128]), # NOTE: data was recorded using (128, 128), but ideally we should have (128, 128, 1)
                "rgb": np.zeros([batch_length, 128, 128, 3]),
                "audiogoal": np.zeros([batch_length, 2, 16000]),
                "spectrogram": np.zeros([batch_length, 65, 26, 2]),
                "category": np.zeros([batch_length, 21]),
                "pointgoal_with_gps_compass": np.zeros([batch_length, 2]),
                "pose": np.zeros([batch_length, 4]),
            }

            action_list, reward_list, done_list = \
                np.zeros([batch_length, 1]), \
                np.zeros([batch_length, 1]), \
                np.zeros([batch_length, 1])
            ssf = 0 # Step affected so far
            while ssf < batch_length:
                idx = th.randint(len(self.ep_filenames), ())
                ep_filename = self.ep_filenames[idx]
                ep_filepath = os.path.join(self.dataset_path, ep_filename)
                with open(ep_filepath, "rb") as f:
                    edd = cpkl.load(f)
                # print(f"Sampled traj idx: {idx} ; Len: {edd['ep_length']}")
                
                # Append the data to the bathc trjectory
                rs = batch_length - ssf # Reamining steps
                horizon = ssf + min(rs, edd["ep_length"])
                for k, v in edd["obs_list"].items():
                    obs_list[k][ssf:horizon] = v[:rs]
                action_list[ssf:horizon] = np.array(edd["action_list"][:rs])[:, None]
                reward_list[ssf:horizon] = np.array(edd["reward_list"][:rs])[:, None]
                done_list[ssf:horizon] = np.array(edd["done_list"][:rs])[:, None]

                ssf += edd["ep_length"]

                if ssf >= self.batch_length:
                    break

            # Adjust shape of "depth" to be [T, H, W, 1] instead of [T, H, W]
            obs_list["depth"] = obs_list["depth"][:, :, :, None]
            
            yield obs_list, action_list, reward_list, done_list
            # endregion: Sample episode data until there is enough to fill the hole batch traj
    
def make_dataloader3(dataset_path, batch_size, batch_length, seed=111, num_workers=2):
    def worker_init_fn(worker_id):
        # worker_seed = th.initial_seed() % (2 ** 32)
        worker_seed = 133754134 + worker_id

        random.seed(worker_seed)
        np.random.seed(worker_seed)

    th_seed_gen = th.Generator()
    th_seed_gen.manual_seed(133754134 + seed)

    dloader = iter(
        DataLoader(
            BCIterableDataset3(
                dataset_path=dataset_path, batch_length=batch_length),
                batch_size=batch_size, num_workers=num_workers,
                worker_init_fn=worker_init_fn, generator=th_seed_gen
            )
    )

    return dloader

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
def eval_agent(args, eval_envs, agent, device, tblogger, env_config, current_step, n_eval_envs=1, n_episodes=5, save_videos=True, is_SAVi=False):
    obs = eval_envs.reset()
    done = [False for _ in range(n_eval_envs)]
    done_th = th.Tensor(done).to(device)
    prev_acts = th.zeros([n_eval_envs, 4], device=device)

    masks = 1. - done_th[:, None]
    if args.agent_type in ["ss-default", "custom-gru", "custom-gwt", "custom-gwt-bu"]:
        rnn_hidden_state = th.zeros((1, n_eval_envs, args.hidden_size), device=device)
    elif args.agent_type in ["perceiver-gwt-gwwm"]:
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

        # Sample action
        action, _, _, _, _, _, rnn_hidden_state, _ = \
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

def main():
    # region: Generating additional hyparams
    CUSTOM_ARGS = [
        # General hyper parameters
        get_arg_dict("seed", int, 111),
        get_arg_dict("total-steps", int, 1_000_000),
        
        # Behavior cloning gexperiment config
        get_arg_dict("dataset-path", str, "SAVI_Oracle_Dataset_v0"),

        # SS env config
        get_arg_dict("config-path", str, "env_configs/savi/savi_ss1.yaml"),

        # PPO Hyper parameters
        get_arg_dict("num-envs", int, 10), # Number of parallel envs. 10 by default
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
            choices=["ss-default", "perceiver-gwt-gwwm",
                      "custom-gru",
                      "custom-gwt", "custom-gwt-bu"]),
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
        get_arg_dict("pgwt-ca-prev-latents", bool, False, metatype="bool"), # if True, passes the prev latent to CA as KV input data

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
        ### Further parameterization of the SSL vision reconstruction task
        get_arg_dict("ssl-rec-rgb-detach", bool, True, metatype="bool"), # When doing SSL rec-rgb, detach the grads. from decoder to latents

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
    # endregion: Generating additional hyparams

    # Load environment config
    is_SAVi = str.__contains__(args.config_path, "savi")
    if is_SAVi:
        env_config = get_savi_config(config_paths=args.config_path)
    else:
        env_config = get_config(config_paths=args.config_path)

    # Additional PPO overrides
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    # Gradient accumulation support
    if args.batch_chunk_length == 0:
        args.batch_chunk_length = args.num_envs

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)
    th.backends.cudnn.deterministic = args.torch_deterministic
    # th.backends.cudnn.benchmark = args.cudnn_benchmark

    # Set device as GPU
    device = tools.get_device(args) if (not args.cpu and th.cuda.is_available()) else th.device("cpu")

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

    # NOTE: using less environments for eval to save up system memory -> run more experiment at the same time
    env_config.NUM_PROCESSES = 1 # Corresponds to number of envs, makes script startup faster for debugs
    # env_config.CONTINUOUS = args.env_continuous
    ## In caes video saving is enabled, make sure there is also the rgb videos
    agent_extra_rgb = False
    if args.save_videos:
        # For RGB video sensors
        if "RGB_SENSOR" not in env_config.SENSORS:
            env_config.SENSORS.append("RGB_SENSOR")
            # Indicates to the agent that RGB obs should not be used as observational inputs
            agent_extra_rgb = True
        # For Waveform to generate audio over the videos
        if "AUDIOGOAL_SENSOR" not in env_config.TASK_CONFIG.TASK.SENSORS:
            env_config.TASK_CONFIG.TASK.SENSORS.append("AUDIOGOAL_SENSOR")
    # Add support for TOP_DOWN_MAP
    # NOTE: it seems to induce "'DummySimulator' object has no attribute 'pathfinder'" error
    # If top down map really needed, probably have to run the env without pre-rendered observations ?
    # env_config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")

    env_config.freeze()

    # Environment instantiation
    if args.eval:
        # In case there is need for eval, instantiate some environments
        envs = construct_envs(env_config, get_env_class(env_config.ENV_NAME))
        single_observation_space = envs.observation_spaces[0]
        single_action_space = envs.action_spaces[0]
    else:
        # Otherwise, just use dummy obs. and act. spaces for agent structure init
        from gym import spaces
        single_action_space = spaces.Discrete(4)
        single_observation_space = spaces.Dict({
            # "rgb": spaces.Box(shape=[128,128,3], low=0, high=255, dtype=np.uint8),
            # "depth": spaces.Box(shape=[128,128,1], low=0, high=255, dtype=np.uint8),
            "audiogoal": spaces.Box(shape=[2,16000], low=-3.4028235e+38, high=3.4028235e+38, dtype=np.float32),
            "spectrogram": spaces.Box(shape=[65,26,2], low=-3.4028235e+38, high=3.4028235e+38, dtype=np.float32),
            "pose": spaces.Box(shape=[4], low=-3.4028235e+38, high=3.4028235e+38, dtype=np.float32)
        })
    
    # Override the observation space for "rgb" and "depth" from (256,256,C) to (128,128,C)
    from gym import spaces
    if "RGB_SENSOR" in env_config.SENSORS:
        single_observation_space["rgb"] = spaces.Box(shape=[128,128,3], low=0, high=255, dtype=np.uint8)
    if "DEPTH_SENSOR" in env_config.SENSORS:
        single_observation_space["rgb"] = spaces.Box(shape=[128,128,1], low=0, high=255, dtype=np.uint8)

    # TODO: delete the envrionemtsn / find a more efficient method to do this

    # TODO: make the ActorCritic components parameterizable through comand line ?
    if args.agent_type == "ss-default":
        agent = ActorCritic(single_observation_space, single_action_space, args, extra_rgb=agent_extra_rgb).to(device)
    elif args.agent_type == "custom-gru":
        # TODO: add toggle for 'pose' usage for ablations later ?
        agent = ActorCritic2(single_observation_space, single_action_space, args, extra_rgb=agent_extra_rgb).to(device)
    elif args.agent_type == "custom-gwt":
        agent = GWTAgent(single_action_space, args).to(device)
    elif args.agent_type == "custom-gwt-bu":
        agent = GWTAgent_BU(single_action_space, args).to(device)
    elif args.agent_type == "perceiver-gwt-gwwm":
        agent = Perceiver_GWT_GWWM_ActorCritic(single_observation_space, single_action_space,
            args, extra_rgb=agent_extra_rgb).to(device)
    else:
        raise NotImplementedError(f"Unsupported agent-type:{args.agent_type}")

    if not args.cpu and th.cuda.is_available():
        # TODO: GPU only. But what if we still want to use the default pytorch one ?
        optimizer = apex.optimizers.FusedAdam(agent.parameters(), lr=args.lr, eps=1e-5, weight_decay=args.optim_wd)
    else:
        optimizer = th.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5, weight_decay=args.optim_wd)

    optimizer.zero_grad()

    # Dataset loading
    dloader = make_dataloader3(args.dataset_path, batch_size=args.num_envs,
                                batch_length=args.num_steps, seed=args.seed, num_workers=8)

    ## Compute action coefficient for CEL of BC
    dataset_stats_filepath = f"{args.dataset_path}/dataset_statistics.bz2"
    # Override dataset statistics if the file already exists
    if os.path.exists(dataset_stats_filepath):
        with open(dataset_stats_filepath, "rb") as f:
            dataset_statistics = cpkl.load(f)

    # Reads args.ce_weights if passed
    ce_weights = args.ce_weights

    # In case args.dataset_ce_weights is True,
    # override the args.ce_weigths manual setting
    if args.dataset_ce_weights:
        # TODO: make some assert on 1) the existence of the "dataset_statistics.bz2" file
        # and 2) that it contains the "action_cel_coefs" of proper dimension
        ce_weights = [dataset_statistics["action_cel_coefs"][a] for a in range(4)]
        print("### INFO: Loading CEL weights from dataset: ", ce_weights)
    
        # Override the CE weights in the args, more readable
        args.ce_weights = [round(cew, 2) for cew in ce_weights]

    # Otherwise, use manually specified CEL weights
    if ce_weights is not None:
        # TODO: assert it has same shape as action space.
        ce_weights = th.Tensor(ce_weights).to(device)
        print("### INFO: Manually set CEL weights from dataset: ", ce_weights)

    print(" ### INFO: CEL weights")
    print(args.ce_weights)
    print("")
    
    # Experiment logger
    tblogger = TBLogger(exp_name=args.exp_name, args=args)
    print(f"# Logdir: {tblogger.logdir}")
    should_log_training_stats = tools.Every(args.log_training_stats_every)
    should_eval = tools.Every(args.eval_every)

    # Info logging
    print(" ### INFO: Agent summary and structure ###")
    summary(agent)
    print("")
    print(agent)
    print("")

    # Adding independent components for SSL

    # Checking the dataset steps
    print(" ### INFO: Dataset statistics ###")
    from pprint import pprint
    pprint(dict_without_keys(dataset_statistics, ["episode_lengths",
        "cat_scene_filenames", "scene_cat_filenames", "scene_filenames"]))
    print("")

    # Training start
    start_time = time.time()
    num_updates = args.total_steps // args.batch_size # Total number of updates that will take place in this experiment
    n_updates = 0 # Progressively tracks the number of network updats

    # NOTE: 10 * 150 as step to match the training rate of an RL Agent, irrespective of which batch size / batch length is used
    # Ideally, both RL and BC variants should be trained with the same number of steps, with batch of data as similar as possible.
    # In some BC experiments we would use a single expisode as batch trajectory, while RL can have multiple episode cat.ed together
    # to fill up one batch trajectory.
    for global_step in range(1, args.total_steps + (args.num_envs * args.num_steps), args.num_envs * args.num_steps):
        # Load batch data
        obs_list, action_list, _, done_list = \
            [ {k: th.Tensor(v).float().to(device) for k,v in b.items()} if isinstance(b, dict) else 
                b.float().to(device) for b in next(dloader)]
        
        # NOTE: RGB are normalized in the VisualCNN module
        # PPO networks expect input of shape T,B, ... so doing the permutation first
        # then flatten over T x B dimensions. The RNN will reshape it as necessary
        for k, v in obs_list.items():
            if k in ["rgb", "spectrogram", "depth"]:
                obs_list[k] = v.permute(1, 0, 2, 3, 4) # BTCHW -> TBCHW
                obs_list[k] = obs_list[k].reshape(-1, *obs_list[k].shape[-3:])
            elif k in ["audiogoal"]:
                obs_list[k] = v.permute(1, 0, 2, 3) # BTCL -> TBCL
                obs_list[k] = obs_list[k].reshape(-1, *obs_list[k].shape[-2:])
            else:
                # TODO: handle other fields like "category", etc...
                pass
        
        action_list = action_list.permute(1, 0, 2)
        done_list = done_list.permute(1, 0, 2)
        mask_list = 1. - done_list
        
        prev_actions_list = th.zeros_like(action_list)
        prev_actions_list[1:] = action_list[:-1]
        prev_actions_list = F.one_hot(prev_actions_list.long()[:, :, 0], num_classes=4).float()
        prev_actions_list[0] = prev_actions_list[0] * 0.0

        # Finally, also flatten across T x B, let the RNN do the unflattening if needs be
        action_list = action_list.reshape(-1) # Because it is used for the target later
        done_list = done_list.reshape(-1, 1)
        mask_list = mask_list.reshape(-1, 1)
        prev_actions_list = prev_actions_list.reshape(-1, 1)

        # PPO Update Phase: actor and critic network updates
        for _ in range(args.update_epochs):
            # np.random.shuffle(envinds)
            # TODO / MISSING: mini-batch updates support like in ppo_av_nav.py
            assert args.num_envs % args.batch_chunk_length == 0, \
                f"num-envs (batch-size) of {args.num_envs} should be integer divisible by {args.batch_chunk_length}"
            
            # Reset optimizer for each chunk over the "trajectory length" axis
            optimizer.zero_grad()

            # This will be used to recompute the rnn_hidden_states when computiong the new action logprobs
            if args.agent_type in ["ss-default", "custom-gru", "custom-gwt", "custom-gwt-bu"]:
                rnn_hidden_state = th.zeros((1, args.batch_chunk_length, args.hidden_size), device=device)
            elif args.agent_type in ["perceiver-gwt-gwwm"]:
                rnn_hidden_state = agent.state_encoder.latents.repeat(args.batch_chunk_length, 1, 1)
            else:
                raise NotImplementedError(f"Unsupported agent-type:{args.agent_type}")

            # TODO: maybe detach the rnn_hidden_state between two chunks ?
            actions, _, _, action_logits, entropies, _, _, ssl_outputs = \
                agent.act(obs_list, rnn_hidden_state, masks=mask_list, ssl_tasks=args.ssl_tasks) #, prev_actions=prev_actions_list)


            bc_loss = F.cross_entropy(action_logits, action_list.long(),
                                        weight=ce_weights, reduction="mean")

            ssl_losses = {}
            full_ssl_loss = 0
            if args.ssl_tasks is not None:
                for i, ssl_task in enumerate(args.ssl_tasks):
                    ssl_task_coef = 1 if args.ssl_task_coefs is None else float(args.ssl_task_coefs[i])
                    if ssl_task in ["rec-rgb-ae", "rec-rgb-ae-2", "rec-rgb-ae-3", "rec-rgb-ae-4"
                                    "rec-rgb-vis-ae", "rec-rgb-vis-ae-3", "rec-rgb-vis-ae-4"]:
                        assert args.obs_center, f"SSL task rec-rgb expects having args.obs_center = True, which is not the case now."
                        rec_rgb_mean = ssl_outputs[ssl_task]
                        rec_rgb_dist = th.distributions.Independent(
                            th.distributions.Normal(rec_rgb_mean, 1), 3)
                        rec_rgb_loss = rec_rgb_dist.log_prob(obs_list["rgb"].permute(0, 3, 1, 2) / 255.0 - 0.5).neg().mean()
                        full_ssl_loss += ssl_task_coef * rec_rgb_loss
                        ssl_losses[ssl_task] = rec_rgb_loss
                    elif ssl_task in ["rec-rgb-vis-ae-mse"]:
                        rec_rgb_mean = ssl_outputs[ssl_task]
                        rec_rgb_loss = F.mse_loss(rec_rgb_mean, 
                                        obs_list["rgb"].permute(0, 3, 1, 2) / 255.0 - 0.5, reduction="none"
                                       ).mean(dim=0).sum()
                        full_ssl_loss += ssl_task_coef * rec_rgb_loss
                        ssl_losses[ssl_task] = rec_rgb_loss
                    else:
                        raise NotImplementedError(f"Unsupported SSL task: {ssl_task}")
                ssl_losses["full_ssl_loss"] = full_ssl_loss

            # Entropy loss
            # TODO: consider making this decaying as training progresses
            entropy = entropies.mean()

            # Backpropagate and accumulate gradients over the batch size axis
            (bc_loss - args.ent_coef * entropy + full_ssl_loss).backward()

            grad_norms_preclip = agent.get_grad_norms()
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

            n_updates += args.update_epochs

        if n_updates > 0 and should_log_training_stats(n_updates):
            print(f"Step {global_step} / {args.total_steps}")

            # TODO: add entropy logging
            train_stats = {
                "bc_loss": bc_loss.item(),
                "entropy": entropy.item()
            } # * n_bchunks undoes the scaling applied during grad accum
            
            tblogger.log_stats(train_stats, global_step, prefix="train")
            tblogger.log_stats({k: v.item() for k, v in ssl_losses.items()},
                                global_step, prefix="train/ssl")

            # TODO: Additional dbg stats; add if needed
            # debug_stats = {
            #     # Additional debug stats
            #     "b_state_feats_avg": b_state_feats.flatten(start_dim=1).mean().item(),
            #     "init_rnn_states_avg": init_rnn_state.flatten(start_dim=1).mean().item(),
            # }
            # if args.pgwt_mod_embed:
            #     debug_stats["mod_embed_avg"] = agent.state_encoder.modality_embeddings.mean().item()

            # TODO: Tracking stats about the actions distribution in the batches # TODO: fix typo
            T, B = args.num_steps, args.num_envs
            # How many '0' actions sampled by the agent given the observations ?
            n_zero_agent_final_actions = len(th.where(actions == 0)[0])
            # Ratio of 0 to other actions in the batch
            n_zero_agent_final_actions_ratio = n_zero_agent_final_actions / (B * T)
            # How many '0' actions in the batch trajs., as well as ratio relative to the latter.
            n_zero_batch_final_actions = len(th.where(action_list == 0)[0])
            n_zero_batch_final_actions_ratio = n_zero_batch_final_actions / (B * T)

            # Special: tracking the 'STOP' action stats across in the batch
            tblogger.log_stats({
                "n_zero_batch_final_actions": n_zero_batch_final_actions,
                "n_zero_batch_final_actions_ratio": n_zero_batch_final_actions_ratio ,
                "n_zero_agent_final_actions": n_zero_agent_final_actions,
                "n_zero_agent_final_actions_ratio": n_zero_agent_final_actions_ratio
            }, step=global_step, prefix="metrics")

            # Logging grad norms
            tblogger.log_stats(agent.get_grad_norms(), global_step, prefix="debug/grad_norms")
            tblogger.log_stats(grad_norms_preclip, global_step, prefix="debug/grad_norms_preclip")

            info_stats = {
                "global_step": global_step,
                "duration": time.time() - start_time,
                "fps": tblogger.track_duration("fps", global_step),
                "n_updates": n_updates,

                "env_step_duration": tblogger.track_duration("fps_inv", global_step, inverse=True),
                "model_updates_per_sec": tblogger.track_duration("model_updates",
                    n_updates),
                "model_update_step_duration": tblogger.track_duration("model_updates_inv",
                    n_updates, inverse=True),
                "batch_real_steps_ratio": 1.0
            }
            tblogger.log_stats(info_stats, global_step, "info")

        # TODO: remove / comment out after debugs
        # Same as in the args.eval if condition, but allows us
        # to check that plotting works without having to wait
        # for eval_envs instantiation when debugging
        # if args.ssl_tasks is not None and \
        #     isinstance(args.ssl_tasks, list) and \
        #     len(args.ssl_tasks):

        #     for ssl_task in args.ssl_tasks:
        #         # Vision based SSL tasks: reconstruction to gauge how good
        #         if ssl_task in ["rec-rgb-ae", "rec-rgb-ae-2", "rec-rgb-ae-3", "rec-rgb-ae-4",
        #                         "rec-rgb-vis-ae", "rec-rgb-vis-ae-3", "rec-rgb-vis-ae-4", "rec-rgb-vis-ae-mse"]:
        #             rec_rgb_mean = ssl_outputs[ssl_task]
        #             tmp_img_data = th.cat([
        #                 obs_list["rgb"][:3].permute(0, 3, 1, 2).int(),
        #                 ((rec_rgb_mean[:3].detach() + 0.5).clamp(0, 1) * 255).int()],
        #             dim=2)
        #             img_data = th.cat([i for i in tmp_img_data], dim=2).cpu().numpy().astype(np.uint8)
        #             # tblogger.log_image(ssl_task, img_data, global_step, prefix="ssl")
        #             # NOTE: log the RGB reconstruction under the same tag no matter the ssl_task,
        #             # works better with Wandb
        #             tblogger.log_image("rec-rgb", img_data, global_step, prefix="ssl")

        if args.eval and should_eval(global_step):
            eval_window_episode_stas = eval_agent(args, envs, agent,
                device=device, tblogger=tblogger,
                env_config=env_config, current_step=global_step,
                n_eval_envs=env_config.NUM_PROCESSES,
                n_episodes=args.eval_n_episodes, save_videos=args.save_videos,
                is_SAVi=is_SAVi)
            episode_stats = {k: np.mean(v) for k, v in eval_window_episode_stas.items()}
            episode_stats["last_actions_min"] = np.min(eval_window_episode_stas["last_actions"])
            tblogger.log_stats(episode_stats, global_step, "metrics")
        
            # SSL qualitative eval
            if args.ssl_tasks is not None and \
                isinstance(args.ssl_tasks, list) and \
                len(args.ssl_tasks):
                
                for ssl_task in args.ssl_tasks:
                    # Vision based SSL tasks: reconstruction to gauge how good
                    if ssl_task in ["rec-rgb-ae", "rec-rgb-ae-2", "rec-rgb-ae-3", "rec-rgb-ae-4",
                                    "rec-rgb-vis-ae", "rec-rgb-vis-ae-3", "rec-rgb-vis-ae-4", "rec-rgb-vis-ae-mse"]:
                        rec_rgb_mean = ssl_outputs[ssl_task]
                        tmp_img_data = th.cat([
                            obs_list["rgb"][:3].permute(0, 3, 1, 2).int(),
                            ((rec_rgb_mean[:3].detach() + 0.5).clamp(0, 1) * 255).int()],
                            dim=2
                        )
                        img_data = th.cat([i for i in tmp_img_data], dim=2).cpu().numpy().astype(np.uint8)
                        # tblogger.log_image(ssl_task, img_data, global_step, prefix="ssl")
                        # NOTE: log the RGB reconstruction under the same tag no matter the ssl_task,
                        # works better with Wandb
                        tblogger.log_image("rec-rgb", img_data, global_step, prefix="ssl")

            if args.save_model:
                model_save_dir = tblogger.get_models_savedir()
                model_save_name = f"ppo_agent.{global_step}.ckpt.pth"
                model_save_fullpath = os.path.join(model_save_dir, model_save_name)

                th.save(agent.state_dict(), model_save_fullpath)

    # Clean up
    tblogger.close()
    if args.eval:
        envs.close()

if __name__ =="__main__":
    main()
