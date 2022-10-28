# Custom PPO Behavior Cloning implementation with Soundspaces 2.0
# Borrows from 
## - CleanRL's PPO LSTM: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_lstm.py
## - SoundSpaces AudioNav Baselines: https://github.com/facebookresearch/sound-spaces/tree/main/ss_baselines/av_nav

import os
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
from models import ActorCritic, ActorCritic_DeepEthologyVirtualRodent, \
    Perceiver_GWT_GWWM_ActorCritic, Perceiver_GWT_AttGRU_ActorCritic

# Dataset utils
from torch.utils.data import IterableDataset, DataLoader
import compress_pickle as cpkl

## Shape of the dave ep_data_dict:, for reference
# obs_list dict:
# 	 rgb -> (94, 128, 128, 3)
# 	 audiogoal -> (94, 2, 16000)
# 	 spectrogram -> (94, 65, 26, 2)
# action_list -> (94, 1)
# done_list -> (94,)
# reward_list -> (94,)
# info_list -> (94,)
# ep_length -> 94

# This variant will sample one single (sub) seuqence of an episode as a trajectoyr
# and add zero paddign to the rest
class BCIterableDataset3(IterableDataset):
    def __init__(self, dataset_path, batch_length, seed=111):
        self.seed = seed
        self.batch_length = batch_length
        self.dataset_path = dataset_path

        # Read episode filenames in the dataset path
        self.ep_filenames = os.listdir(dataset_path)
        print(f"Initialized IterDset with {len(self.ep_filenames)} episodes.")
    
    def __iter__(self):
        batch_length = self.batch_length
        while True:
            # Sample one episode file
            idx = th.randint(len(self.ep_filenames), ())
            ep_filename = self.ep_filenames[idx]
            ep_filepath = os.path.join(self.dataset_path, ep_filename)
            with open(ep_filepath, "rb") as f:
                edd = cpkl.load(f)
            is_success = edd["info_list"][-1]["success"]
            last_action = [int(a[0]) for a in edd["action_list"]][-1]
            print(f"Sampled traj idx: {idx}; Length: {edd['ep_length']}; Success: {is_success}; Last act: {last_action}")
            if edd["ep_length"] < 30:
                continue # Skips short episodes
            
            edd_start = th.randint(0, edd["ep_length"]-20, ()).item() # Sample start of sub-squence for this episode
            edd_end = min(edd_start + batch_length, edd["ep_length"])
            subseq_len = edd_end - edd_start
            
            horizon = subseq_len

            obs_list = {
                k: np.zeros([batch_length, *np.shape(v)[1:]]) for k,v in edd["obs_list"].items()
            }
            action_list, reward_list, done_list, depad_mask_list = \
                np.zeros([batch_length, 1]), \
                np.zeros([batch_length, 1]), \
                np.zeros([batch_length, 1]), \
                np.zeros((batch_length, 1)).astype(np.bool8)

            for k, v in edd["obs_list"].items():
                obs_list[k][:horizon] = v[edd_start:edd_end]
            action_list[:horizon] = edd["action_list"][edd_start:edd_end]
            reward_list[:horizon] = np.array(edd["reward_list"][edd_start:edd_end])[:, None]
            done_list[:horizon] = np.array(edd["done_list"][edd_start:edd_end])[:, None]
            depad_mask_list[:horizon] = True

            yield obs_list, action_list, reward_list, done_list, depad_mask_list
    
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

# NOTE: DEBUG use
# DATASET_DIR_PATH = f"ppo_gru_dset_2022_09_21__750000_STEPS"

# dloader = make_dataloader3(DATASET_DIR_PATH, batch_size=1, batch_length=30)
# for _ in range(2):
#     obs_batch, action_batch, reward_batch, done_batch, depad_mask_list = next(dloader)

# Tensorize current observation, store to rollout data
def tensorize_obs_dict(obs, device, observations=None, rollout_step=None):
    obs_th = {}
    for obs_field, _ in obs[0].items():
        v_th = th.Tensor(np.array([step_obs[obs_field] for step_obs in obs], dtype=np.float32)).to(device)
        # in SS1.0, the dcepth observations comes as [B, 128, 128, 1, 1], so fix that
        if obs_field == "depth" and v_th.dim() == 5:
            v_th = v_th.squeeze(-1)
        obs_th[obs_field] = v_th
        # Special case when doing the rollout, also stores the 
        if observations is not None:
            observations[obs_field][rollout_step] = v_th
    
    return obs_th

@th.no_grad()
def eval_agent(args, eval_envs, agent, device, tblogger, env_config, current_step, n_episodes=5, save_videos=True,is_SAVi=False):

    n_eval_envs = 2 # TODO: maybe make this parameterizable ? and tie with environment creation part in main()
    obs = eval_envs.reset()
    done = [False for _ in range(n_eval_envs)]
    done_th = th.Tensor(done).to(device)
    prev_acts = th.zeros([n_eval_envs, 4], device=device)

    masks = 1. - done_th[:, None]
    if args.agent_type == "ss-default":
        rnn_hidden_state = th.zeros((1, n_eval_envs, args.hidden_size), device=device)
    elif args.agent_type in ["perceiver-gwt-gwwm", "perceiver-gwt-attgru"]:
        rnn_hidden_state = agent.state_encoder.latents.clone().repeat(n_eval_envs, 1, 1)
    elif args.agent_type == "deep-etho":
        rnn_hidden_state = th.zeros((1, n_eval_envs, args.hidden_size * 2), device=device)
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
        action, _, _, _, _, rnn_hidden_state = \
            agent.act(obs_th, rnn_hidden_state, masks=masks, deterministic=True, prev_actions=prev_acts if args.prev_actions else None)
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
                        fps=5 # env_config.TASK_CONFIG.SIMULATOR.VIEW_CHANGE_FPS # Default is 10 it seems
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
        get_arg_dict("dataset-path", str, "ppo_gru_dset_2022_09_21__750000_STEPS"),

        # SS env config
        get_arg_dict("config-path", str, "env_configs/audiogoal_rgb_nocont.yaml"),

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
            choices=["ss-default", "deep-etho",
                     "perceiver-gwt-gwwm", "perceiver-gwt-attgru"]),
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
    
    # Experiment logger
    tblogger = TBLogger(exp_name=args.exp_name, args=args)
    print(f"# Logdir: {tblogger.logdir}")
    should_log_training_stats = tools.Every(args.log_training_stats_every)
    should_eval = tools.Every(args.eval_every)

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
    # NOTE: using less environments for eval to save up system memory -> run more experiment at thte same time
    env_config.NUM_PROCESSES = 2 # Corresponds to number of envs, makes script startup faster for debugs
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
    env_config.freeze()
    # print(env_config)

    # Environment instantiation
    envs = construct_envs(env_config, get_env_class(env_config.ENV_NAME))
    single_observation_space = envs.observation_spaces[0]
    single_action_space = envs.action_spaces[0]

    # TODO: delete the envrionemtsn / find a more efficient method to do this
    
    # TODO: make the ActorCritic components parameterizable through comand line ?
    if args.agent_type == "ss-default":
        agent = ActorCritic(single_observation_space, single_action_space,
            args.hidden_size, extra_rgb=agent_extra_rgb, prev_actions=args.prev_actions).to(device)
    elif args.agent_type == "perceiver-gwt-gwwm":
        agent = Perceiver_GWT_GWWM_ActorCritic(single_observation_space, single_action_space,
            args, extra_rgb=agent_extra_rgb).to(device)
    elif args.agent_type == "perceiver-gwt-attgru":
        agent = Perceiver_GWT_AttGRU_ActorCritic(single_observation_space, single_action_space,
            args, extra_rgb=agent_extra_rgb).to(device)
    elif args.agent_type == "deep-etho":
        raise NotImplementedError(f"Unsupported agent-type:{args.agent_type}")
        # TODO: support for storing the rnn_hidden_statse, so that the policy 
        # that takes in the 'core_modules' 's rnn hidden output can also work.
        agent = ActorCritic_DeepEthologyVirtualRodent(single_observation_space,
                single_action_space, 512).to(device)
    else:
        raise NotImplementedError(f"Unsupported agent-type:{args.agent_type}")

    if not args.cpu and th.cuda.is_available():
        # TODO: GPU only. But what if we still want to use the default pytorch one ?
        optimizer = apex.optimizers.FusedAdam(agent.parameters(), lr=args.lr, eps=1e-5, weight_decay=args.optim_wd)
    else:
        optimizer = th.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5, weight_decay=args.optim_wd)

    ce_weights = args.ce_weights
    if ce_weights is not None:
        # TODO: assert it has same shape as action space.
        ce_weights = th.Tensor(args.ce_weights).to(device)
    
    optimizer.zero_grad()

    # Dataset loading
    dloader = make_dataloader3(args.dataset_path, batch_size=args.num_envs,
                              batch_length=args.num_steps, seed=args.seed)

    # Info logging
    summary(agent)
    print("")
    print(agent)
    print("")

    # Training start
    start_time = time.time()
    num_updates = args.total_steps // args.batch_size # Total number of updates that will take place in this experiment
    n_updates = 0 # Progressively tracks the number of network updats

    # NOTE: 10 * 150 as step to match the training rate of an RL Agent, irrespective of which batch size / batch length is used
    for global_step in range(1, args.total_steps + 1, 10 * 150):
        # Load batch data
        obs_list, action_list, _, done_list, depad_mask_list = \
            [ {k: th.Tensor(v).float().to(device) for k,v in b.items()} if isinstance(b, dict) else 
               b.float().to(device) for b in next(dloader)] # NOTE this will not suport "audiogoal" waveform audio, only rgb / depth / spectrogram
        
        # NOTE: RGB are normalized in the VisualCNN module
        # PPO networks expect input of shape T,B, ... so doing the permutation here
        for k, v in obs_list.items():
            if k in ["rgb", "spectrogram", "depth"]:
                obs_list[k] = v.permute(1, 0, 2, 3, 4) # BTCHW -> TBCHW
            elif k in ["audiogoal"]:
                obs_list[k] = v.permute(1, 0, 2, 3) # BTCL -> TBC
            else:
                pass
        
        action_list = action_list.permute(1, 0, 2)
        done_list = done_list.permute(1, 0, 2)
        depad_mask_list = depad_mask_list.permute(1, 0, 2)

        prev_actions_list = th.zeros_like(action_list)
        prev_actions_list[1:] = action_list[:-1]
        prev_actions_list = F.one_hot(prev_actions_list.long()[:, :, 0], num_classes=4).float()
        prev_actions_list[0] = prev_actions_list[0] * 0.

        # PPO Update Phase: actor and critic network updates
        # assert args.num_envs % args.num_minibatches == 0
        # envsperbatch = args.num_envs // args.num_minibatches
        # envinds = np.arange(args.num_envs)
        # flatinds = np.arange(args.batch_size).reshape(args.num_envs, args.num_steps)

        for _ in range(args.update_epochs):
            # np.random.shuffle(envinds)
            # TODO / MISSING: mini-batch updates support like in ppo_av_nav.py
            assert args.num_envs % args.batch_chunk_length == 0, \
                f"num-envs (batch-size) of {args.num_envs} should be integer divisible by {args.batch_chunk_length}"
            
            # For gradient accumulation over large batches
            n_bchunks = args.num_envs // args.batch_chunk_length
            
            # Reset optimizer for each chunk over the "trajectory length" axis
            optimizer.zero_grad()

            # Placeholder for tracking the actions of the agent
            batch_traj_agent_actions = th.zeros_like(action_list).long()

            for bchnk_idx in range(n_bchunks):
                # This will be used to recompute the rnn_hidden_states when computiong the new action logprobs
                if args.agent_type == "ss-default":
                    rnn_hidden_state = th.zeros((1, args.batch_chunk_length, args.hidden_size), device=device)
                elif args.agent_type in ["perceiver-gwt-gwwm", "perceiver-gwt-attgru"]:
                    rnn_hidden_state = agent.state_encoder.latents.repeat(args.batch_chunk_length, 1, 1)
                else:
                    raise NotImplementedError(f"Unsupported agent-type:{args.agent_type}")

                b_chnk_start = bchnk_idx * args.batch_chunk_length
                b_chnk_end = (bchnk_idx + 1) * args.batch_chunk_length

                # NOTE: v.shape[-3:] only valid for "rgb", "depth", and "spectrogram"
                obs_chunk_list = {}
                for k,v in obs_list.items():
                    if k in ["rgb", "depth", "spectrogram"]:
                        reshaped_v = v[:, b_chnk_start:b_chnk_end].reshape(-1, *v.shape[-3:])
                    elif k in ["audiogoal"]:
                        reshaped_v = v[:, b_chnk_start:b_chnk_end].reshape(-1, *v.shape[-2:])
                    else:
                        reshaped_v = v[:, b_chnk_start:b_chnk_end].reshape(-1, *v.shape[-1:])
                    
                    obs_chunk_list[k] = reshaped_v

                # obs_chunk_list = {k: v[:, b_chnk_start:b_chnk_end].reshape(-1, *v.shape[(-3 if k in ["rgb", "depth", "spectrogram"] else -2):])
                #                     for k, v in obs_list.items()}
                masks_chunk_list = 1. - done_list[:, b_chnk_start:b_chnk_end].reshape(-1, 1)
                action_target_chunk_list = action_list[:, b_chnk_start:b_chnk_end, 0].reshape(-1).long()
                prev_actions_chunk_list = prev_actions_list[:, b_chnk_start:b_chnk_end].reshape(-1, 4)

                # TODO: maybe detach the rnn_hidden_state between two chunks ?
                actions, action_probs, _, entropies, _, _ = \
                    agent.act(obs_chunk_list, rnn_hidden_state,
                        masks=masks_chunk_list, prev_actions=prev_actions_chunk_list)
                

                bc_loss = F.cross_entropy(action_probs, action_target_chunk_list,
                                          weight=ce_weights, reduction="none")
                bc_loss = th.masked_select(
                    bc_loss,
                    depad_mask_list[:, b_chnk_start:b_chnk_end, 0].reshape(-1).bool()
                ).mean()
                
                bc_loss /= n_bchunks # Normalize accumulated grads over batch axis
                
                # Entropy loss
                # TODO: consider making this decaying as training progresses
                entropy = th.masked_select(
                    entropies,
                    depad_mask_list[:, b_chnk_start:b_chnk_end, 0].reshape(-1).bool()
                ).mean()
                entropy /= n_bchunks # Normalize accumulated grads over batch axis

                # Backpropagate and accumulate gradients over the batch size axis
                (bc_loss - args.ent_coef * entropy).backward()

                # Temporarily save the batch chunk actions of the agent for statistics later
                batch_traj_agent_actions[:, b_chnk_start:b_chnk_end, :] = actions.detach().view(args.num_steps, args.batch_chunk_length, 1)

            grad_norms_preclip = agent.get_grad_norms()
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

            n_updates += 1

        if n_updates > 0 and should_log_training_stats(n_updates):
            print(f"Step {global_step} / {args.total_steps}")

            # TODO: add entropy logging
            train_stats = {
                "bc_loss": bc_loss.item() * n_bchunks,
                "entropy": entropy.item() * n_bchunks
            } # * n_bchunks undoes the scaling applied during grad accum
            
            tblogger.log_stats(train_stats, global_step, prefix="train")
        
            # TODO: Additional dbg stats; add if needed
            # debug_stats = {
            #     # Additional debug stats
            #     "b_state_feats_avg": b_state_feats.flatten(start_dim=1).mean().item(),
            #     "init_rnn_states_avg": init_rnn_state.flatten(start_dim=1).mean().item(),
            # }
            # if args.pgwt_mod_embed:
            #     debug_stats["mod_embed_avg"] = agent.state_encoder.modality_embeddings.mean().item()

            # Tracking stats about the actions distribution in the batches # TODO: fix typo
            batch_traj_final_step_idxs = (depad_mask_list.sum(dim=0)-1)[None, :, :].long().reshape(-1).tolist()
            batch_traj_final_step_mask = th.zeros_like(depad_mask_list).long()
            for bi, done_t in enumerate(batch_traj_final_step_idxs):
                batch_traj_final_step_mask[done_t, bi, 0] = 1

            batch_traj_final_actions = th.masked_select(
                action_list,
                batch_traj_final_step_mask.bool()
            )
            # How many '0' actions are sampled in a batch ?
            n_zero_batch_final_actions = len(th.where(batch_traj_final_actions == 0)[0])
            # Ratio of 0 to other actions in the batch
            n_zero_batch_final_actions_ratio = n_zero_batch_final_actions / depad_mask_list.sum().item()


            batch_traj_agent_final_actions = th.masked_select(
                batch_traj_agent_actions,
                batch_traj_final_step_mask.bool()
            )
            # How many '0' actions sampled by the agent given the observations ?
            n_zero_agent_final_actions = len(th.where(batch_traj_agent_final_actions == 0)[0])
            # Ratio of 0 to other actions in the batch
            n_zero_agent_final_actions_ratio = n_zero_agent_final_actions / depad_mask_list.sum().item()

            # Special: tracking the 'StOP' action stats across in the batch
            tblogger.log_stats({
                "n_zero_batch_final_actions": n_zero_batch_final_actions,
                "n_zero_batch_final_actions_ratio": n_zero_batch_final_actions_ratio,
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
                "batch_real_steps_ratio": depad_mask_list.sum().item() / np.prod(depad_mask_list.shape)
            }
            tblogger.log_stats(info_stats, global_step, "info")

        if args.eval and should_eval(global_step):
            eval_window_episode_stas = eval_agent(args, envs, agent,
                device=device, tblogger=tblogger,
                env_config=env_config, current_step=global_step,
                n_episodes=args.eval_n_episodes, save_videos=True,
                is_SAVi=is_SAVi)
            episode_stats = {k: np.mean(v) for k, v in eval_window_episode_stas.items()}
            episode_stats["last_actions_min"] = np.min(eval_window_episode_stas["last_actions"])
            tblogger.log_stats(episode_stats, global_step, "metrics")
        
            if args.save_model:
                model_save_dir = tblogger.get_models_savedir()
                model_save_name = f"ppo_agent.{global_step}.ckpt.pth"
                model_save_fullpath = os.path.join(model_save_dir, model_save_name)

                th.save(agent.state_dict(), model_save_fullpath)

    # Clean up
    tblogger.close() 
    envs.close()

if __name__ =="__main__":
    main()
