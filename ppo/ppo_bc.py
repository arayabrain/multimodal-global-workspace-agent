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

class BCIterableDataset(IterableDataset):
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
            # Sample epsiode data until there is enough for one trajectory
            # Hardcoded for now, make flexible later
            # Done later to recover the 
            obs_list = {
                "rgb": np.zeros([batch_length, 128, 128, 3]),
                "audiogoal": np.zeros([batch_length, 2, 16000]),
                "spectrogram": np.zeros([batch_length, 65, 26, 2])
            }
            action_list, reward_list, done_list = \
                np.zeros([batch_length, 1]), \
                np.zeros([batch_length, 1]), \
                np.zeros([batch_length, 1])
            
            ssf = 0 # Step affected so far
            while ssf < batch_length:
                idx = th.randint(len(self.ep_filenames), ())
                print(f"Sampled traj idx: {idx}")
                ep_filename = self.ep_filenames[idx]
                ep_filepath = os.path.join(DATASET_DIR_PATH, ep_filename)
                with open(ep_filepath, "rb") as f:
                    edd = cpkl.load(f)

                # Append the data to the bathc trjectory
                rs = batch_length - ssf # Reamining steps
                horizon = ssf + min(rs, edd["ep_length"])

                for k, v in edd["obs_list"].items():
                    obs_list[k][ssf:horizon] = v[:rs]
                action_list[ssf:horizon] = edd["action_list"][:rs]
                reward_list[ssf:horizon] = np.array(edd["reward_list"][:rs])[:, None]
                done_list[ssf:horizon] = np.array(edd["done_list"][:rs])[:, None]

                ssf += edd["ep_length"]

                if ssf >= self.batch_length:
                    break

            yield obs_list, action_list, reward_list, done_list
    
def make_dataloader(dataset_path, batch_size, batch_length, seed=111, num_workers=4):
    def worker_init_fn(worker_id):
        # worker_seed = th.initial_seed() % (2 ** 32)
        worker_seed = 133754134 + worker_id

        random.seed(worker_seed)
        np.random.seed(worker_seed)

    th_seed_gen = th.Generator()
    th_seed_gen.manual_seed(133754134 + seed)

    dloader = iter(
        DataLoader(
            BCIterableDataset(
                dataset_path=dataset_path, batch_length=batch_length),
                batch_size=batch_size, num_workers=num_workers,
                worker_init_fn=worker_init_fn, generator=th_seed_gen
            )
    )

    return dloader


def main():
    # region: Generating additional hyparams
    CUSTOM_ARGS = [
        # General hyper parameters
        get_arg_dict("seed", int, 111),
        get_arg_dict("total-steps", int, 1_000_000),
        
        # Behavior cloning gexperiment config
        get_arg_dict("dataset-path", str, "ppo_gru_dset_2022_09_21__750000_STEPS"),

        # SS env config
        get_arg_dict("config-path", str, "env_configs/audiogoal_depth_nocont.yaml"),

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
        get_arg_dict("ent-coef", float, 0.2), # Entropy loss coef; 0.2 in SS baselines
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

    # Experiment logger
    tblogger = TBLogger(exp_name=args.exp_name, args=args)
    print(f"# Logdir: {tblogger.logdir}")
    # should_eval = tools.Every(args.eval_every)
    should_log_sampling_stats = tools.Every(args.log_sampling_stats_every)
    should_log_video = tools.Every(args.log_sampling_stats_every)
    should_log_training_stats = tools.Every(args.log_training_stats_every)

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)
    th.backends.cudnn.deterministic = args.torch_deterministic
    # th.backends.cudnn.benchmark = args.cudnn_benchmark

    # Set device as GPU
    device = tools.get_device(args)

    # Overriding some envs parametes from the .yaml env config
    env_config.defrost()
    env_config.NUM_PROCESSES = args.num_envs # Corresponds to number of envs, makes script startup faster for debugs
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
            args.hidden_size, extra_rgb=agent_extra_rgb).to(device)
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

    optimizer = th.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5, weight_decay=args.optim_wd)

    # Dataset loading
    dloader = make_dataloader(args.dataset_path, batch_size=args.num_envs, batch_length=args.num_steps)

    # Info logging
    summary(agent)
    print("")
    print(agent)
    print("")

    # Training start
    start_time = time.time()
    num_updates = args.total_steps // args.batch_size # Total number of updates that will take place in this experiment

    if args.agent_type == "ss-default":
        rnn_hidden_state = th.zeros((1, args.num_envs, args.hidden_size), device=device)
    elif args.agent_type in ["perceiver-gwt-gwwm", "perceiver-gwt-attgru"]:
        rnn_hidden_state = agent.state_encoder.latents.repeat(args.num_envs, 1, 1)
    elif args.agent_type == "deep-etho":
        rnn_hidden_state = th.zeros((1, args.num_envs, args.hidden_size * 2), device=device)
    else:
        raise NotImplementedError(f"Unsupported agent-type:{args.agent_type}")

    for step in range(1, args.total_steps + 1):
        # Load batch data
        obs_list, action_list, reward_list, done_list = \
            [b.to(device) for b in next(dloader)]
        
        pass
