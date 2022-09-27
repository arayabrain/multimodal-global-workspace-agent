import random
import numpy as np

import os
import uuid
import datetime
import pickle as pkl
import compress_pickle as cpkl

import torch
import torch as th
import torch.nn as nn
from torch import Tensor

# General config related
from configurator import get_arg_dict, generate_args

# Env config related
from ss_baselines.av_nav.config import get_config
from ss_baselines.savi.config.default import get_config as get_savi_config
from ss_baselines.common.env_utils import construct_envs
from ss_baselines.common.environments import get_env_class

# Helpers
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

def save_episode_to_dataset(ep_data_dict, dataset_path):
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    epid = str(uuid.uuid4().hex)
    ep_length = ep_data_dict["ep_length"]

    # TODO: consider the downside of compression if this has to be used for 
    ep_data_filename = f"{timestamp}-{epid}-{ep_length}.bz2"
    ep_data_fullpath = os.path.join(dataset_path, ep_data_filename)
    with open(ep_data_fullpath, "wb") as f:
        cpkl.dump(ep_data_dict, f)

def main():
    # region: Generating additional hyparams
    CUSTOM_ARGS = [
        # General hyper parameters
        get_arg_dict("seed", int, 111),
        get_arg_dict("total-steps", int, 1_000_000),

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
        get_arg_dict("pgwt-ca-prev-latents", bool, True, metatype="bool"), # if True, passes the prev latent to CA as KV input data

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
    env_config.NUM_PROCESSES = args.num_envs # Corresponds to number of envs, makes script startup faster for debugs
    # env_config.USE_SYNC_VECENV = True
    # env_config.USE_VECENV = False
    # env_config.CONTINUOUS = args.env_continuous
    ## In caes video saving is enabled, make sure there is also the rgb videos
    env_config.freeze()
    # print(env_config)

    # Environment instantiation
    envs = construct_envs(env_config, get_env_class(env_config.ENV_NAME))
    single_observation_space = envs.observation_spaces[0]
    single_action_space = envs.action_spaces[0]

    single_observation_space, single_action_space

    # TODO seeding for reproducibility ? Make sure that we can control the generated episode trajs ?

    # Loading pretrained agent
    import models
    from models import ActorCritic, Perceiver_GWT_GWWM_ActorCritic
    dev = th.device("cuda:0")
    ## PPO GRU agent working with RGB observations
    ppo_gru_agent_state_dict = th.load("/home/rousslan/random/rl/exp-logs/ss-hab/ppo_av_nav__ss1_rgb_spectro_seed_111__2022_09_06_13_59_13_931240.musashi/models/ppo_agent.994501.ckpt.pth")
    ppo_gru_agent = ActorCritic(single_observation_space, single_action_space, args.hidden_size, extra_rgb=False,
        analysis_layers=models.GRU_ACTOR_CRITIC_DEFAULT_ANALYSIS_LAYER_NAMES).to(dev)
    ppo_gru_agent.eval()
    ppo_gru_agent.load_state_dict(ppo_gru_agent_state_dict)

    # Parallelized dataset collection
    DATASET_TOTAL_STEPS= int(1e6) # int(1e6)
    DATASET_DIR_PATH = f"ppo_gru_dset_2022_09_21__{DATASET_TOTAL_STEPS}_STEPS"
    MINIMUM_EP_LENGTH = 20

    # Placeholders for episode data
    obs_list, \
    reward_list, \
    done_list, \
    info_list, \
    action_list = \
        [[] for _ in range(args.num_envs)], \
        [[] for _ in range(args.num_envs)], \
        [[] for _ in range(args.num_envs)], \
        [[] for _ in range(args.num_envs)], \
        [[] for _ in range(args.num_envs)]
    
    rnn_hidden_state = th.zeros((1, args.num_envs, args.hidden_size), device=dev)

    obs, done = envs.reset(), [False for _ in range(args.num_envs)]
    done_th = th.Tensor(done).to(dev)
    masks = 1. - done_th[:, None]
    
    step = 0
    ep_returns = []

    while step < DATASET_TOTAL_STEPS:
        
        obs_th = tensorize_obs_dict(obs, dev)
        done_th = th.Tensor(done).to(dev)
        masks = 1. - done_th[:, None]

        with th.no_grad():
            action, _, _, _, value, rnn_hidden_state = \
                ppo_gru_agent.act(obs_th, rnn_hidden_state, masks=masks)
        
        outputs = envs.step([a[0].item() for a in action])
        next_obs, reward, next_done, info = [list(x) for x in zip(*outputs)]

        # Recorder episode trajectoreis
        for i in range(args.num_envs):
            obs_list[i].append(obs[i])
            done_list[i].append(done[i])
            action_list[i].append(action.cpu().numpy()[i])
            reward_list[i].append(reward[i])
            info_list[i].append(info[i])
            
        # When one or more episode end is detected, write to disk and resk that spot
        if np.sum(next_done) >= 1.:
            finished_envs_idxs = np.where(next_done)[0]

            for i in finished_envs_idxs:
                if not info_list[i][-1]["success"] == 1:
                    continue
                
                ep_length = len(obs_list[i])
                ep_returns = []
                ep_success = []
                ep_norm_dist_to_goal = []

                # Some episodes are too short, we would like to avoid those
                if ep_length >= MINIMUM_EP_LENGTH:
                    # Pre-process the obs_dict to have lists of "rgb", "depth", etc..
                    obs_dict = {k: [] for k in obs_list[i][0].keys()}

                    for t in range(ep_length):
                        [obs_dict[k].append(v) for k, v in obs_list[i][t].items()]
                    
                    ep_data_dict = {
                        "obs_list": obs_dict,
                        "action_list": action_list[i],
                        "done_list": done_list[i],
                        "reward_list": reward_list[i],
                        "info_list": info_list[i], # This can arguably be skipped ?,
                        "ep_length": ep_length
                    }

                    ep_returns.append(np.sum(reward_list[i]))
                    # TODO: double check why the last info list is not the final one
                    ep_success.append(info_list[i][-1]["success"])
                    ep_norm_dist_to_goal.append(info_list[i][-1]["normalized_distance_to_goal"])

                    # Saves to disk
                    save_episode_to_dataset(ep_data_dict, DATASET_DIR_PATH)

                    step += ep_length
                    

                # Reset the data placeholders
                obs_list[i], action_list[i], done_list[i], reward_list[i], info_list[i] = \
                    [], [], [], [], []

            print(f"Collected {step} / {DATASET_TOTAL_STEPS}; Avg return: {np.mean(ep_returns):0.2f}; Avg Suc.: {np.mean(ep_success)}; Avg: Norm Dist Goal: {np.mean(ep_norm_dist_to_goal)}")

        # Prepare for the next step
        obs = next_obs
        done = next_done

        # Stop collection ASAP
        if step >= DATASET_TOTAL_STEPS:
            break

if __name__ == "__main__":
    main()
