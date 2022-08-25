# Custom PPO implementation with Soundspaces 2.0
# Borrows from 
## - CleanRL's PPO LSTM: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_lstm.py
## - SoundSpaces AudioNav Baselines: https://github.com/facebookresearch/sound-spaces/tree/main/ss_baselines/av_nav

import os
import time
import random
import einops
import numpy as np
import torch as th
import torch.nn as nn
from collections import deque, defaultdict
from torchinfo import summary

import tools
from configurator import generate_args, get_arg_dict
from th_logger import TBXLogger as TBLogger

# Env deps: Soundspaces and Habitat
from habitat.datasets import make_dataset
from ss_baselines.av_nav.config import get_config
from ss_baselines.common.env_utils import construct_envs
from ss_baselines.common.environments import get_env_class
from ss_baselines.common.utils import images_to_video_with_audio

# Custom ActorCritic agent for PPO
from models import ActorCritic, ActorCritic_DeepEthologyVirtualRodent, \
    Perceiver_GWT_ActorCritic, PerceiverIO_GWT_ActorCritic

# Helpers
# Tensorize current observation, store to rollout data
def tensorize_obs_dict(obs, device, observations=None, rollout_step=None):
    obs_th = {}
    for obs_field, _ in obs[0].items():
        v_th = th.Tensor(np.array([step_obs[obs_field] for step_obs in obs], dtype=np.float32)).to(device)
        obs_th[obs_field] = v_th
        # Special case when doing the rollout, also stores the 
        if observations is not None:
            observations[obs_field][rollout_step] = v_th
    
    return obs_th

def main():
    # region: Generating additional hyparams
    CUSTOM_ARGS = [
        # General hyepr parameters
        get_arg_dict("seed", int, 111),
        get_arg_dict("total-steps", int, 10_000_000),
        
        # SS env config
        get_arg_dict("config-path", str, "env_configs/audiogoal_depth.yaml"),

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
        ## Agent network params
        get_arg_dict("agent-type", str, "ss-default", metatype="choice",
            choices=["ss-default", "perceiver-gwt", "perceiverio-gwt", "deep-etho"]),
        get_arg_dict("hidden-size", int, 512), # Size of the visual / audio features and RNN hidden states 
        ## Perceiver / PerceiverIO params: TODO: num_latnets, latent_dim, etc...
        get_arg_dict("pgwt-latent-type", str, "randn", metatype="choice",
            choices=["randn", "zeros"]), # Depth of the Perceiver
        get_arg_dict("pgwt-latent-learned", bool, True, metatype="bool"),
        get_arg_dict("pgwt-depth", int, 4), # Depth of the Perceiver
        get_arg_dict("pgwt-num-latents", int, 32),
        get_arg_dict("pgwt-latent-dim", int, 32),
        get_arg_dict("pgwt-cross-heads", int, 1),
        get_arg_dict("pgwt-latent-heads", int, 8),
        get_arg_dict("pgwt-cross-dim-head", int, 64),
        get_arg_dict("pgwt-latent-dim-head", int, 64),
        get_arg_dict("pgwt-weight-tie-layers", bool, False, metatype="bool"),

        # Logging params
        # NOTE: While supported, video logging is expensive because the RGB generation in the
        # envs hogs a lot of GPU, especially with multiple envs 
        get_arg_dict("save-videos", bool, False, metatype="bool"),
        get_arg_dict("save-model", bool, True, metatype="bool"),
        get_arg_dict("log-sampling-stats-every", int, int(1.5e4)), # Every X frames || steps sampled
        get_arg_dict("log-training-stats-every", int, int(10)), # Every X model update
        get_arg_dict("logdir-prefix", str, "./logs/"), # Overrides the default one
    ]
    args = generate_args(CUSTOM_ARGS)

    # Load environment config
    env_config = get_config(config_paths=args.config_path)

    # Additional PPO overrides
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    # endregion: Generating additional hyparams

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

    # env_config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP") # Note: can we add audio sensory info fields here too ?
    # NOTE: when evaluating, if we use the same dataset split as the training mode, then evaluation will not be fair to the baseline
    # env_config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
    
    # Overriding some envs parametes from the .yaml env config
    env_config.defrost()
    env_config.NUM_PROCESSES = args.num_envs # Corresponds to number of envs, makes script startup faster for debugs
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
    
    # TODO: make the ActorCritic components parameterizable through comand line ?
    if args.agent_type == "ss-default":
        agent = ActorCritic(single_observation_space, single_action_space,
            args.hidden_size, extra_rgb=agent_extra_rgb).to(device)
    elif args.agent_type == "perceiver-gwt":
        agent = Perceiver_GWT_ActorCritic(single_observation_space, single_action_space,
            args, extra_rgb=agent_extra_rgb).to(device)
    elif args.agent_type == "perceiverio-gwt":
        agent = PerceiverIO_GWT_ActorCritic(single_observation_space, single_action_space,
            args, extra_rgb=agent_extra_rgb).to(device)
    elif args.agent_type == "deep-etho":
        raise NotImplementedError(f"Unsupported agent-type:{args.agent_type}")
        # TODO: support for storing the rnn_hidden_statse, so that the policy 
        # that takes in the 'core_modules' 's rnn hidden output can also work.
        agent = ActorCritic_DeepEthologyVirtualRodent(single_observation_space,
                single_action_space, 512).to(device)
    else:
        raise NotImplementedError(f"Unsupported agent-type:{args.agent_type}")

    optimizer = th.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    
    # Info logging
    summary(agent)
    print("")
    print(agent)
    print("")
    
    # Rollout storage setup # TODO: make this systematic for a
    observations = {}
    for obs_sensor in single_observation_space.keys():
        observations[obs_sensor] = th.zeros((args.num_steps, args.num_envs) + 
            single_observation_space[obs_sensor].shape, device=device)
    
    actions = th.zeros((args.num_steps, args.num_envs), dtype=th.int64, device=device)
    logprobs = th.zeros((args.num_steps, args.num_envs), device=device)
    rewards = th.zeros((args.num_steps, args.num_envs), device=device)
    dones = th.zeros((args.num_steps, args.num_envs), device=device)
    values = th.zeros((args.num_steps, args.num_envs), device=device)

    # Training start
    start_time = time.time()
    num_updates = args.total_steps // args.batch_size # Total number of updates that will take place in this experiment.

    obs = envs.reset()
    
    done = [False for _ in range(args.num_envs)]
    done_th = th.Tensor(done).to(device)
    masks = 1. - done_th[:, None]
    if args.agent_type == "ss-default":
        rnn_hidden_state = th.zeros((1, args.num_envs, args.hidden_size), device=device)
    elif args.agent_type in ["perceiver-gwt", "perceiverio-gwt"]:
        rnn_hidden_state = einops.repeat(agent.state_encoder.latents.clone(), 'n d -> b n d', b = args.num_envs)
    elif args.agent_type == "deep-etho":
        rnn_hidden_state = th.zeros((1, args.num_envs, args.hidden_size * 2), device=device)
    else:
        raise NotImplementedError(f"Unsupported agent-type:{args.agent_type}")

    # Variables to track episodic return, videos, and other SS relevant stats
    current_episode_return = th.zeros(envs.num_envs, 1).to(device)
    # window_episode_stats = defaultdict(
    #     lambda: deque(maxlen=env_config.RL.PPO.reward_window_size))
    window_episode_stats = {}
    # NOTE: by enabling RGB frame generation, it is possible that the env sampling
    # gets slower
    train_video_data_env_0 = {
        "rgb": [], "depth": [], 
        "audiogoal": [], "top_down_map": [],
        "spectrogram": []
    }
    
    n_episodes = 0
    n_updates = 0 # Count how many updates so far. Not to confuse with "num_updatesy"

    for global_step in range(1, args.total_steps+1, args.num_steps * args.num_envs):
        # Copy the rnn_hidden_state at the start of the rollout.
        # This will be used to recompute the rnn_hidden_states when computiong the new action logprobs
        init_rnn_state = rnn_hidden_state.clone()

        for rollout_step in range(args.num_steps):
        
            # NOTE: the following line tensorize and also appends data to the rollout storage
            obs_th = tensorize_obs_dict(obs, device, observations, rollout_step)
            dones[rollout_step] = done_th

            # Sample action
            with th.no_grad():
                action, action_logprobs, _, value, rnn_hidden_state = \
                    agent.act(obs_th, rnn_hidden_state, masks=masks)
                values[rollout_step] = value.flatten()
            actions[rollout_step] = action.squeeze(-1) # actions: [T, B] but action: [B, 1]
            logprobs[rollout_step] = action_logprobs.sum(-1)

            outputs = envs.step([a[0].item() for a in action])
            obs, reward, done, info = [list(x) for x in zip(*outputs)]
            reward_th = th.Tensor(np.array(reward, dtype=np.float32)).to(device)
            rewards[rollout_step] = reward_th
            
            ## This is done to update the masks that will be used to track episodic return. Anyway to make this more efficient ?
            done_th = th.Tensor(done).to(device)
            masks = 1. - done_th[:, None]

            # Tracking episode return
            # TODO: keep this on GPU for more efficiency ? We log less than we update, so ...
            current_episode_return += reward_th[:, None]

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
                        if k in ["distance_to_goal", "normalized_distance_to_goal", "success", "spl", "softspl", "na", "sna"]:
                            if k not in list(window_episode_stats.keys()):
                                window_episode_stats[k] = deque(maxlen=env_config.RL.PPO.reward_window_size)
                            
                            # Append the metric of interest to the queue
                            window_episode_stats[k].append(v)
                    
                    # Add episodic return too
                    if "episodic_return" not in list(window_episode_stats.keys()):
                        window_episode_stats["episodic_return"] = deque(maxlen=env_config.RL.PPO.reward_window_size)
                    env_done_ep_returns = current_episode_return[env_done_idxs].flatten().tolist()
                    # Append the episodic returns for the env that are dones to the window stats list
                    window_episode_stats["episodic_return"].extend(env_done_ep_returns)
                
                # Track total number of episodes
                n_episodes += len(env_done_idxs)

            if should_log_sampling_stats(global_step) and (True in done):
                info_stats = {
                    "global_step": global_step,
                    "duration": time.time() - start_time,
                    "fps": tblogger.track_duration("fps", global_step),
                    "n_updates": n_updates,
                    "env_step_duration": tblogger.track_duration("fps_inv", global_step, inverse=True),
                    "model_updates_per_sec": tblogger.track_duration("model_updates",
                        n_updates),
                    "model_update_step_duration": tblogger.track_duration("model_updates_inv",
                        n_updates, inverse=True)
                }
                tblogger.log_stats(info_stats, global_step, "info")
  
                episode_stats = {
                    # Episodic return at the current step, averaged over the envs that are done
                    "current_episode_return": (current_episode_return.sum() / done_th.sum()).item(),
                    "total_episodes": n_episodes,
                    # Episodic return average over the window statistic, as well as other SS relevant metrics: 50 episodes by default
                    **{k: np.mean(v) for k, v in window_episode_stats.items()}
                }
                tblogger.log_stats(episode_stats, global_step, "metrics")

                # Save the model
                if args.save_model:
                    model_save_dir = tblogger.get_models_savedir()
                    model_save_name = f"ppo_agent.{global_step}.ckpt.pth"
                    model_save_fullpath = os.path.join(model_save_dir, model_save_name)

                    th.save(agent.state_dict(), model_save_fullpath)

            # Resets the episodic return tracker
            current_episode_return *= masks

            if args.save_videos:
                # Accumulate data for video + audio rendering
                train_video_data_env_0["rgb"].append(obs[0]["rgb"])
                train_video_data_env_0["audiogoal"].append(obs[0]["audiogoal"])
                # train_video_data_env_0["depth"].append(obs[0]["depth"])
                # train_video_data_env_0["spectrogram"].append(obs[0]["spectrogram"])
                train_video_data_env_0["top_down_map"].append(info[0]["top_down_map"]) # info[i]["top_down_map"] is a dict itself
                
                # Log video as soon as the ep in the first env is done
                if done[0]:
                    if should_log_video(global_step):
                        # TODO: video logging: fuse with top_down_map and other stats,
                        # then save to disk, tensorboard, wandb, etc...
                        # Video plotting is limited to the first environment
                        base_video_name = "train_video_0"
                        video_name = f"{base_video_name}_gstep_{global_step}"
                        video_fullpath = os.path.join(tblogger.get_videos_savedir(), f"{video_name}.mp4")
                        
                        # Saves to disk
                        images_to_video_with_audio(
                            images=train_video_data_env_0["rgb"],
                            audios=train_video_data_env_0["audiogoal"],
                            output_dir=tblogger.get_videos_savedir(),
                            video_name=video_name,
                            sr=env_config.TASK_CONFIG.SIMULATOR.AUDIO.RIR_SAMPLING_RATE, # 16000 for mp3d dataset
                            fps=env_config.TASK_CONFIG.SIMULATOR.VIEW_CHANGE_FPS
                        )
                        
                        # Save to tensorboard
                        # tblogger.log_video("train_video_0_rgb",
                        #     np.array([train_video_data_env_0["rgb"]]).transpose(0, 1, 4, 2, 3), # From [1, THWC] to [1,TCHW]
                        #     global_step, fps=env_config.TASK_CONFIG.SIMULATOR.VIEW_CHANGE_FPS)
                        
                        # Upload to wandb
                        tblogger.log_wandb_video_audio(base_video_name, video_fullpath)

                    # Reset the placeholder for the video
                    # If it is too early to log, then the episode data is trashed
                    train_video_data_env_0 = {
                        "rgb": [], "depth": [], 
                        "audiogoal": [], "top_down_map": [],
                        "spectrogram": []
                    }

        # Prepare for PPO update phase
        ## Bootstrap value if not done
        with th.no_grad():
            obs_th = tensorize_obs_dict(obs, device)
            done_th = th.Tensor(done).to(device)
            value = agent.get_value(obs_th, rnn_hidden_state, masks=1.-done_th[:, None]).flatten()
            # By default, use GAE
            advantages = th.zeros_like(rewards)
            lastgaelam = 0.
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - done_th
                    nextvalues = value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Form batch data of dim [ NUM_ENVS * NUM_STEPS, ...]
        b_observations = {}
        for k, v in observations.items():
            b_observations[k] = th.flatten(v, start_dim=0, end_dim=1)
        # b_observations = observations.reshape()
        b_logprobs = logprobs.reshape(-1) # From [B, T] -> [B * T]
        b_actions = actions.reshape(-1) # From [B, T] -> [B * T]
        b_dones = dones.reshape(-1) # From [B, T] -> [B * T]
        b_advantages = advantages.reshape(-1) # From [B, T] -> [B * T]
        b_returns = returns.reshape(-1) # From [B, T] -> [B * T]
        b_values = values.reshape(-1) # From [B, T] -> [B * T]
        
        # PPO Update Phase: actor and critic network updates
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []

        for _ in range(args.update_epochs):
            np.random.shuffle(envinds)
            # Why minibatch ? Some empirical evidence that using smaller batch around 32 or 64
            # are generally better. Also, LeCun.
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                # Make a minibatch of observation dict
                mb_observations = {k: v[mb_inds] for k, v in b_observations.items()}

                # NOTE: should the RNN hit states be reused when recomputiong ?
                if args.agent_type in ["ss-default"]:
                    b_init_rnn_state = init_rnn_state[:, mbenvinds]
                elif args.agent_type in ["perceiver-gwt", "perceiverio-gwt"]:
                    b_init_rnn_state = init_rnn_state[mbenvinds, :, :]
                else:
                    raise NotImplementedError(f"Unsupported agent-type:{args.agent_type}")
                
                _, newlogprob, entropy, newvalue, _ = \
                    agent.act(mb_observations, b_init_rnn_state,
                        masks=1-b_dones[mb_inds], actions=b_actions[mb_inds])

                newlogprob = newlogprob.sum(-1) # From [B * T, 1] -> [B * T]
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with th.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * th.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = th.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + th.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = th.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            n_updates += 1
            
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if n_updates > 0 and should_log_training_stats(n_updates):
            train_stats = {
                "value_loss": v_loss.item(),
                "policy_loss": pg_loss.item(),
                "entropy": entropy_loss.item(),
                "old_approx_kl": old_approx_kl.item(),
                "approx_kl": approx_kl.item(),
                "clipfrac": np.mean(clipfracs),
                "explained_variance": explained_var
            }
            tblogger.log_stats(train_stats, global_step, prefix="train")
        
    # Clean up
    envs.close()
    tblogger.close()

if __name__ == "__main__":
    main()
