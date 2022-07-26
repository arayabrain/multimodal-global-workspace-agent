# Custom PPO implementation with Soundspaces 2.0
# Borrows from 
## - CleanRL's PPO LSTM: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_lstm.py
## - SoundSpaces AudioNav Baselines: https://github.com/facebookresearch/sound-spaces/tree/main/ss_baselines/av_nav

import time
import random
import numpy as np
import torch as th
import torch.nn as nn
from collections import deque, defaultdict

import tools
from configurator import generate_args, get_arg_dict
from th_logger import TBXLogger as TBLogger

# Env deps: Soundspaces and Habitat
from habitat.datasets import make_dataset
from ss_baselines.av_nav.config import get_config
from ss_baselines.common.environments import AudioNavRLEnv
from ss_baselines.common.env_utils import construct_envs
from ss_baselines.common.environments import get_env_class
from ss_baselines.common.rollout_storage import RolloutStorage

# Custom ActorCritic agent for PPO
from models import ActorCritic

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
    # Environment config
    # TODO: Override some of the config elements through arg parse ?
    env_config = get_config(
        config_paths="env_configs/audiogoal_rgb.yaml",
        # run_type="train"
    )

    # region: Generating additional hyparams
    CUSTOM_ARGS = [
        # General hyepr parameters
        get_arg_dict("seed", int, 111),
        get_arg_dict("total-steps", int, 10_000_000),
        
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
        get_arg_dict("hidden-size", int, 512), # Size of the visual / audio features and RNN hidden states 

        # Logging params
        get_arg_dict("save-videos", bool, True, metatype="bool"),
        get_arg_dict("save-model", bool, True, metatype="bool"),
        get_arg_dict("log-sampling-stats-every", int, int(1.5e4)), # Every X frames || steps sampled
        get_arg_dict("log-training-stats-every", int, int(10)), # Every X model update
        get_arg_dict("logdir-prefix", str, "./logs/"), # Overrides the default one
    ]
    args = generate_args(CUSTOM_ARGS)
    # Additional PPO overrides
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    # endregion: Generating additional hyparams

    # Experiment logger
    tblogger = TBLogger(exp_name=args.exp_name, args=args)
    print(f"# Logdir: {tblogger.logdir}")
    # should_eval = tools.Every(args.eval_every)
    should_log_sampling_stats = tools.Every(args.log_sampling_stats_every)
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
    env_config.freeze()
    # print(env_config)
    
    # Environment instantiation
    envs = construct_envs(env_config, get_env_class(env_config.ENV_NAME))
    single_observation_space = envs.observation_spaces[0]
    single_action_space = envs.action_spaces[0]
    
    # TODO: make the ActorCritic components parameterizable through comand line ?
    agent = ActorCritic(single_observation_space, single_action_space, 512).to(device)
    optimizer = th.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    
    # Rollout storage setup
    observations = {
        "rgb": th.zeros((args.num_steps, args.num_envs) + single_observation_space["rgb"].shape, device=device),
        "spectrogram": th.zeros((args.num_steps, args.num_envs) + single_observation_space["spectrogram"].shape, device=device),
        "audiogoal": th.zeros((args.num_steps, args.num_envs) + single_observation_space["audiogoal"].shape, device=device)
    }
    actions = th.zeros((args.num_steps, args.num_envs), dtype=th.int64, device=device)
    logprobs = th.zeros((args.num_steps, args.num_envs), device=device)
    rewards = th.zeros((args.num_steps, args.num_envs), device=device)
    dones = th.zeros((args.num_steps, args.num_envs), device=device)
    values = th.zeros((args.num_steps, args.num_envs), device=device)

    # Variables to track episode reward
    current_episode_reward = th.zeros(envs.num_envs, 1)
    running_episode_stats = dict(
        count=th.zeros(envs.num_envs, 1),
        reward=th.zeros(envs.num_envs, 1),
    )
    latest_successes = deque([], env_config.RL.PPO.reward_window_size)
    window_episode_stats = defaultdict(
        lambda: deque(maxlen=env_config.RL.PPO.reward_window_size)
    )

    # Training start
    start_time = time.time()
    num_updates = args.total_steps // args.batch_size

    obs = envs.reset()
    done = [False for _ in range(args.num_envs)]
    init_hidden_state = th.zeros((1, args.num_envs, args.hidden_size), device=device)
    rnn_hidden_state = init_hidden_state.clone()

    for global_step in range(1, args.total_steps+1, args.num_steps * args.num_envs):

        for rollout_step in range(args.num_steps):
            obs_th = tensorize_obs_dict(obs, device, observations, rollout_step)
            done_th = th.Tensor(done).to(device)
            dones[rollout_step] = done_th
            masks = 1. - done_th[:, None]

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
            
            # Tracking episode return
            current_episode_reward += reward_th[:, None].to(current_episode_reward.device)
            running_episode_stats["reward"] += (1 - masks.to(current_episode_reward.device)) * current_episode_reward
            running_episode_stats["count"] += (1 - masks.to(current_episode_reward.device))

            if should_log_sampling_stats(global_step) and (True in done):
                # TODO: instead of True, make sure that there is one final episode we can 
                # log the video and other stats of
                done_th = done_th = th.Tensor(done).to(device) # Overrides the done for now

                info_stats = {
                    "global_step": global_step,
                    "duration": time.time() - start_time,
                    "fps": tblogger.track_duration("fps", global_step),
                    "env_step_duration": tblogger.track_duration("fps_inv", global_step, inverse=True),
                    "model_updates_per_sec": tblogger.track_duration("model_updates",
                        num_updates),
                    "model_update_step_duration": tblogger.track_duration("model_updates_inv",
                        num_updates, inverse=True)
                }
                tblogger.log_stats(info_stats, global_step, "info")

                # TODO: extract the success rate and other variables
                episode_stats = {
                    "episode_return": (running_episode_stats["reward"].sum() / done_th.sum()).item(),
                    "episode_count": running_episode_stats["count"].sum().item()
                }
                tblogger.log_stats(episode_stats, global_step, "metrics")

            current_episode_reward *= masks # Resets the episodic return tracker

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

                _, newlogprob, entropy, newvalue, _ = \
                    agent.act(
                        mb_observations, init_hidden_state[:, mbenvinds],
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
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            num_updates += 1
            
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if num_updates > 0 and should_log_training_stats(num_updates):
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
