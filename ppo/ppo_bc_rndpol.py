import os
import cv2
import time
import random
import numpy as np
import torch as th

from collections import deque

import tools
from configurator import generate_args, get_arg_dict
from th_logger import TBXLogger as TBLogger

# Env deps: Soundspaces and Habitat
from ss_baselines.av_nav.config import get_config
from ss_baselines.savi.config.default import get_config as get_savi_config
from ss_baselines.common.env_utils import construct_envs
from ss_baselines.common.environments import get_env_class
from ss_baselines.common.utils import images_to_video_with_audio

# Custom ActorCritic agent for PPO
from models import Random_Actor

# Helpers
def dict_without_keys(d, keys_to_ignore):
	return {x: d[x] for x in d if x not in keys_to_ignore}

# This variant will fill each batch trajectory using cat.ed episode data
# There is no empty step in this batch
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

@th.inference_mode()
def eval_agent(args, eval_envs, agent, device, tblogger, env_config, current_step, n_eval_envs=1, n_episodes=5, save_videos=True, is_SAVi=False):
	obs = eval_envs.reset()
	done = [False for _ in range(n_eval_envs)]
	done_th = th.Tensor(done).to(device)

	masks = 1. - done_th[:, None]
	gw = th.zeros((n_eval_envs, args.gw_size), device=device)
	modality_features = {
		"audio": gw.new_zeros([n_eval_envs, args.hidden_size]),
		"visual": gw.new_zeros([n_eval_envs, args.hidden_size])
	}
	
	n_finished_episodes = 0

	window_episode_stats = {}
	# Variables to track episodic return, videos, and other SS relevant stats
	current_episode_return = th.zeros(n_eval_envs, 1).to(device)
	current_episode_entropy = th.zeros(n_eval_envs, 1).to(device)
	current_episode_length = th.zeros(n_eval_envs, 1).to(device)

	eval_video_data_env_0 = {
		"rgb": [], "depth": [],
		"audiogoal": [], "spectrogram": [],
		"actions": [], "top_down_map": []
	}

	while n_finished_episodes < n_episodes:
		# NOTE: the following line tensorize and also appends data to the rollout storage
		obs_th = tensorize_obs_dict(obs, device)

		# Sample action
		action, _, _, _, \
		entropies, gw, modality_features = \
			agent.act(obs_th, gw, masks=masks, 
			modality_features=modality_features, deterministic=True)
		# outputs = eval_envs.step([a[0].item() for a in action])
		outputs = eval_envs.step(action)

		obs, reward, done, info = [list(x) for x in zip(*outputs)]
		reward_th = th.Tensor(np.array(reward, dtype=np.float32)).to(device)

		## This is done to update the masks that will be used to track 
		# episodic return. Anyway to make this more efficient ?
		done_th = th.Tensor(done).to(device)
		masks = 1. - done_th[:, None]
		
		# Tracking episode return
		current_episode_return += reward_th[:, None]
		# current_episode_entropy += entropies[:, None]
		current_episode_length += 1

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
					"actions": [], "top_down_map": [],
					"entropies": []
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
				if "entropy" not in list(window_episode_stats.keys()):
					window_episode_stats["entropy"] = []
				window_episode_stats["entropy"].extend([(current_episode_entropy / current_episode_length)[i].item() for i in env_done_idxs])
			
			# Track total number of episodes
			n_finished_episodes += len(env_done_idxs)
		
		# Resets the episodic return tracker
		current_episode_return *= masks
		current_episode_entropy *= masks
		current_episode_length *= masks

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
		get_arg_dict("config-path", str, "env_configs/savi/savi_ss1_rgb_spectro.yaml"),

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
		get_arg_dict("agent-type", str, "random", metatype="choice",
			choices=["random"]),
		get_arg_dict("gru-type", str, "layernorm", metatype="choice",
						choices=["default", "layernorm"]),
		get_arg_dict("hidden-size", int, 512), # Size of the visual / audio features

		## BC related hyper parameters
		get_arg_dict("batch-chunk-length", int, 0), # For gradient accumulation
		get_arg_dict("dataset-ce-weights", bool, False, metatype="bool"), # If True, will read CEL weights based on action dist. from the 'dataset_statistics.bz2' file.
		get_arg_dict("ce-weights", float, None, metatype="list"), # Weights for the Cross Entropy loss

		## GW Agent with custom attention, recurrent encoder and null inputs
		get_arg_dict("gw-size", int, 512), # Dim of the GW vector
		get_arg_dict("recenc-use-gw", bool, True, metatype="bool"), # Use GW at Recur. Enc. level
		get_arg_dict("recenc-gw-detach", bool, True, metatype="bool"), # When using GW at Recurrent Encoder level, whether to detach the grads or not
		get_arg_dict("gw-use-null", bool, True, metatype="bool"), # Use Null at CrossAtt level
		get_arg_dict("gw-cross-heads", int, 1), # num_heads of the CrossAttn

		# Eval protocol
		get_arg_dict("eval", bool, True, metatype="bool"),
		get_arg_dict("eval-every", int, int(1.5e4)), # Every X frames || steps sampled
		get_arg_dict("eval-n-episodes", int, 5),

		# Logging params
		# NOTE: Video logging expensive
		get_arg_dict("save-videos", bool, False, metatype="bool"),
		get_arg_dict("save-model", bool, True, metatype="bool"),
		get_arg_dict("save-model-every", int, int(5e5)), # Every X frames || steps sampled
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
	## In case video saving is enabled, make sure there is also the rgb videos
	if args.save_videos:
		# For RGB video sensors
		if "RGB_SENSOR" not in env_config.SENSORS:
			env_config.SENSORS.append("RGB_SENSOR")
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

	if args.agent_type == "random":
		agent = Random_Actor(
			single_observation_space,
			single_action_space
		)
	else:
		raise NotImplementedError(f"Unsupported agent-type:{args.agent_type}")

	# Dataset loading
	
	# Experiment logger
	tblogger = TBLogger(exp_name=args.exp_name, args=args)
	print(f"# Logdir: {tblogger.logdir}")
	should_log_training_stats = tools.Every(args.log_training_stats_every)
	should_eval = tools.Every(args.eval_every)

	# Info logging
	print(" ### INFO: Agent summary and structure ###")
	print(agent)
	print("")

	# Adding independent components for SSL

	# Training start
	start_time = time.time()
	# num_updates = args.total_steps // args.batch_size # Total number of updates that will take place in this experiment
	n_updates = 0 # Progressively tracks the number of network updats
	# Log the number of parameters of the model
	tblogger.log_stats({
		"n_params": agent.get_n_params()
	}, 0, "info")

	# NOTE: 10 * 150 as step to match the training rate of an RL Agent, 
	# irrespective of which batch size / batch length is used
	# Ideally, both RL and BC variants should be trained with the same number of steps,
	# with batch of data as similar as possible.
	# In some BC experiments we would use a single expisode as batch trajectory,
	# while RL can have multiple episode concated together to fill up one batch trajectory.
	for global_step in range(0, args.total_steps + (args.num_envs * args.num_steps), args.num_envs * args.num_steps):
		
		if n_updates > 0 and should_log_training_stats(n_updates):
			print(f"Step {global_step} / {args.total_steps}")

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


	# Clean up
	tblogger.close()
	if args.eval:
		envs.close()

if __name__ =="__main__":
	main()