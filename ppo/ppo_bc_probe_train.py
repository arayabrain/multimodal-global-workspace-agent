# This script expects to train probes on features learned 
# by one specific pre-trained model
# There is however support for multiple probing target
# namely "category" of the goal / object in the episode
# and the scene the episode takes place in.

# General config related
import os
import copy
import time
import random
import numpy as np
import compress_pickle as cpkl

# Custom imports
from configurator import get_arg_dict, generate_args
from th_logger import TBXLogger as TBLogger

# ML deps
import apex
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# Env config related
from ss_baselines.av_nav.config import get_config
from ss_baselines.savi.config.default import get_config as get_savi_config
from ss_baselines.common.env_utils import construct_envs
from ss_baselines.common.environments import get_env_class
from ss_baselines.common.utils import plot_top_down_map

# Dataset utils
from torch.utils.data import IterableDataset, DataLoader
import compress_pickle as cpkl

# Loading pretrained agent
import tools
import models
from models import ActorCritic, Perceiver_GWT_GWWM_ActorCritic

# Helpers
def dict_without_keys(d, keys_to_ignore):
    return {x: d[x] for x in d if x not in keys_to_ignore}

# region: Generating additional hyparams
CUSTOM_ARGS = [
    # General hyper parameters
    get_arg_dict("seed", int, 111),
    get_arg_dict("total-steps", int, 500_000), # By default, should be the number of steps in the dataset
    
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
    get_arg_dict("pgwt-ca-prev-latents", bool, False, metatype="bool"), # if True, passes the prev latent to CA as KV input data

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

## Compute action coefficient for CEL of BC
dataset_stats_filepath = f"{args.dataset_path}/dataset_statistics.bz2"
# Override dataset statistics if the file already exists
if os.path.exists(dataset_stats_filepath):
    with open(dataset_stats_filepath, "rb") as f:
        dataset_statistics = cpkl.load(f)

# Scene list for probe target label generation
SCENES = list(dataset_statistics["scene_counts"].keys())
CATEGORIES = list(dataset_statistics["category_counts"].keys())

N_SCENES = len(SCENES)
N_CATEGORIES = len(CATEGORIES)

# Checking the dataset steps
print(" ### INFO: Dataset statistics ###")
from pprint import pprint
pprint(dict_without_keys(dataset_statistics, ["episode_lengths",
    "cat_scene_filenames", "scene_cat_filenames", "scene_filenames"]))
print("")

# Fake environment instantiation to create the agent models later on

# TODO: add adaptive creation of single_observation_space so that RGB and RGBD based variants
# can be evaluated at thet same time
from gym import spaces
single_action_space = spaces.Discrete(4)
if args.pretrained_model_name.__contains__("rgb"):
    single_observation_space = spaces.Dict({
        "rgb": spaces.Box(shape=[128,128,3], low=0, high=255, dtype=np.uint8),
        "audiogoal": spaces.Box(shape=[2,16000], low=-3.4028235e+38, high=3.4028235e+38, dtype=np.float32),
        "spectrogram": spaces.Box(shape=[65,26,2], low=-3.4028235e+38, high=3.4028235e+38, dtype=np.float32)
    })
if args.pretrained_model_name.__contains__("rgbd"):
    # Override the single action space in case "rgbd" is in the experiment name
    single_observation_space = spaces.Dict({
        "rgb": spaces.Box(shape=[128,128,3], low=0, high=255, dtype=np.uint8),
        "depth": spaces.Box(shape=[128,128,1], low=0, high=255, dtype=np.uint8),
        "audiogoal": spaces.Box(shape=[2,16000], low=-3.4028235e+38, high=3.4028235e+38, dtype=np.float32),
        "spectrogram": spaces.Box(shape=[65,26,2], low=-3.4028235e+38, high=3.4028235e+38, dtype=np.float32)
    })

# Define the target of probing
## "category" -> how easy to predict category based on the learned features / inputs
## "scene" -> how easy to predict scene based on the learned features / inputs
PROBING_TARGETS = {}
for probe_target in args.probing_targets:
    if probe_target == "category":
        PROBING_TARGETS["category"] = {}
        PROBING_TARGETS["category"]["n_classes"] = N_CATEGORIES
    elif probe_target == "scene":
        PROBING_TARGETS["scene"] = {}
        PROBING_TARGETS["scene"]["n_classes"] = N_SCENES

# Define which fields of an agent to use for the probes
PROBING_INPUTS = args.probing_inputs

# Load the model which features will be probed
args_copy = copy.copy(args)
if args.pretrained_model_name.__contains__("gru"):
    agent = ActorCritic(single_observation_space, single_action_space, args.hidden_size, extra_rgb=False,
        analysis_layers=models.GRU_ACTOR_CRITIC_DEFAULT_ANALYSIS_LAYER_NAMES).to(device)
elif args.pretrained_model_name.__contains__("pgwt"):
    agent = Perceiver_GWT_GWWM_ActorCritic(single_observation_space, single_action_space, args, extra_rgb=False,
        analysis_layers=models.PGWT_GWWM_ACTOR_CRITIC_DEFAULT_ANALYSIS_LAYER_NAMES + ["state_encoder.ca.mha"]).to(device)
agent.eval()

# TODO: add more controls on the model path ?
if args.pretrained_model_path is not None:
    agent_state_dict = th.load(args.pretrained_model_path)
    agent.load_state_dict(agent_state_dict)

# Class for a generic linear probe network
# TODO: might want to add a few layers layer ?
class GenericProbeNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, depth=1, hid_size=512, bias=False):
        super().__init__()
        assert depth >= 1, "Probe not deep enough: {depth}"
            
        hiddens = [hid_size for _ in range(depth)]
        network = []
        for h0, h1 in zip([input_dim, *hiddens[:-1]], [*hiddens[1:], output_dim]):
            network.append(nn.Linear(h0, h1, bias=bias))
            network.append(nn.ReLU())
        network.pop()
            
        self.network = nn.Sequential(*network)
    
    def forward(self, x):
        return self.network(x)

# Instantiating probes
PROBES_METADATA = {}
for probe_target_name, probe_target_info in PROBING_TARGETS.items():
    if probe_target_name not in PROBES_METADATA.keys():
        PROBES_METADATA[probe_target_name] = {}
    for probe_input in PROBING_INPUTS: # NOTE: maybe switch order with the MODEL_VARIANTS ???

        if probe_input not in PROBES_METADATA[probe_target_name].keys():
            PROBES_METADATA[probe_target_name][probe_input] = {}

        # TODO: make the probe's input dim adapt to what will actually be probed.
        PROBES_METADATA[probe_target_name][probe_input]["probe_input_dim"] = 512
        PROBES_METADATA[probe_target_name][probe_input]["probe_output_dim"] = probe_target_info["n_classes"]

# Dictionary that will holds the probe networks and their optimizers
PROBES = copy.copy(PROBES_METADATA)
for probe_target_name, probe_target_info in PROBING_TARGETS.items():
    for probe_input in PROBING_INPUTS: # NOTE: maybe switch order with the MODEL_VARIANTS ???
        probe_input_dim = PROBES[probe_target_name][probe_input]["probe_input_dim"]
        probe_output_dim = PROBES[probe_target_name][probe_input]["probe_output_dim"]

        probe_network = GenericProbeNetwork(probe_input_dim, probe_output_dim,
                                            args.probe_depth, args.probe_hid_size, args.probe_bias).to(device)
        print(probe_network)
        if not args.cpu and th.cuda.is_available():
            # TODO: GPU only. But what if we still want to use the default pytorch one ?
            optimizer = apex.optimizers.FusedAdam(probe_network.parameters(), lr=args.lr, eps=1e-5, weight_decay=args.optim_wd)
        else:
            optimizer = th.optim.Adam(probe_network.parameters(), lr=args.lr, eps=1e-5, weight_decay=args.optim_wd)

        PROBES[probe_target_name][probe_input]["probe_network"] = probe_network
        PROBES[probe_target_name][probe_input]["probe_optimizer"] = optimizer
        PROBES[probe_target_name][probe_input]["agent"] = args.pretrained_model_name
        PROBES[probe_target_name][probe_input]["agent_state_dict_path"] = args.pretrained_model_path

# NOTE / TODO: probe training might benefit from using different batch sizes ?

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
        
        ## Compute action coefficient for CEL of BC
        dataset_stats_filepath = f"{args.dataset_path}/dataset_statistics.bz2"
        # Override dataset statistics if the file already exists
        if os.path.exists(dataset_stats_filepath):
            with open(dataset_stats_filepath, "rb") as f:
                dataset_statistics = cpkl.load(f)
        
        self.scenes = list(dataset_statistics["scene_counts"].keys())
        # Dictionary that returns the scene index given the scene id
        self.scene_name_to_idx = {scene: i for i, scene in enumerate(self.scenes)}
        
        print(f"Initialized IterDset with {len(self.ep_filenames)} episodes.")
    
    def __iter__(self):
        batch_length = self.batch_length
        N_SCENES = len(self.scenes)
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
                "scene": np.zeros([batch_length])
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
                
                # Add scene info to the obs dict list
                scene_idx = self.scene_name_to_idx[edd["scene_id"]]
                scene_idx_list = [scene_idx for _ in range(edd["ep_length"])]

                # Append the data to the bathc trjectory
                rs = batch_length - ssf # Reamining steps
                horizon = ssf + min(rs, edd["ep_length"])
                for k, v in edd["obs_list"].items():
                    obs_list[k][ssf:horizon] = v[:rs]
                action_list[ssf:horizon] = np.array(edd["action_list"][:rs])[:, None]
                reward_list[ssf:horizon] = np.array(edd["reward_list"][:rs])[:, None]
                done_list[ssf:horizon] = np.array(edd["done_list"][:rs])[:, None]
                obs_list["scene"][ssf:horizon] = np.array(scene_idx_list)[:rs]
                ssf += edd["ep_length"]

                if ssf >= self.batch_length:
                    break

            # Adjust shape of "depth" to be [T, H, W, 1] instead of [T, H, W]
            obs_list["depth"] = obs_list["depth"][:, :, :, None]
            # TODO: add enough data about the scene to be able to do the probing
            # Since the dataset statistics can be accessed here too, we can generate
            # the vector of targets for the scene
            
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

# Instantiate the dataset object
dloader = make_dataloader3(args.dataset_path, batch_size=args.num_envs,
                            batch_length=args.num_steps, seed=args.seed, num_workers=8)

# TODO: consider pre-computing CE weights for categories / scenes to balance the CE loss ?

# TODO
# Could we maybe pre-cmpute all the foreward passes for all the model variants once,
# then we don't have to re-run those in case we train for more than one epoch ?
# Do note that the memory cost is likely to be large, so trading training speed for memory ?
# This could hamper the training of multiple probes at the same time though

# Although even the "epoch" is not a real epoch, since we don't have the guarantee
# that all the steps are sampled exactly once.
# Experiment logger
tblogger = TBLogger(exp_name=args.exp_name, args=args)
print(f"# Logdir: {tblogger.logdir}")
should_log_training_stats = tools.Every(args.log_training_stats_every)
should_eval = tools.Every(args.eval_every)
## Save the PROBES dict as metadata
## Note that we first remove the "probe_network" field which has the actual weights
PROBES_METADATA__FILEPATH = f"{tblogger.get_logdir()}/PROBES_METADATA.bz2"
with open(PROBES_METADATA__FILEPATH, "wb") as f:
    cpkl.dump(PROBES_METADATA, f)

# Training start
start_time = time.time()

# NOTE: this time total-steps means how many time .backward() is called on each probe
# One epoch would be equual to "DATASET_SIZE" in steps / (num_envs * num_steps)
n_updates = 0
steps_per_update = args.num_envs * args.num_steps
total_updates = int((args.total_steps * args.n_epochs) / args.num_envs / args.num_steps) + 1# How many updates expected in total for one epoch ?
print(f"Expected number of updates: {total_updates}")

for global_step in range(1, args.total_steps * args.n_epochs + steps_per_update, steps_per_update):
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
        elif k in ["category"]:
            obs_list[k] = v.permute(1, 0, 2) # BTC -> TBC
        elif k in ["scene"]:
            obs_list[k] = v.permute(1, 0) # BT -> TB
        else:
            # TODO: handle other fields like "category", etc...
            pass
    
    action_list = action_list.permute(1, 0, 2) # TODO: probably uneeded for probing ?
    done_list = done_list.permute(1, 0, 2)
    mask_list = 1. - done_list
    
    prev_actions_list = th.zeros_like(action_list)
    prev_actions_list[1:] = action_list[:-1]
    prev_actions_list = F.one_hot(prev_actions_list.long()[:, :, 0], num_classes=4).float()
    prev_actions_list[0] = prev_actions_list[0] * 0.0

    # Finally, also flatten across T x B, let the RNN do the unflattening if needs be
    action_list = action_list.reshape(-1) # TODO: probably uneeded for probing ?
    done_list = done_list.reshape(-1, 1)
    mask_list = mask_list.reshape(-1, 1)
    prev_actions_list = prev_actions_list.reshape(-1, 1)

    # Holder for the probe losses and accs.
    probe_losses_dict = {}

    # For each "agent_variant", iterate
    # TODO: once we have more than "category", it is more efficient to iterate over the agent first,
    # Do the forward pass, then iterate over the probe target (category, scene, etc...) then

    # Forward pass with the agent model to collect the intermediate features
    # Stores in agent_features
    # This will be used to recompute the rnn_hidden_states when computiong the new action logprobs
    if args.pretrained_model_name.__contains__("gru"):
        rnn_hidden_state = th.zeros((1, args.batch_chunk_length, args.hidden_size), device=device)
    elif args.pretrained_model_name.__contains__("pgwt"):
        rnn_hidden_state = agent.state_encoder.latents.repeat(args.batch_chunk_length, 1, 1)
    else:
        raise NotImplementedError(f"Unsupported agent-type:{args.pretrained_agent_name}")
    
    with th.no_grad():
        agent_outputs = agent.act(obs_list, rnn_hidden_state, masks=mask_list) #, prev_actions=prev_actions_list)
        
    for probe_target_name, probe_target_dict in PROBES.items():
        # probe_target_name: "category", "scene", more generally the targeted concept of the probing
        # probe_target_dict: { "state_encoder": {"agent_variant": Torch Model} }
        for probe_target_input_name, agent_variant_probes in probe_target_dict.items():
            # probe_target_input_name: the input of the probe, such as "state_encoder", and other
            # agent_variant_probes: dict taht holds {"agent_variant": Torch Model}
            
            probe = agent_variant_probes["probe_network"]
            probe_optim = agent_variant_probes["probe_optimizer"]
            probe_optim.zero_grad()

            # Forward pass of the probe network itself
            if probe_target_input_name == "state_encoder":
                probe_inputs = agent._features["state_encoder"][0]
            elif probe_target_input_name.__contains__("visual_encoder") or \
                    probe_target_input_name.__contains__("audio_encoder"):
                probe_inputs = agent._features[probe_target_input_name]
            else:
                raise NotImplementedError(f"Attempt to use {probe_target_input_name} as probe input.")
            
            probe_logits = probe(probe_inputs)
            
            # TODO: generate probe_targets
            if probe_target_name == "category":
                probe_targets = obs_list["category"].reshape(-1, N_CATEGORIES).argmax(axis=1)
            elif probe_target_name == "scene":
                probe_targets = obs_list["scene"].long().reshape(-1)
            else:
                raise NotImplementedError(f"Unsupported probe target: {probe_target_name}.")
            
            # Loss
            # TODO: CE weights depending on the probing target and such
            probe_ce_loss = F.cross_entropy(probe_logits, probe_targets)

            probe_ce_loss.backward()
            probe_optim.step()

            # Store the loss valuesl for logging later
            metric_stem = f"{probe_target_name}|{probe_target_input_name}"
            loss_name = f"{metric_stem}__probe_loss"
            probe_losses_dict[loss_name] = probe_ce_loss.item()
            acc_name = f"{metric_stem}__probe_acc"
            probe_losses_dict[acc_name] = (F.softmax(probe_logits, dim=1).argmax(1) == probe_targets).float().mean().item()

    # Tracking the number of NN updates (for all probes)
    n_updates += 1

    if should_log_training_stats(n_updates):
        for k, v in probe_losses_dict.items():
            print(f"{k}: {round(v,3)}")
        print("")

        tblogger.log_stats(probe_losses_dict, global_step, prefix="train")
        # Log addtional data
        tblogger.log_stats({
            "n_updates": n_updates,
        }, global_step, prefix="info")
        tblogger.log_stats({
            "global_step": global_step,
        }, global_step)

# Saving models after the training
for probe_target_name, probe_target_dict in PROBES.items():
    for probe_target_input_name, agent_variant_probes in probe_target_dict.items():
        probe = agent_variant_probes["probe_network"]
        probe_statedict_filename = f"{probe_target_name}__{probe_target_input_name}__probe.pth"
        
        tblogger.save_model_dict(probe.state_dict(), probe_statedict_filename)

tblogger.close()
