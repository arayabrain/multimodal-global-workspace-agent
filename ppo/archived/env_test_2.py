import numpy as np
import matplotlib.pyplot as plt

import argparse
from ss_baselines.av_nav.config import get_config
from ss_baselines.common.env_utils import construct_envs
from ss_baselines.common.environments import get_env_class


if __name__ == "__main__":
    config = get_config(
        # config_paths="ss_baselines/av_nav/config/audionav/mp3d/env_test_0.yaml", # RGB + AudiogoalSensor
        config_paths="env_configs/audiogoal_rgb_depth_nocont.yaml",
        # opts=["CONTINUOUS", "False"],
        # run_type="eval"
    )
    config.defrost()
    config.NUM_PROCESSES = 1
    config.USE_SYNC_VECENV = True
    config.USE_VECENV = False
    config.freeze()
    # print(config)

    envs = construct_envs(config, get_env_class(config.ENV_NAME))
    obs = envs.reset()

    outputs = envs.step([0])
    obs, reward, done, info = [list(x) for x in zip(*outputs)]


    # Instantiate a Shortest Path Follower agent
    from soundspaces.tasks.shortest_path_follower import ShortestPathFollower

    spf_agent = ShortestPathFollower(sim = env._env.sim, goal_radius = 0.5)
    spf_agent._build_follower()
    
    tmp = 1
