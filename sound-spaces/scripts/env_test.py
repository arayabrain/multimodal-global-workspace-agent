# %matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

import argparse

from habitat.datasets import make_dataset
from ss_baselines.av_nav.config import get_config
from ss_baselines.common.environments import AudioNavRLEnv

config = get_config(
    config_paths="ss_baselines/av_nav/config/audionav/mp3d/env_test_0.yaml", # RGB + AudiogoalSensor
    # opts=["CONTINUOUS", "True"],
    run_type="eval")
config.defrost()
config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP") # Note: can we add audio sensory info fields here too ?
config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
config.freeze()
print(config)

dataset = make_dataset(id_dataset=config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
env = AudioNavRLEnv(config=config, dataset=dataset)

## Collect one full episode
observation, done, ep_length = env.reset(), False, 0
ep_observations = [observation]

while not done:
    # The "STOP" action will make the episode finish early, but 
    # this work around is not very efficient.
    action = {"action": 'STOP', "action_args": None}
    while action["action"] == 'STOP':
        action = env.action_space.sample()

    observation, reward, done, info = env.step(**action)
    ep_length += 1

    ep_observations.append(observation)

    print("")
    print("###########################################")
    print(f"# DEBUG: Episode length: {ep_length}")
    print(f"DEBUG: Done value {done}")
    print("###########################################")
    print("")

ep_rgb_observations = [obs["rgb"] for obs in ep_observations]
ep_audiogoal_observations = [obs["audiogoal"] for obs in ep_observations]

import os
from typing import Dict, List, Optional
import moviepy.editor as mpy
from moviepy.audio.AudioClip import CompositeAudioClip, AudioArrayClip

def images_to_video_with_audio(
    images: List[np.ndarray],
    output_dir: str,
    video_name: str,
    audios: List[str],
    sr: int,
    fps: int = 1,
    quality: Optional[float] = 5,
    **kwargs
):
    r"""Calls imageio to run FFMPEG on a list of images. For more info on
    parameters, see https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
    Args:
        images: The list of images. Images should be HxWx3 in RGB order.
        output_dir: The folder to put the video in.
        video_name: The name for the video.
        audios: raw audio files
        fps: Frames per second for the video. Not all values work with FFMPEG,
            use at your own risk.
        quality: Default is 5. Uses variable bit rate. Highest quality is 10,
            lowest is 0.  Set to None to prevent variable bitrate flags to
            FFMPEG so you can manually specify them using output_params
            instead. Specifying a fixed bitrate using ‘bitrate’ disables
            this parameter.
    """
    assert 0 <= quality <= 10
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"

    audio_clips = []
    multiplier = 0.5
    for i, audio in enumerate(audios):
        audio_clip = AudioArrayClip(audio.T[:int(sr * 1 / fps)] * multiplier, fps=sr)
        audio_clip = audio_clip.set_start(1 / fps * i)
        audio_clips.append(audio_clip)
    composite_audio_clip = CompositeAudioClip(audio_clips)
    video_clip = mpy.ImageSequenceClip(images, fps=fps)
    video_with_new_audio = video_clip.set_audio(composite_audio_clip)
    video_with_new_audio.write_videofile(os.path.join(output_dir, video_name))

images_to_video_with_audio(
    images=ep_rgb_observations,
    output_dir="/tmp/ss-videos",
    video_name="ss_video_dgb",
    audios=ep_audiogoal_observations,
    sr=config.TASK_CONFIG.SIMULATOR.AUDIO.RIR_SAMPLING_RATE, # 16000 for mp3d dataset
    fps=config.TASK_CONFIG.SIMULATOR.VIEW_CHANGE_FPS,
)

# Experimenting with WANDB and audio video logging
import wandb
wandb.init(project="ss-hab-test", entity="dosssman", settings=wandb.Settings(start_method='thread'))
wandb.log({"audio_video": wandb.Video(
                data_or_path="/tmp/ss-videos/ss_video_dgb.mp4")})
