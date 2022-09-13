# Soundspaces - Habitat-lab - Habitat-sim setup

# General guideline as of 2022-07-21

- Soundspaces currently at commit fb68e410a4a1388e2d63279e6b92b6f082371fec
- While `habitat-lab` and `habitat-sim` recommend using python 3.7 at leats, this procedure goes as far as python 3.9 to have better compatibility with more recent Torch libraries.
- `habitat-lab` is built at version 0.2.2
- `habitat-sim` is built from the commit [80f8e31140eaf50fe6c5ab488525ae1bdf250bd9](https://github.com/facebookresearch/habitat-sim/tree/80f8e31140eaf50fe6c5ab488525ae1bdf250bd9).

# System specifics

- Ubuntu 20.04
- SuperMicro X11DAI-N Motherboard
- Intel(R) Xeon(R) Silver 4216 CPU @ 2.10GHz | 2 Sockets * 16 Cores pers sockets * 2 Threads per core
- NVIDIA RTX 3090 GPU
- NVIDIA 515.57 (straight from NVIDIA's website)
- CUDA Toolkit 11.7 (also from NVIDIA's website)


```bash
conda create -n ss-hab-headless-py39 python=3.9 cmake=3.14.0 -y
conda activate ss-hab
pip install pytest-xdist
pip install rsatoolbox # Neural activity pattern analysis
```

# Habitat-lab Stable 0.2.2

```bash
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git # Currently @ 0f454f62e41050bc90ca468c62db35d7484923ff
cd habitat-lab
# pip install -e .
# While installing deps., pip threw out error that some conflict due to TF 2.1.0 was preventing installing all the deps.
# Manually ran `pip install tensorflow-gpu==2.1.0` just to be sure
# Additionally, install all the other deps for dev. purposes
pip install -r requirements.txt
python setup.py develop --all # So far so good

# Leave the directory for the subsequent steps
cd ..
```

# Habitat-sim

- According to SS's docs, using the sound simulator requires building with `--audio` flag for sound support.

- Building `--with-cuda` requires CUDA toolkit to be installed and accessible through the following variable environmetns:
    - `PATH` contains `/usr/local/cuda-11.7/bin`
    or similar
    - `LD_LIBRARY_PATH` contains `/usr/local/cuda-11.7/lib64`
    
```bash
# Makes sure all system level deps are installed.
sudo apt-get update || True
sudo apt-get install -y --no-install-recommends libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev # Ubuntu
# Clone habitat-sim repository
git clone https://github.com/facebookresearch/habitat-sim.git # Current @ 80f8e31140eaf50fe6c5ab488525ae1bdf250bd9
cd habitat-sim
# Checkout the commit suggest by ChanganVR
git checkout 80f8e31140eaf50fe6c5ab488525ae1bdf250bd9
# Build habitat-sim with audio and headless support
python setup.py install --with-cuda --bullet --audio --headless # compilation goes brrrr...
pip install hypothesis # For the tests mainly
```

[Additional building instructions](https://github.com/facebookresearch/habitat-sim/blob/main/BUILD_FROM_SOURCE.md)

## Preliminary testing

__Getting the test scene_datasets__:

Some tests (tests/test_physics.py, tests/test_simlulator.py) require "habitat test scenes" and "habitat test objects". Where are those ? Datasets ?
The provided download tool is broken. Manually downloading those:

To get the `habitat-test-scenes`, from the `habitat-sim` root folder:

```bash
python src_python/habitat_sim/utils/datasets_download.py --uids habitat_test_scenes
```

__Getting the (test / example) habitat objects__:

```bash
python src_python/habitat_sim/utils/datasets_download.py --uids habitat_example_objects
```

With this, `pytest tests/test_physics.py` should have 100% success rate.

__Interactive testing__

This assumes `habitat-sim` was built with display support.

```
python examples/viewer.py --scene data/scene_datasets/habitat-test-scenes/skokloster-castle.glb
```
Note: In case it return `ModuleNotFound examples.settings` error, edit the `examples/viewer.py` and remove the `examples.` from the relevant import line.

__Non-interactive testing__

With compiled `habitat-sim`, this fails to run, returning a `free(): invalid pointer`.
Should work if run from `habitat-lab` directory instead of `habitat-sim`.

```bash
python examples/example.py --scene data/scene_datasets/habitat-test-scenes/skokloster-castle.glb
```

## Acquiring datasets necessary for simulations

__Getting the ReplicaCAD dataset__

```bash
python src_python/habitat_sim/utils/datasets_download.py --uids replica_cad_dataset
```

Running the physics interaction simulation:
```bash
python examples/viewer.py --dataset data/replica_cad/replicaCAD.scene_dataset_config.json --scene apt_1
```

<!-- __Getting the HM3D dataset__

NOTE: This is not necessary for Soundspaces 2.0

```bash
python src_python/habitat_sim/utils/datasets_download.py --username USERNAME --password PASSWORD --uids hm3d_minival
python src_python/habitat_sim/utils/datasets_download.py --username USERNAME --password PASSWORD --uids hm3d_minival_habitat
``` -->

__Other notes__
- To get `tests/test_controls.py` to pass, need to `pip install hypothesis` if not already.
- When habitat-sim is manually compiled, it seems that commands such as `python -m habitat_sim.utils.datasets_download` do not work. Instead use `cd /path/to/habitat-sim && python src_python/utils/dowload_datasets.py`. This allegedly does not happen if `habitat-sim` was installed through `conda`.

To check that both `habitat-lab` and `habitat-sim` work with each other:

```bash
cd ../habitat-lab
python examples/example.py
```

It should say "... ran for 200 steps" or something similar.


# Soundspaces 2.0

```bash
# git clone https://github.com/facebookresearch/sound-spaces.git # Currently @ fb68e410a4a1388e2d63279e6b92b6f082371fec
git clone https://github.com/facebookresearch/sound-spaces.git
cd sound-spaces
git checkout fb68e410a4a1388e2d63279e6b92b6f082371fec
pip install -e .
```

## Downloading the `scene_datasets` for SS 2.0 habitat audio visual simulations
Requires access to the `download_mp.py` tool from official Matterport3D.
See https://github.com/facebookresearch/habitat-lab#matterport3d

```bash
mkdir -p data/scene_datasets
mkdir -p data/versioned_data/mp3d
python /path/to/download_mp.py --task habitat -o /path/to/sound-spaces/data/versioned_data/mp3d
```

This will download a ZIP file into `/path/to/sound-spaces/data/versioned_data/mp3d/v1/tasks/mp3d_habitat.zip`

Unzip this file to obtain `/path/to/sound-spaces/data/versioned_data/mp3d/v1/tasks/mp3d`.
This folder should contain files like: `17DRP5sb8fy`, `1LXtFkjw3qL`, etc...

Make it so that `/path/to/sound-spaces/data/scene_datasets/mp3d` points to `/path/to/sound-spaces/data/versioned_data/mp3d/v1/tasks/mp3d`. For example:

```bash
ln -s /path/to/sound-spaces/data/versioned_data/mp3d/v1/tasks/mp3d` `/path/to/sound-spaces/data/scene_datasets/mp3d`
```

Some additional metadata that is intertwined with other datasets and features of soundspaces is also required:

```bash
# From /path/to/soundspaces/data, run:
wget http://dl.fbaipublicfiles.com/SoundSpaces/metadata.tar.xz && tar xvf metadata.tar.xz # 1M
wget http://dl.fbaipublicfiles.com/SoundSpaces/sounds.tar.xz && tar xvf sounds.tar.xz #13M
wget http://dl.fbaipublicfiles.com/SoundSpaces/datasets.tar.xz && tar xvf datasets.tar.xz #77M
wget http://dl.fbaipublicfiles.com/SoundSpaces/pretrained_weights.tar.xz && tar xvf 
pretrained_weights.tar.xz
# This heavy file can be ignored for SS 2.0
# wget http://dl.fbaipublicfiles.com/SoundSpaces/binaural_rirs.tar && tar xvf binaural_rirs.tar # 867G
```

SS 2.0 command provided in the Reamde are based on `mp3d` datasets.
If trying to run the interactive mode command latter on, it will likely throw up some error about `data/metadata/default/...` being absent.

This will require the following tweak:

- in `sound-spaces/data/metadata`: `ln -s mp3d default`

## Downloading `mp3d_material_config.json`

Download from the following link: https://github.com/facebookresearch/rlr-audio-propagation/blob/main/RLRAudioPropagationPkg/data/mp3d_material_config.json and put it in `/path/to/soundspaces/data/`.

<!-- This file is also expected to be found in `habitat-lab/data` and potentially `habitat-sim/data`.
For good measure, also reflect the existence of this file in those directories, using either symbolic link or copying the actual file.
Although not rigorously checked, it might be useful to also symlink or copy the installed dataset at /soundspaces/data/scene_dataset` in the appropriate directory under `habitat-sim` and habitat-lab`
too, to mirror the data available to all three libraries. 
NOTE: Maybe not. Maybe because I did not use the proper link when donwloading with wget, result in an HTML code source page instead of JSON.
-->

## Torch

The `torch` install that comes with the dependencies should work by default on something like GTX 1080 Ti.
However, because that one relies on `CUDA 10.2` it cannot be used with an RTX 3090 for example (_CUDA Error: no kernel image is available for execution on the device ..._).
Training on an RTX 3090 as of 2022-07-21, thus requires upgrading to a `torch` version that supports `CUDA 11`.
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
This will install `torch==1.12.0`.

CUDA 11.6 unfortunately seems to create too many conflicts with other dependencies, solving the environment ad infinitum.
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

<!-- Also take this change to get `torchvision`:
```bash
pip install torchvision
``` -->

### Testing SS2.0 in interactive mode

For a machine with display, and with `habitat-sim` not being built with the `--headless` flag.

```bash
python scripts/interactive_mode.py
```

**Note**: This worked on the internal graphics of the motherboard, but not on the RTX 3090 GPU.
It might work on an older, or not so *fancy* GPU.
In any case, interactive mode is not that important for the RL use case.

### **[New]** Training continuous navigation agent DDPPO baseline

```bash
python ss_baselines/av_nav/run.py --exp-config ss_baselines/av_nav/config/audionav/mp3d/train_telephone/audiogoal_depth_ddppo.yaml --model-dir data/models/ss2/mp3d/dav_nav CONTINUOUS True
```

### Training continuous navigation PPO baseline

```bash
python ss_baselines/av_nav/run.py --exp-config ss_baselines/av_nav/config/audionav/mp3d/train_telephone/audiogoal_depth.yaml --model-dir data/models/ss2/mp3d/dav_nav_ppo/ CONTINUOUS True
```

### Evulating the trained agent

This is done using the test confgurations suggested in the `sound-spaces/ss_baselines/av_nav/README.md` file.

```
python ss_baselines/av_nav/run.py --run-type eval --exp-config ss_baselines/av_nav/config/audionav/mp3d/test_telephone/audiogoal_depth.yaml EVAL_CKPT_PATH_DIR data/models/ss2/mp3d/dav_nav/data/ckpt.100.pth CONTINUOUS True
```

**Missing `audiogoal` in `observations` error**

Runnin the commnad above will probably spit out an error related to the `audiogal` field missing in the `observations`
directly.
The fix is to change the base task configuration with the following:

```
TASK:
  TYPE: AudioNav
  SUCCESS_DISTANCE: 1.0

  # Original
  # SENSORS: ['SPECTROGRAM_SENSOR']
  # GOAL_SENSOR_UUID: spectrogram

  # For eval support
  SENSORS: ['AUDIOGOAL_SENSOR', 'SPECTROGRAM_SENSOR']
  GOAL_SENSOR_UUID: spectrogram # audiogoal
```
Basically, it adds the `AUDIOGOAL_SENSOR` to the config, which in turns generates the corresponding field in the observation of the agent

**TypeError: write_gif() got an unexpected keyword argument 'verbose'**

Best guess is some form of mismatch between the moviepy version that the tensorboard used here expectes and the one that is actually installed.
Current versions are `torch==1.12.0` installed from conda according to the official pytorch website, and `moviepy==2.0.0.dev2` install from PyPI.

A work around was to edit the `make_video` in `/path/to/venv/lib/python3.9/site-packages/torch/utils/tensorboard/summary.py` to add the case when `moviepy` does not support the `verbose` argument:

```python
def make_video(tensor, fps):
    try:
        import moviepy  # noqa: F401
    except ImportError:
        print("add_video needs package moviepy")
        return
    try:
        from moviepy import editor as mpy
    except ImportError:
        print(
            "moviepy is installed, but can't import moviepy.editor.",
            "Some packages could be missing [imageio, requests]",
        )
        return
    import tempfile

    t, h, w, c = tensor.shape

    # encode sequence of images into gif string
    clip = mpy.ImageSequenceClip(list(tensor), fps=fps)

    filename = tempfile.NamedTemporaryFile(suffix=".gif", delete=False).name
    try:  # newer version of moviepy use logger instead of progress_bar argument.
        clip.write_gif(filename, verbose=False, logger=None)
    except TypeError:
        try:  # older version of moviepy does not support progress_bar argument.
            clip.write_gif(filename, verbose=False, progress_bar=False)
        except TypeError:
            try: # in case verebose argument is also not supported
                clip.write_gif(filename, verbose=False)
            except TypeError:
                clip.write_gif(filename)
```

### Generating audio and video from SS2.0 trajectories.

SS2.0 supports `RGB_SENSOR` and `DEPTH_SENSOR` for agent visual percpetion.
For the acoustic perception, it supports the `SPECTROGRAM_SENSOR` and the `AUDIOGOAL_SENSOR`, the latter returns the waveform that are initially generated in thte `observations` field of the `env.step()` method returns.

A demonstration is given in the `sound-spaces/env_test.ipynb` notebook in this repository.
Unfortunately, Tensorboard does not seem to support logging of video with incorporated audio.
However, WANDB is capable of doing so, but the logging step will be out of sync with the actual training step (Tensorboard logging step) of the agent.

# Custom PPO Implementation

A simplified PPO + GRU implementation that exposes the core of the algorithm, as well interactoins with the environment.

## Additional dependencies
```
# Individual deps. install
pip install wandb # 0.13.1
pip install nvsmi # 0.4.2, for experimetn GPU usage configuration

# One liner
pip install wandb nvsmi
```

## Usage

This will use RGB + Spectrogram as input for the agent, create a timestamped TensorBoard folder automatically and log training metrics as well as video, with and without audio.

```bash
python ppo_av_nav.py
```

# SAVi

Two main features that might be of interest for this project
- removes the limitation of only having a _ringing telephone_. Namely, adds 21 objects with their distinct sounds
- the sound is not continuous over the whole episode, but of variable length insetad. This supposedly forces the agent to learn the association between the category of the object and its acoustic features.

## Addtional setup

On top of all the above steps so far for the default SS installation,
1. Run the `python scripts/cache_observations.py`. Note that since we only use mp3d dataset for now, it will require editing that script to comment out line 105, to make it skip the `replica` data set, which proper installation is skipped in the steps above.
Once this is done, add the symbolic link so that the training scripts can find the file at the expected path:
```bash
ln -s /path/to/sound-spaces/data/scene_observations/mp3d /path/to/sound-spaces/data/scene_observations/default
```

2. Run the scripts as per the README in the SAVi folder:

a. First, the pre-training the goal label predictor:
```bash
python ss_baselines/savi/pretraining/audiogoal_trainer.py --run-type train --model-dir data/models/savi --predict-label
```
This step seems to require that huge binaural dataset that was skipped back in sound-spaces dataset acquisition section earlier.
```
wget http://dl.fbaipublicfiles.com/SoundSpaces/binaural_rirs.tar && tar xvf binaural_rirs.tar # 867G
```

a. without pre-training, continuous simulator
```bash
python ss_baselines/savi/run.py --exp-config ss_baselines/savi/config/semantic_audionav/savi.yaml --model-dir data/models/savi CONTINUOUS True
```

b. with pre-training
```bash

```

**Additional notes**
- The pretrained weights will be found in `/path/to/sound-spaces/data/pretrained_weights`, assuming they were properly donwload in the dataset acquisition phase.

## TODO:
- [ ] Does it support continuous mode ? or just dataset based ?

# Perceiver
The base impelementation of Perceiver and PerceiverIO that I planned to use was from [lucidrains/perceiver-pytorch](https://github.com/lucidrains/perceiver-pytorch) repository.
However it did not have support for multiple modality as input.
Found the [oepnclimatefix/perceiver-pytorch](https://github.com/openclimatefix/perceiver-pytorch) fork that seems to have a owrking multi modal PerceiverIO, so using that as a base instead.
After forking into [dosssman/perceiver-pytorch](https://github.com/dosssman/perceiver-pytorch), install the `main-ofc` branch for PerceiverIO with multi-modal support.
```
git clone git@github.com:dosssman/perceiver-pytorch.git --branch main-ocf
cd perceiver-pytorch
pip install -e .
```

# AudiCLIP

Attempt at using the pre-trained audo encoder for av_nav / SAVi tasks in SS baselines

## Additional dependencies
From conda:
```bash
conda install ignite -c pytorch
```
Might require a downgrade of Numpy version to <= 1.22 .

## Prototype 1
- AudioCLIP uses the ESResNeXt to extract audio features from raw wave form files.
- Said seems to have been trained on monoraul data from the ECS50 and US8K sound datasets.
- In this variant, the pre-trained ESResNeXt is duplicazted at the beginning of the training, with each instance processing the left and right channel of the audio stream, respectively.
While this is relatively straight foward to implement, the downside is that the memory cost of maintaining two of such wide model is non-negligible

As of 2022-08-18, after running the PPO based on SS's baselines on AvNav task, while the reference run reaches around 20% success rate within 1M steps, the variant that uses AudioCLIP's audio encoder iterates at least 5 times slower, hogs the hole RTX 3090 GPU memory, and has success rate near 0 for around 600K steps, show now sign of improvement.
And avenue left to explore later.

## Prototype 2:
Motivated by memory capacity limit.
The idea is to have independent "first layers" for each channel of the RIR data, then fuse the results before passing it into the shared ESResNetXt network. Should atl east reduce the burden by 75%, since there are that much layers of the network that end up being shared instead.
- WIP

## Prototype 3:
The general idea is to shave of a few layers and components from the original model, and use that instead.
Such network sould still be able to produce some features that are better suited to represent infor from audio sources ?
- WIP

# Other
### [OUTDATED as of 2022-07-21] RLRAudioPropagationChannelLayoutType` error workaround

Past this point, there might be some `RLRAudioPropagationChannelLayoutType` related error when trying to run the interactive mode.

If this fork of soundspaces was used: `git clone https://github.com/dosssman/sound-spaces.git --branch ss2-tweaks`, then skip until the **Finally testing SS2** section.

Otherwise, this will require a workaround in the soundspaces simulator.

Namely, comment or delete the line 125 of `sound-spaces/soundspaces/simulator_continuous.py` and add:

```python
import habitat_sim._ext.habitat_sim_bindings as hsim_bindings
channel_layout.channelType = hsim_bindings.RLRAudioPropagationChannelLayoutType.Binaural
```
instead.

This workaround is adapted from `habitat-sim/examples/tutorials/audio_agent.py`.

The reason is because the soundspace author are using a version of `habitat-sim` that is more recent than `v2.2.0`, and where the   `habitat_sim.sensor.RLRAudioPropagationChannelLayoutType` object is properly defined.
Since we clone `habitat-sim@v2.2.0`, however, we revert to using the `hsim_bindings` directly.