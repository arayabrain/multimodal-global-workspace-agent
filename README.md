# Soundspaces - Habitat-lab - Habitat-sim setup

## General guideline as of 2022-07-06

- `habitat-lab` and `habitat-sim` require python>=3.7. The documentation mentions that 3.7 is the one tested and confirmed to work.
This can be a potential source of conflict when installing pytorch with more recent version of CUDA / Python.
- Soundspaces (SS) recommends using `habitat-lab` and `habitat-sim` v0.2.1.
However, the current version of `habitat-lab` and `habitat-sim` is 0.2.2.
Furthermore, to use SoundSpaces 2.0 (audio visual environemnt simulation for RL agents) requires to build `habitat-sim` with `--audio` flag. Audio support requires at least v0.2.2 of `habitat-sim`. 

## Conda environemnt

```bash
conda create -n ss-hab python=3.7 cmake=3.14.0 -y
conda activate ss-hab
pip install pytest-xdist
```

## Habitat-lab

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

## Habitat-sim

According to SS's docs, using the sound simulator requires building with `--audio` flag for sound support.

```bash
# Makes sure all system level deps are installed.
sudo apt-get update || True
sudo apt-get install -y --no-install-recommends libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev # Ubuntu
# Clone habitat-sim repository
git clone --branch stable https://github.com/facebookresearch/habitat-sim.git # Current @ 011191f65f37587f5a5452a93d840b5684593a00
cd habitat-sim
# Build habitat-sim with audio and headless support
python setup.py install --with-cuda --bullet --audio --headless # compilation goes brrrr...
pip install hypothesis # For the tests mainly
```

[Additional building instructions](https://github.com/facebookresearch/habitat-sim/blob/main/BUILD_FROM_SOURCE.md)

### Preliminary testing

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

### Acquiring datasets necessary for simulations

__Getting the ReplicaCAD dataset__

```bash
python src_python/habitat_sim/utils/datasets_download.py --uids replica_cad_dataset
```

Running the physics interaction simulation:
```bash
python examples/viewer.py --dataset data/replica_cad/replicaCAD.scene_dataset_config.json --scene apt_1
```

__Getting the HM3D dataset__

```bash
python src_python/habitat_sim/utils/datasets_download.py --username USERNAME --password PASSWORD --uids hm3d_minival
python src_python/habitat_sim/utils/datasets_download.py --username USERNAME --password PASSWORD --uids hm3d_minival_habitat
```

__TODO / Notes__
- [ ] Download the scene_dataset that seems to be indispensable for simulation too. Also, does it go to `habitat-lab/data` or `habitat-sim/data` ?
- [ ] Better instructions on what to do to download the datasets.
- What exactly is needed to compile `--with-cuda` ? It did not work on `zhora`, but worked on `pris`.
- To get `tests/test_controls.py` to pass, need to `pip install hypothesis`
- When habitat-sim was manually compiled, it seems that commands such as `python -m habitat_sim.utils.datasets_download` do not work. Instead use `cd /path/to/habitat-sim && python src_python/utils/dowload_datasets.py`. This allegedly does not happen if `habitat-sim` was installed through `conda`.


To check that both `habitat-lab` and `habitat-sim` work with each other:

```bash
cd ../habitat-lab
python examples/example.py
```

It should say "... ran for 200 steps" or something similar.

## Soundspaces 2.0

```bash
git clone https://github.com/facebookresearch/sound-spaces.git # Currently @ 4e400abaf65c7759a287355386dcd97de2b17e2b
cd sound-spaces
pip install -e .
```

Next, as per the README instructions, either manually download the datasets:

```bash
wget http://dl.fbaipublicfiles.com/SoundSpaces/binaural_rirs.tar && tar xvf binaural_rirs.tar # 867G
wget http://dl.fbaipublicfiles.com/SoundSpaces/metadata.tar.xz && tar xvf metadata.tar.xz # 1M
wget http://dl.fbaipublicfiles.com/SoundSpaces/sounds.tar.xz && tar xvf sounds.tar.xz #13M
wget http://dl.fbaipublicfiles.com/SoundSpaces/datasets.tar.xz && tar xvf datasets.tar.xz #77M
wget http://dl.fbaipublicfiles.com/SoundSpaces/pretrained_weights.tar.xz && tar xvf pretrained_weights.tar.xz
```

or use the provided script to do so:

```bash
# in sound-spaces directory
python scripts/download_data.py --dataset replica
```
## Torch
