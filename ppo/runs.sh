#!/bin/bash
NUM_CORES=$(nproc --all)
export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES

export LOGDIR_PREFIX=~/random/rl/exp-logs/ss-hab
if [ ! -d $LOGDIR_PREFIX ]; then
  mkdir -p $LOGDIR_PREFIX
fi

export WANDB_DIR="$LOGDIR_PREFIX""_wandb"
if [ ! -d $WANDB_DIR ]; then
  mkdir -p $WANDB_DIR
fi

# region: Custom PPO based on ss_baselinesg
    # region: Custom PPO; Depth + Spectrogram based task
    # export MASTER_PORT=8738 # Default port is 8738
    # for seed in 111; do
    #     (sleep 1s && python ppo_av_nav.py \
    #         --exp-name "ppo_av_nav__depth_spectro" \
    #         --config-path "env_configs/audiogoal_depth.yaml" \
    #         --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #         --logdir-prefix $LOGDIR_PREFIX \
    #         --seed $seed \
    #     ) & # >& /dev/null &
    # done
    # endregion: Custom PPO; Depth + Spectrogram based task

    # region: Custom PPO + Perceiver GWT Fair architecture; Depth + Spectrogram based task
    # export MASTER_PORT=8748 # Default port is 8738
    # for seed in 111; do
    #     (sleep 1s && python ppo_av_nav.py \
    #         --exp-name "ppo_av_nav__depth_spectro__pgwt__dpth_1_nlats_8_latdim_64" \
    #         --agent-type "perceiver-gwt" \
    #         --pgwt-depth 1 \
    #         --pgwt-num-latents 8 \
    #         --pgwt-latent-dim 64 \
    #         --config-path "env_configs/audiogoal_depth.yaml" \
    #         --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #         --logdir-prefix $LOGDIR_PREFIX \
    #         --seed $seed \
    #     ) & # >& /dev/null &
    # done
    # region: Custom PPO + Perceiver GWT Fair architecture; Depth + Spectrogram based task

    # region: Custom PPO + Perceiver GWT; Depth + Spectrogram based task
    # export MASTER_PORT=8748 # Default port is 8738
    # for seed in 111; do
    #     (sleep 1s && python ppo_av_nav.py \
    #         --exp-name "ppo_av_nav__depth_spectro__pgwt" \
    #         --agent-type "perceiver-gwt" \
    #         --config-path "env_configs/audiogoal_depth.yaml" \
    #         --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #         --logdir-prefix $LOGDIR_PREFIX \
    #         --seed $seed \
    #     ) & # >& /dev/null &
    # done
    # endregion: Custom PPO + Perceiver GWT; Depth + Spectrogram based task

    # region: Custom PPO + Perceiver GWT; Depth + Spectrogram based task
    # export MASTER_PORT=8758 # Default port is 8738
    # for seed in 111; do
    #     (sleep 1s && python ppo_av_nav.py \
    #         --exp-name "ppo_av_nav__depth_spectro__piogwt" \
    #         --agent-type "perceiverio-gwt" \
    #         --config-path "env_configs/audiogoal_depth.yaml" \
    #         --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #         --logdir-prefix $LOGDIR_PREFIX \
    #         --seed $seed \
    #     ) & # >& /dev/null &
    # done
    # endregion: Custom PPO + Perceiver GWT; Depth + Spectrogram based task

    # region: SAVi Env: Depth based task
    # for seed in 111; do
    #     (sleep 1s && python ppo_savi.py \
    #         --exp-name "ppo_savi_continuous" \
    #         --total-steps 10000000 \
    #         --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #         --logdir-prefix $LOGDIR_PREFIX \
    #         --seed $seed
    #     ) # & # >& /dev/null &
    # done
    # endregion: SAVi Env: Depth based task
# endregion: Custom PPO based on ss_baselines

# Clean up env vars
export CUDA_VISIBLE_DEVICES=
export LOGDIR_PREFIX=
export MASTER_PORT=
export WANDB_DIR=