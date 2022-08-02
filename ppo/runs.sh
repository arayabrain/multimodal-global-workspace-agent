#!/bin/bash
# NUM_CORES=$(nproc --all)
# export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES

export LOGDIR_PREFIX=~/random/rl/exp-logs/ss-hab
if [ ! -d $LOGDIR_PREFIX ]; then
  mkdir -p $LOGDIR_PREFIX
fi

export WANDB_DIR="$LOGDIR_PREFIX""_wandb"
if [ ! -d $WANDB_DIR ]; then
  mkdir -p $WANDB_DIR
fi

# region: Custom PPO based on ss_baselines
    # region: RGB based task
    # for seed in 111; do
    #     (sleep 1s && python ppo_av_nav.py \
    #         --exp-name "ppo_av_nav__rgb" \
    #         --num-envs 8 \
    #         --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #         --logdir-prefix $LOGDIR_PREFIX \
    #         --seed $seed \
    #     ) & # >& /dev/null &
    # done
    # endregion: RGB based task

    # region: Depth based task
    # for seed in 111; do
    #     (sleep 1s && python ppo_av_nav.py \
    #         --exp-name "ppo_av_nav__depth__rnn_hid_fix__num_envs_8" \
    #         --config-path "env_configs/audigoal_depth.yaml" \
    #         --num-envs 8 \
    #         --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #         --logdir-prefix $LOGDIR_PREFIX \
    #         --seed $seed \
    #     ) & # >& /dev/null &
    # done
    # endregion: Depth based task
# endregion: Custom PPO based on ss_baselines

# Clean up env vars
export CUDA_VISIBLE_DEVICES=
export LOGDIR_PREFIX=
export WANDB_DIR=