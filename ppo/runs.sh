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
    # for seed in 222; do
    #     (sleep 1s && python ppo_av_nav.py \
    #         --exp-name "ppo_av_nav__depth__rnn_hid_fix" \
    #         --config-path "env_configs/audiogoal_depth.yaml" \
    #         --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #         --logdir-prefix $LOGDIR_PREFIX \
    #         --seed $seed \
    #     ) & # >& /dev/null &
    # done
    # endregion: Depth based task

    # region: Depth based task, with Deep Etho agent structure
    # for seed in 111; do
    #     (sleep 1s && python ppo_av_nav.py \
    #         --exp-name "ppo_av_nav__depth__deep_etho" \
    #         --config-path "env_configs/audiogoal_depth.yaml" \
    #         --agent-type "deep-etho" \
    #         --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #         --logdir-prefix $LOGDIR_PREFIX \
    #         --seed $seed \
    #     ) & # >& /dev/null &
    # done
    # endregion: Depth based task, with Deep Etho agent structure

    # # region: SAVi Env: Depth based task
    # for seed in 111; do
    #     (sleep 1s && python ppo_savi.py \
    #         --exp-name "ppo_savi_continuous" \
    #         --total-steps 10000000 \
    #         --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #         --logdir-prefix $LOGDIR_PREFIX \
    #         --seed $seed
    #     ) # & # >& /dev/null &
    # done
    # # endregion: SAVi Env: Depth based task

    # region: Depth based task, with waveform audio and AudioCLIP's audio encode based RIR Audio feature extractor
    # for seed in 111; do
    #   (sleep 1s && python ppo_av_nav_audioclip.py \
    #       --num-envs 2 --num-steps 60 \
    #       --exp-name "ppo_av_nav_depth_audioclip__audioenc_pretrained" \
    #       --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #       --logdir-prefix $LOGDIR_PREFIX \
    #       --seed $seed \
    #   ) & # >& /dev/null &
    # done
    # endregion: Depth based task, with waveform audio and AudioCLIP's audio encode based RIR Audio feature extractor
# endregion: Custom PPO based on ss_baselines

# Clean up env vars
export CUDA_VISIBLE_DEVICES=
export LOGDIR_PREFIX=
export WANDB_DIR=