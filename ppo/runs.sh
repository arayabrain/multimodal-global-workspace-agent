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

# region: Custom PPO based on SS tasks
    # region: Custom PPO; Depth + Spectrogram based task, SS1.0
    # export MASTER_PORT=8738 # Default port is 8738
    # for seed in 111; do
    #     (sleep 1s && python ppo_av_nav.py \
    #         --exp-name "ppo_av_nav__ss1_depth_spectro" \
    #         --config-path "env_configs/audiogoal_depth_nocont.yaml" \
    #         --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #         --logdir-prefix $LOGDIR_PREFIX \
    #         --seed $seed \
    #     ) & # >& /dev/null &
    # done
    # endregion: Custom PPO; Depth + Spectrogram based task, SS1.0
    
    # region: Custom PPO + Perceiver GWT Fair architecture NoSA No FF No Embed; Depth + Spectrogram SS1
    # export MASTER_PORT=8748 # Default port is 8738
    # for seed in 111; do
    #     (sleep 1s && python ppo_av_nav.py \
    #         --exp-name "ppo_av_nav__ss1_depth_spectro__pgwt__dpth_1_nlats_8_latdim_64_noSA__noff__modembed_0" \
    #         --agent-type "perceiver-gwt" \
    #         --pgwt-depth 1 \
    #         --pgwt-num-latents 8 \
    #         --pgwt-latent-dim 64 \
    #         --pgwt-ff False \
    #         --config-path "env_configs/audiogoal_depth_nocont.yaml" \
    #         --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #         --logdir-prefix $LOGDIR_PREFIX \
    #         --seed $seed \
    #     ) & # >& /dev/null &
    # done
    # endregion: Custom PPO + Perceiver GWT Fair architecture NoSA No FF No Embed; Depth + Spectrogram SS1

    # region: Custom PPO + Perceiver GWT Fair architecture NoSA No FF Mod Embed 128; Depth + Spectrogram SS1
    # export MASTER_PORT=8748 # Default port is 8738
    # for seed in 111; do
    #     (sleep 1s && python ppo_av_nav.py \
    #         --exp-name "ppo_av_nav__ss1_depth_spectro__pgwt__dpth_1_nlats_8_latdim_64_noSA__noff__modembed_128" \
    #         --agent-type "perceiver-gwt" \
    #         --pgwt-depth 1 \
    #         --pgwt-num-latents 8 \
    #         --pgwt-latent-dim 64 \
    #         --pgwt-ff False \
    #         --pgwt-mod-embed 128 \
    #         --config-path "env_configs/audiogoal_depth_nocont.yaml" \
    #         --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #         --logdir-prefix $LOGDIR_PREFIX \
    #         --seed $seed \
    #     ) & # >& /dev/null &
    # done
    # region: Custom PPO + Perceiver GWT Fair architecture NoSA No FF Mod Embed 128; Depth + Spectrogram SS1

    # region: Custom PPO + Perceiver GWT Fair architecture but Thicker Attention layers NoSA No FF Mod Embed 128; Depth + Spectrogram SS1
    # export MASTER_PORT=8748 # Default port is 8738
    # for seed in 111; do
    #     (sleep 1s && python ppo_av_nav.py \
    #         --exp-name "ppo_av_nav__ss1_depth_spectro__pgwt__dpth_1_nlats_8_latdim_64_CAdim_128_SAdim_128_noSA__noff__modembed_0" \
    #         --agent-type "perceiver-gwt" \
    #         --pgwt-depth 1 \
    #         --pgwt-num-latents 8 \
    #         --pgwt-latent-dim 64 \
    #         --pgwt-cross-dim-head 128 \
    #         --pgwt-latent-dim-head 128 \
    #         --pgwt-ff False \
    #         --pgwt-mod-embed 0 \
    #         --config-path "env_configs/audiogoal_depth_nocont.yaml" \
    #         --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #         --logdir-prefix $LOGDIR_PREFIX \
    #         --seed $seed \
    #     ) & # >& /dev/null &
    # done
    # endregion: Custom PPO + Perceiver GWT Fair architecture but Thicker Attention layers NoSA No FF Mod Embed 128; Depth + Spectrogram SS1

    # region: Custom PPO + Perceiver GWT Fair architecture With FF Perceiver Depth 6; Depth + Spectrogram SS1
    # export MASTER_PORT=8748 # Default port is 8738
    # for seed in 111; do
    #     (sleep 1s && python ppo_av_nav.py \
    #         --exp-name "ppo_av_nav__ss1_depth_spectro__pgwt__dpth_6_nlats_8_latdim_64__ff_maxfreq_10_numfreq_6" \
    #         --agent-type "perceiver-gwt" \
    #         --pgwt-depth 6 \
    #         --pgwt-num-latents 8 \
    #         --pgwt-latent-dim 64 \
    #         --pgwt-ff True \
    #         --config-path "env_configs/audiogoal_depth_nocont.yaml" \
    #         --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #         --logdir-prefix $LOGDIR_PREFIX \
    #         --seed $seed \
    #     ) & # >& /dev/null &
    # done
    # endregion: Custom PPO + Perceiver GWT Fair architecture With FF Perceiver Depth 6; Depth + Spectrogram SS1
    
    # region: Custom PPO; Depth + Spectrogram based task SS2
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
    # endregion: Custom PPO; Depth + Spectrogram based task SS2

    # region: Custom PPO + Perceiver GWT Fair architecture No FF; Depth + Spectrogram based task
    # export MASTER_PORT=8748 # Default port is 8738
    # for seed in 111; do
    #     (sleep 1s && python ppo_av_nav.py \
    #         --exp-name "ppo_av_nav__depth_spectro__pgwt__dpth_1_nlats_8_latdim_64__noff" \
    #         --agent-type "perceiver-gwt" \
    #         --pgwt-depth 1 \
    #         --pgwt-num-latents 8 \
    #         --pgwt-latent-dim 64 \
    #         --pgwt-ff False \
    #         --config-path "env_configs/audiogoal_depth.yaml" \
    #         --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #         --logdir-prefix $LOGDIR_PREFIX \
    #         --seed $seed \
    #     ) & # >& /dev/null &
    # done
    # endregion: Custom PPO + Perceiver GWT Fair architecture No FF; Depth + Spectrogram based task

    # region: Custom PPO + Perceiver GWT Fair architecture With FF No SA; Depth + Spectrogram based task
    # export MASTER_PORT=8748 # Default port is 8738
    # for seed in 111; do
    #     (sleep 1s && python ppo_av_nav.py \
    #         --exp-name "ppo_av_nav__depth_spectro__pgwt__dpth_1_nlats_8_latdim_64__ff_maxfreq_10_numfreq_6_noSA" \
    #         --agent-type "perceiver-gwt" \
    #         --pgwt-depth 1 \
    #         --pgwt-num-latents 8 \
    #         --pgwt-latent-dim 64 \
    #         --pgwt-ff True \
    #         --pgwt-use-sa False \
    #         --config-path "env_configs/audiogoal_depth.yaml" \
    #         --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #         --logdir-prefix $LOGDIR_PREFIX \
    #         --seed $seed \
    #     ) & # >& /dev/null &
    # done
    # endregion: Custom PPO + Perceiver GWT Fair architecture With FF No SA; Depth + Spectrogram based task




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

# endregion: Custom PPO based on SS tasks

# Clean up env vars
export CUDA_VISIBLE_DEVICES=
export LOGDIR_PREFIX=
export MASTER_PORT=
export WANDB_DIR=