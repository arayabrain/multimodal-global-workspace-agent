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

    # region: Custom PPO + Perceiver GWT Fair architecture Grad Dbg; Depth + Spectrogram based task
    # export MASTER_PORT=8748 # Default port is 8738
    # for seed in 111; do
    #     (sleep 1s && python ppo_av_nav.py \
    #         --exp-name "ppo_av_nav__depth_spectro__pgwt__dpth_1_nlats_8_latdim_64__blowup_dbg" \
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
    # endregion: Custom PPO + Perceiver GWT Fair architecture Grad Dbg; Depth + Spectrogram based task
    
    # region: Custom PPO + Perceiver GWT Fair architecture Grad Dbg Num envs 5 with FF encodeing; Depth + Spectrogram based task
    # export MASTER_PORT=8748 # Default port is 8738
    for seed in 111; do
        (sleep 1s && python ppo_av_nav.py \
            --exp-name "ppo_av_nav__depth_spectro__pgwt__dpth_1_nlats_8_latdim_64__ff_maxfreq_10_num_freq_6__blowup_dbg" \
            --agent-type "perceiver-gwt" \
            --pgwt-depth 1 \
            --pgwt-num-latents 8 \
            --pgwt-latent-dim 64 \
            --pgwt-ff True \
            --config-path "env_configs/audiogoal_depth.yaml" \
            --wandb --wandb-project ss-hab --wandb-entity dosssman \
            --logdir-prefix $LOGDIR_PREFIX \
            --seed $seed \
        ) & # >& /dev/null &
    done
    # endregion: Custom PPO + Perceiver GWT Fair architecture Grad Dbg Num envs 5 with FF encodeing; Depth + Spectrogram based task

    # region: Custom PPO + Perceiver GWT Fair architecture Grad Dbg Grad Norm 10.0  Depth + Spectrogram based task
    # export MASTER_PORT=8748 # Default port is 8738
    # for seed in 111; do
    #     (sleep 1s && python ppo_av_nav.py \
    #         --exp-name "ppo_av_nav__depth_spectro__pgwt__dpth_1_nlats_8_latdim_64__lr_1e-4_wd_2e-5_maxgradnorm_10" \
    #         --agent-type "perceiver-gwt" \
    #         --pgwt-depth 1 \
    #         --pgwt-num-latents 8 \
    #         --pgwt-latent-dim 64 \
    #         --config-path "env_configs/audiogoal_depth.yaml" \
    #         --lr 0.0001 --optim-wd 0.00002 \
    #         --max-grad-norm 10 \
    #         --log-training-stats-every 1 \
    #         --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #         --logdir-prefix $LOGDIR_PREFIX \
    #         --seed $seed \
    #     ) & # >& /dev/null &
    # done
    # endregion: Custom PPO + Perceiver GWT Fair architecture Grad Dbg Grad Norm 10.0  Depth + Spectrogram based task

    # region: Custom PPO + Perceiver GWT Fair architecture Zero Init Learned; Depth + Spectrogram based task
    # export MASTER_PORT=8748 # Default port is 8738
    # for seed in 111; do
    #     (sleep 1s && python ppo_av_nav.py \
    #         --exp-name "ppo_av_nav__depth_spectro__pgwt__dpth_1_nlats_8_latdim_64_lat_zero_learned" \
    #         --agent-type "perceiver-gwt" \
    #         --pgwt-latent-type "zeros" \
    #         --pgwt-latent-learned True \
    #         --pgwt-depth 1 \
    #         --pgwt-num-latents 8 \
    #         --pgwt-latent-dim 64 \
    #         --config-path "env_configs/audiogoal_depth.yaml" \
    #         --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #         --logdir-prefix $LOGDIR_PREFIX \
    #         --seed $seed \
    #     ) & # >& /dev/null &
    # done
    # endregion: Custom PPO + Perceiver GWT Fair architecture Zero Init Non Learned; Depth + Spectrogram based task

    # region: Custom PPO + Perceiver GWT Fair architecture Zero Init Learned No SA; Depth + Spectrogram based task
    # export MASTER_PORT=8748 # Default port is 8738
    # for seed in 111; do
    #     (sleep 1s && python ppo_av_nav.py \
    #         --exp-name "ppo_av_nav__depth_spectro__pgwt__dpth_1_nlats_8_latdim_64_lat_zero_learned__nenvs_5" \
    #         --agent-type "perceiver-gwt" \
    #         --num-envs 5 \
    #         --pgwt-latent-type "zeros" \
    #         --pgwt-latent-learned True \
    #         --pgwt-depth 1 \
    #         --pgwt-num-latents 8 \
    #         --pgwt-latent-dim 64 \
    #         --pgwt-use-sa False \
    #         --config-path "env_configs/audiogoal_depth.yaml" \
    #         --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #         --logdir-prefix $LOGDIR_PREFIX \
    #         --seed $seed \
    #     ) & # >& /dev/null &
    # done
    # endregion: Custom PPO + Perceiver GWT Fair architecture Zero Init Learned No SA; Depth + Spectrogram based task

    # region: Custom PPO + Perceiver GWT Fair architecture Zero Init Learned; Shorter horizon; Depth + Spectrogram based task
    # export MASTER_PORT=8748 # Default port is 8738
    # for seed in 111; do
    #     (sleep 1s && python ppo_av_nav.py \
    #         --exp-name "ppo_av_nav__depth_spectro__pgwt__dpth_1_nlats_8_latdim_64_lat_zero_learned__nenvs_5_nsteps_30" \
    #         --agent-type "perceiver-gwt" \
    #         --num-envs 5 \
    #         --num-steps 30 \
    #         --pgwt-latent-type "zeros" \
    #         --pgwt-latent-learned True \
    #         --pgwt-depth 1 \
    #         --pgwt-num-latents 8 \
    #         --pgwt-latent-dim 64 \
    #         --config-path "env_configs/audiogoal_depth.yaml" \
    #         --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #         --logdir-prefix $LOGDIR_PREFIX \
    #         --seed $seed \
    #     ) & # >& /dev/null &
    # done
    # endregion: Custom PPO + Perceiver GWT Fair architecture Zero Init Learned; Shorter horizon; Depth + Spectrogram based task

    # region: Custom PPO + Perceiver GWT GWWM Fair architecture; Depth + Spectrogram based task
    # export MASTER_PORT=8748 # Default port is 8738
    # for seed in 111; do
    #     (sleep 1s && python ppo_av_nav.py \
    #         --exp-name "ppo_av_nav__depth_spectro__pgwt_gwwm__nlats_8_latdim_64" \
    #         --agent-type "perceiver-gwt-gwwm" \
    #         --pgwt-num-latents 8 \
    #         --pgwt-latent-dim 64 \
    #         --config-path "env_configs/audiogoal_depth.yaml" \
    #         --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #         --logdir-prefix $LOGDIR_PREFIX \
    #         --seed $seed \
    #     ) & # >& /dev/null &
    # done
    # endregion: Custom PPO + Perceiver GWT GWWM Fair architecture; Depth + Spectrogram based task

    # region: Custom PPO + Perceiver GWT Fair architecture Zero Init Non Learned; Depth + Spectrogram based task
    # export MASTER_PORT=8748 # Default port is 8738
    # for seed in 111; do
    #     (sleep 1s && python ppo_av_nav.py \
    #         --exp-name "ppo_av_nav__depth_spectro__pgwt__dpth_1_nlats_8_latdim_64_lat_zero_nolearn" \
    #         --agent-type "perceiver-gwt" \
    #         --pgwt-latent-type "zeros" \
    #         --pgwt-latent-learned False \
    #         --pgwt-depth 1 \
    #         --pgwt-num-latents 8 \
    #         --pgwt-latent-dim 64 \
    #         --config-path "env_configs/audiogoal_depth.yaml" \
    #         --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #         --logdir-prefix $LOGDIR_PREFIX \
    #         --seed $seed \
    #     ) & # >& /dev/null &
    # done
    # endregion: Custom PPO + Perceiver GWT Fair architecture Zero Init Non Learned; Depth + Spectrogram based task
    
    # region: Custom PPO + Perceiver GWT Fair architecture CA head 4 SA head 4; Depth + Spectrogram based task
    # export MASTER_PORT=8748 # Default port is 8738
    # for seed in 111; do
    #     (sleep 1s && python ppo_av_nav.py \
    #         --exp-name "ppo_av_nav__depth_spectro__pgwt__dpth_1_nlats_8_latdim_64_cross_heads_4_self_heads_4" \
    #         --agent-type "perceiver-gwt" \
    #         --pgwt-depth 1 \
    #         --pgwt-num-latents 8 \
    #         --pgwt-latent-dim 64 \
    #         --pgwt-cross-heads 4 \
    #         --pgwt-latent-heads 4 \
    #         --config-path "env_configs/audiogoal_depth.yaml" \
    #         --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #         --logdir-prefix $LOGDIR_PREFIX \
    #         --seed $seed \
    #     ) & # >& /dev/null &
    # done
    # endregion: Custom PPO + Perceiver GWT Fair architecture; Depth + Spectrogram based task

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