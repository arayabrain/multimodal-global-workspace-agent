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
    # for seed in 222 333; do
    #     (sleep 1s && python ppo_av_nav.py \
    #         --exp-name "ppo_av_nav__ss1_depth_spectro" \
    #         --config-path "env_configs/audiogoal_depth_nocont.yaml" \
    #         --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #         --logdir-prefix $LOGDIR_PREFIX \
    #         --seed $seed \
    #     ) & # >& /dev/null &
    # done
    # endregion: Custom PPO; Depth + Spectrogram based task, SS1.0
    
    ## CA Only
    # region: Custom PPO + Perceiver GWT GWWM Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0; Depth + Spectrogram SS1
    # export MASTER_PORT=8748 # Default port is 8738
    # for seed in 111; do
    # for seed in 222; do
    # for seed in 333; do
    #     (sleep 1s && python ppo_av_nav.py \
    #         --exp-name "ppo_av_nav__ss1_depth_spectro__pgwt_gwwm__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0" \
    #         --agent-type "perceiver-gwt-gwwm" \
    #         --pgwt-depth 1 \
    #         --pgwt-num-latents 8 \
    #         --pgwt-latent-dim 64 \
    #         --pgwt-cross-heads 1 \
    #         --pgwt-latent-heads 4 \
    #         --pgwt-use-sa False \
    #         --pgwt-mod-embed 0 \
    #         --config-path "env_configs/audiogoal_depth_nocont.yaml" \
    #         --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #         --logdir-prefix $LOGDIR_PREFIX \
    #         --seed $seed \
    #     ) & # >& /dev/null &
    # done
    # endregion: Custom PPO + Perceiver GWT GWWM Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0; Depth + Spectrogram SS1

    ### Modality embeddings
    # region: Custom PPO + Perceiver GWT GWWM Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0; Depth + Spectrogram SS1
    # export MASTER_PORT=8748 # Default port is 8738
    # for seed in 111; do
    #     (sleep 1s && python ppo_av_nav.py \
    #         --exp-name "ppo_av_nav__ss1_depth_spectro__pgwt_gwwm__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_128" \
    #         --agent-type "perceiver-gwt-gwwm" \
    #         --pgwt-depth 1 \
    #         --pgwt-num-latents 8 \
    #         --pgwt-latent-dim 64 \
    #         --pgwt-cross-heads 1 \
    #         --pgwt-latent-heads 4 \
    #         --pgwt-use-sa False \
    #         --pgwt-mod-embed 128 \
    #         --config-path "env_configs/audiogoal_depth_nocont.yaml" \
    #         --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #         --logdir-prefix $LOGDIR_PREFIX \
    #         --seed $seed \
    #     ) & # >& /dev/null &
    # done
    # endregion: Custom PPO + Perceiver GWT GWWM Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0; Depth + Spectrogram SS1

    ### Pass previous latent with the audio and vision modalities
    # region: Custom PPO + Perceiver GWT GWWM Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents; Depth + Spectrogram SS1
    # export MASTER_PORT=8748 # Default port is 8738
    for seed in 111 222 333; do
        (sleep 1s && python ppo_av_nav.py \
            --exp-name "ppo_av_nav__ss1_depth_spectro__pgwt_gwwm__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0" \
            --agent-type "perceiver-gwt-gwwm" \
            --pgwt-depth 1 \
            --pgwt-num-latents 8 \
            --pgwt-latent-dim 64 \
            --pgwt-cross-heads 1 \
            --pgwt-latent-heads 4 \
            --pgwt-use-sa False \
            --pgwt-mod-embed 0 \
            --pgwt-ca-prev-latents True \
            --config-path "env_configs/audiogoal_depth_nocont.yaml" \
            --wandb --wandb-project ss-hab --wandb-entity dosssman \
            --logdir-prefix $LOGDIR_PREFIX \
            --seed $seed \
        ) & # >& /dev/null &
    done
    # endregion: Custom PPO + Perceiver GWT GWWM Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents; Depth + Spectrogram SS1


    # region: Custom PPO + Perceiver GWT GWWM Basic Arch. WithSA Cross Heads 1 SA Heads 4 mod_emb 0; Depth + Spectrogram SS1
    # export MASTER_PORT=8748 # Default port is 8738
    # for seed in 111; do
    # for seed in 222 333; do
    #   (sleep 1s && python ppo_av_nav.py \
    #       --exp-name "ppo_av_nav__ss1_depth_spectro__pgwt_gwwm__dpth_1_nlats_8_latdim_64_WithSA_CAskipq_CAnheads_1_SAnheads_4_modembed_0" \
    #       --agent-type "perceiver-gwt-gwwm" \
    #       --pgwt-depth 1 \
    #       --pgwt-num-latents 8 \
    #       --pgwt-latent-dim 64 \
    #       --pgwt-cross-heads 1 \
    #       --pgwt-latent-heads 4 \
    #       --pgwt-use-sa True \
    #       --pgwt-mod-embed 0 \
    #       --config-path "env_configs/audiogoal_depth_nocont.yaml" \
    #       --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #       --logdir-prefix $LOGDIR_PREFIX \
    #       --seed $seed \
    #   ) & # >& /dev/null &
    # done
    # endregion: Custom PPO + Perceiver GWT GWWM Basic Arch. WithSA Cross Heads 1 SA Heads 4 mod_emb 0; Depth + Spectrogram SS1

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

    ### SAVi ###

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