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
    
    # region: Custom PPO; RGB + Spectrogram based task, SS1.0
    # for seed in 111; do
    # for seed in 222 333; do
    #   (sleep 1s && python ppo_av_nav.py \
    #       --exp-name "ppo_av_nav__ss1_rgb_spectro" \
    #       --config-path "env_configs/audiogoal_rgb_nocont.yaml" \
    #       --save-videos True \
    #       --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #       --logdir-prefix $LOGDIR_PREFIX \
    #       --seed $seed \
    #   ) & # >& /dev/null &
    # done
    # endregion: Custom PPO; RGB + Spectrogram based task, SS1.0
  
    # region: Custom PPO; Blind + Spectrogram based task, SS1.0
    # for seed in 111 222 333; do
    #   (sleep 1s && python ppo_av_nav.py \
    #       --exp-name "ppo_av_nav__ss1_blind_spectro" \
    #       --config-path "env_configs/audiogoal_blind_nocont.yaml" \
    #       --save-videos False \
    #       --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #       --logdir-prefix $LOGDIR_PREFIX \
    #       --seed $seed \
    #   ) & # >& /dev/null &
    # done
    # endregion: Custom PPO; Blind + Spectrogram based task, SS1.0
    
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
    # for seed in 111 222; do
    # for seed in 333; do
    #     (sleep 1s && python ppo_av_nav.py \
    #         --exp-name "ppo_av_nav__ss1_depth_spectro__pgwt_gwwm__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats" \
    #         --agent-type "perceiver-gwt-gwwm" \
    #         --pgwt-depth 1 \
    #         --pgwt-num-latents 8 \
    #         --pgwt-latent-dim 64 \
    #         --pgwt-cross-heads 1 \
    #         --pgwt-latent-heads 4 \
    #         --pgwt-use-sa False \
    #         --pgwt-mod-embed 0 \
    #         --pgwt-ca-prev-latents True \
    #         --config-path "env_configs/audiogoal_depth_nocont.yaml" \
    #         --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #         --logdir-prefix $LOGDIR_PREFIX \
    #         --seed $seed \
    #     ) & # >& /dev/null &
    # done
    # endregion: Custom PPO + Perceiver GWT GWWM Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents; Depth + Spectrogram SS1

    # region: Custom PPO + Perceiver GWT GWWM Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents; RGB + Spectrogram SS1
    # export MASTER_PORT=8748 # Default port is 8738
    # for seed in 222 333; do
    #     (sleep 1s && python ppo_av_nav.py \
    #         --exp-name "ppo_av_nav__ss1_rgb_spectro__pgwt_gwwm__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats" \
    #         --agent-type "perceiver-gwt-gwwm" \
    #         --pgwt-depth 1 \
    #         --pgwt-num-latents 8 \
    #         --pgwt-latent-dim 64 \
    #         --pgwt-cross-heads 1 \
    #         --pgwt-latent-heads 4 \
    #         --pgwt-use-sa False \
    #         --pgwt-mod-embed 0 \
    #         --pgwt-ca-prev-latents True \
    #         --config-path "env_configs/audiogoal_rgb_nocont.yaml" \
    #         --save-videos True \
    #         --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #         --logdir-prefix $LOGDIR_PREFIX \
    #         --seed $seed \
    #     ) & # >& /dev/null &
    # done
    # endregion: Custom PPO + Perceiver GWT GWWM Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents; RGB + Spectrogram SS1

    ## CA SA
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

    # region: Custom PPO + Perceiver GWT GWWM Basic Arch. WithSA Cross Heads 1 SA Heads 4 mod_emb 0; RGB + Spectrogram SS1
    # export MASTER_PORT=8748 # Default port is 8738
    # for seed in 111 222 333; do
    #   (sleep 1s && python ppo_av_nav.py \
    #       --exp-name "ppo_av_nav__ss1_rgb_spectro__pgwt_gwwm__dpth_1_nlats_8_latdim_64_WithSA_CAskipq_CAnheads_1_SAnheads_4_modembed_0" \
    #       --agent-type "perceiver-gwt-gwwm" \
    #       --pgwt-depth 1 \
    #       --pgwt-num-latents 8 \
    #       --pgwt-latent-dim 64 \
    #       --pgwt-cross-heads 1 \
    #       --pgwt-latent-heads 4 \
    #       --pgwt-use-sa True \
    #       --pgwt-mod-embed 0 \
    #       --config-path "env_configs/audiogoal_rgb_nocont.yaml" \
    #       --save-videos True \
    #       --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #       --logdir-prefix $LOGDIR_PREFIX \
    #       --seed $seed \
    #   ) & # >& /dev/null &
    # done
    # endregion: Custom PPO + Perceiver GWT GWWM Basic Arch. WithSA Cross Heads 1 SA Heads 4 mod_emb 0; RGB + Spectrogram SS1

    ### Pass previous latent with the audio and vision modalities
    # region: Custom PPO + Perceiver GWT GWWM Basic Arch. WithSA Cross Heads 1 SA Heads 4 mod_emb 0; Depth + Spectrogram SS1
    # export MASTER_PORT=8748 # Default port is 8738
    # for seed in 111; do
    # for seed in 222 333; do
    #   (sleep 1s && python ppo_av_nav.py \
    #       --exp-name "ppo_av_nav__ss1_depth_spectro__pgwt_gwwm__dpth_1_nlats_8_latdim_64_WithSA_CAskipq_CAnheads_1_SAnheads_4_modembed_0_CAprevlats" \
    #       --agent-type "perceiver-gwt-gwwm" \
    #       --pgwt-depth 1 \
    #       --pgwt-num-latents 8 \
    #       --pgwt-latent-dim 64 \
    #       --pgwt-cross-heads 1 \
    #       --pgwt-latent-heads 4 \
    #       --pgwt-use-sa True \
    #       --pgwt-mod-embed 0 \
    #       --pgwt-ca-prev-latents True \
    #       --config-path "env_configs/audiogoal_depth_nocont.yaml" \
    #       --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #       --logdir-prefix $LOGDIR_PREFIX \
    #       --seed $seed \
    #   ) & # >& /dev/null &
    # done
    # endregion: Custom PPO + Perceiver GWT GWWM Basic Arch. WithSA Cross Heads 1 SA Heads 4 mod_emb 0; Depth + Spectrogram SS1

    # region: Custom PPO + Perceiver GWT GWWM Basic Arch. WithSA Cross Heads 1 SA Heads 4 mod_emb 0; Blind + Spectrogram SS1
    # export MASTER_PORT=8748 # Default port is 8738
    # for seed in 111 222 333; do
    #   (sleep 1s && python ppo_av_nav.py \
    #       --exp-name "ppo_av_nav__ss1_blind_spectro__pgwt_gwwm__dpth_1_nlats_8_latdim_64_WithSA_CAskipq_CAnheads_1_SAnheads_4_modembed_0_CAprevlats" \
    #       --agent-type "perceiver-gwt-gwwm" \
    #       --pgwt-depth 1 \
    #       --pgwt-num-latents 8 \
    #       --pgwt-latent-dim 64 \
    #       --pgwt-cross-heads 1 \
    #       --pgwt-latent-heads 4 \
    #       --pgwt-use-sa True \
    #       --pgwt-mod-embed 0 \
    #       --pgwt-ca-prev-latents True \
    #       --config-path "env_configs/audiogoal_blind_nocont.yaml" \
    #       --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #       --logdir-prefix $LOGDIR_PREFIX \
    #       --seed $seed \
    #   ) & # >& /dev/null &
    # done
    # endregion: Custom PPO + Perceiver GWT GWWM Basic Arch. WithSA Cross Heads 1 SA Heads 4 mod_emb 0; Blind + Spectrogram SS1

    ### Various latent dimension combinations
    # region: Custom PPO + Perceiver GWT GWWM Basic Arch. WithSA Cross Heads 1 SA Heads 4 mod_emb 0; Depth + Spectrogram SS1
    # export MASTER_PORT=8748 # Default port is 8738
    # for seed in 222 333; do
    # for seed in 111; do
    #   #- N x L = 1, 512
    #   (sleep 1s && python ppo_av_nav.py \
    #       --exp-name "ppo_av_nav__ss1_depth_spectro__pgwt_gwwm__dpth_1_nlats_1_latdim_512_WithSA_CAskipq_CAnheads_1_SAnheads_4_modembed_0" \
    #       --agent-type "perceiver-gwt-gwwm" \
    #       --pgwt-depth 1 \
    #       --pgwt-num-latents 1 \
    #       --pgwt-latent-dim 512 \
    #       --pgwt-cross-heads 1 \
    #       --pgwt-latent-heads 4 \
    #       --pgwt-use-sa True \
    #       --pgwt-mod-embed 0 \
    #       --config-path "env_configs/audiogoal_depth_nocont.yaml" \
    #       --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #       --logdir-prefix $LOGDIR_PREFIX \
    #       --seed $seed \
    #   ) & # >& /dev/null &

    #   #- N x L = 4, 128
    #   (sleep 1s && python ppo_av_nav.py \
    #       --exp-name "ppo_av_nav__ss1_depth_spectro__pgwt_gwwm__dpth_1_nlats_4_latdim_128_WithSA_CAskipq_CAnheads_1_SAnheads_4_modembed_0" \
    #       --agent-type "perceiver-gwt-gwwm" \
    #       --pgwt-depth 1 \
    #       --pgwt-num-latents 4 \
    #       --pgwt-latent-dim 128 \
    #       --pgwt-cross-heads 1 \
    #       --pgwt-latent-heads 4 \
    #       --pgwt-use-sa True \
    #       --pgwt-mod-embed 0 \
    #       --config-path "env_configs/audiogoal_depth_nocont.yaml" \
    #       --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #       --logdir-prefix $LOGDIR_PREFIX \
    #       --seed $seed \
    #   ) & # >& /dev/null &
      
    #   #- N x L = 16, 32
    #   (sleep 1s && python ppo_av_nav.py \
    #       --exp-name "ppo_av_nav__ss1_depth_spectro__pgwt_gwwm__dpth_1_nlats_16_latdim_32_WithSA_CAskipq_CAnheads_1_SAnheads_4_modembed_0" \
    #       --agent-type "perceiver-gwt-gwwm" \
    #       --pgwt-depth 1 \
    #       --pgwt-num-latents 16 \
    #       --pgwt-latent-dim 32 \
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


    #- SS2 Av Nav -#
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

    # region: SAVi SS1: PPO GRU Default config
    # for seed in 111; do
    #     (sleep 1s && python ppo_av_nav.py \
    #         --config-path "env_configs/savi/savi_ss1.yaml" \
    #         --exp-name "ppo_savi_ss1__ppo_gru" \
    #         --total-steps 10000000 \
    #         --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #         --logdir-prefix $LOGDIR_PREFIX \
    #         --seed $seed
    #     ) & # >& /dev/null &
    # done
    # endregion: SAVi SS1: PPO GRU Default config

    # region: SAVi SS1: PPO PGWT NoSA CAHeads 1 SAHeads 4 ModEmbed 0 CA WithPrevLatents
    # for seed in 111; do
    #   (sleep 1s && python ppo_av_nav.py \
    #     --config-path "env_configs/savi/savi_ss1.yaml" \
    #     --exp-name "ppo_savi_ss1__pgwt_gwwm__N_8_D_64_CA_CAPrevLatents_CAHeads_1_SAHeads_4" \
    #     --total-steps 10000000 \
    #     --agent-type "perceiver-gwt-gwwm" \
    #     --pgwt-depth 1 \
    #     --pgwt-num-latents 8 \
    #     --pgwt-latent-dim 64 \
    #     --pgwt-cross-heads 1 \
    #     --pgwt-latent-heads 4 \
    #     --pgwt-use-sa False \
    #     --pgwt-mod-embed 0 \
    #     --pgwt-ca-prev-latents True \
    #     --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #     --logdir-prefix $LOGDIR_PREFIX \
    #     --seed $seed \
    #   ) & # >& /dev/null &
    # done
    # endregion: SAVi SS1: PPO PGWT NoSA CAHeads 1 SAHeads 4 ModEmbed 0 CA WithPrevLatents

    # region: SAVi SS1: PPO PGWT CA SA CAHeads 1 SAHeads 4 ModEmbed 0 CA
    # for seed in 111; do
    #   (sleep 1s && python ppo_av_nav.py \
    #     --config-path "env_configs/savi/savi_ss1.yaml" \
    #     --exp-name "ppo_savi_ss1__pgwt_gwwm__N_8_D_64_CA_SA_CAHeads_1_SAHeads_4" \
    #     --total-steps 10000000 \
    #     --agent-type "perceiver-gwt-gwwm" \
    #     --pgwt-depth 1 \
    #     --pgwt-num-latents 8 \
    #     --pgwt-latent-dim 64 \
    #     --pgwt-cross-heads 1 \
    #     --pgwt-latent-heads 4 \
    #     --pgwt-use-sa True \
    #     --pgwt-mod-embed 0 \
    #     --pgwt-ca-prev-latents True \
    #     --wandb --wandb-project ss-hab --wandb-entity dosssman \
    #     --logdir-prefix $LOGDIR_PREFIX \
    #     --seed $seed \
    #   ) & # >& /dev/null &
    # done
    # endregion: SAVi SS1: PPO PGWT CA SA CAHeads 1 SAHeads 4 ModEmbed 0 CA WithPrevLatents

# endregion: Custom PPO based on SS tasks

# Clean up env vars
export CUDA_VISIBLE_DEVICES=
export LOGDIR_PREFIX=
export MASTER_PORT=
export WANDB_DIR=