#!/bin/bash
NUM_CORES=$(nproc --all)
export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES

export LOGDIR_PREFIX=~/random/rl/exp-logs/ss-hab-bc
if [ ! -d $LOGDIR_PREFIX ]; then
  mkdir -p $LOGDIR_PREFIX
fi

export WANDB_DIR="$LOGDIR_PREFIX""_wandb"
if [ ! -d $WANDB_DIR ]; then
  mkdir -p $WANDB_DIR
fi

# region: PPO BC #

  # region: PPO GRU - BC with default hyparams
  for seed in 111 222 333; do
    # export MASTER_PORT=8738 # Default port is 8738
    export TOTAL_STEPS=5000000
    (sleep 1s && python ppo_bc.py \
        --exp-name "ppo_bc__savi_ss1__rgb_depth_spectro__gru" \
        --config-path "env_configs/savi/savi_ss1.yaml" \
        --wandb --wandb-project ss-hab-bc --wandb-entity dosssman \
        --logdir-prefix $LOGDIR_PREFIX \
        --total-steps $TOTAL_STEPS \
        --seed $seed \
    ) & # >& /dev/null &
  done
  # endregion: PPO GRU - BC with default hyparams

  # region: Custom PPO + Perceiver GWT GWWM Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents; RGB + Spectrogram SS1
  for seed in 111 222 333; do
    # export MASTER_PORT=8748 # Default port is 8738
    export TOTAL_STEPS=5000000
    (sleep 1s && python ppo_bc.py \
        --exp-name "ppo_bc__savi_ss1__rgb_depth_spectro__pgwt__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats" \
        --agent-type "perceiver-gwt-gwwm" \
        --pgwt-depth 1 \
        --pgwt-num-latents 8 \
        --pgwt-latent-dim 64 \
        --pgwt-cross-heads 1 \
        --pgwt-latent-heads 4 \
        --pgwt-use-sa False \
        --pgwt-mod-embed 0 \
        --pgwt-ca-prev-latents True \
        --config-path "env_configs/savi/savi_ss1.yaml" \
        --save-videos True \
        --wandb --wandb-project ss-hab-bc --wandb-entity dosssman \
        --total-steps $TOTAL_STEPS \
        --logdir-prefix $LOGDIR_PREFIX \
        --seed $seed \
    ) & # >& /dev/null &
  done
  # endregion: Custom PPO + Perceiver GWT GWWM Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents; RGB + Spectrogram SS1

# endregion: PPO BC #

# Clean up env vars
export CUDA_VISIBLE_DEVICES=
export LOGDIR_PREFIX=
export MASTER_PORT=
export WANDB_DIR=