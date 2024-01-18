#!/bin/bash
NUM_CORES=$(nproc --all)
export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES

export WANDB_PROJECT="ss-hab-bc-revised"

export LOGDIR_PREFIX="~/random/rl/exp-logs/${WANDB_PROJECT}"
if [ ! -d $LOGDIR_PREFIX ]; then
  mkdir -p $LOGDIR_PREFIX
fi

export WANDB_DIR="$LOGDIR_PREFIX""_wandb"
if [ ! -d $WANDB_DIR ]; then
  mkdir -p $WANDB_DIR
fi

# echo $PATH
# TODO: Fix the issue that requires this kind of hardcoding
# export LD_LIBRARY_PATH="/usr/local/cudnn-8.4.1_cuda_11.x:/usr/local/cuda-11.7/lib64:"
# echo "${LD_LIBRARY_PATH}"

# region: GRU

  # region: PPO GRU - BC | gw-size: 64
  # for seed in 111 222; do
  #   export TOTAL_STEPS=20000000
  #   (sleep 1s && python ppo_bc.py \
  #     --exp-name "ppo_bc__gru__h_64" \
  #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
  #     --ent-coef 0.2  \
  #     --gw-size 64 \
  #     --agent-type "gru" \
  #     --save-videos False \
  #     --wandb --wandb-project $WANDB_PROJECT --wandb-entity dosssman \
  #     --logdir-prefix $LOGDIR_PREFIX \
  #     --total-steps $TOTAL_STEPS \
  #     --seed $seed \
  #   ) & # >& /dev/null &
  # done
  # endregion: PPO GRU - BC | gw-size: 512

# endregion: GRU

# region: GW

  # region: PPO GW - BC | gw-size: 64
  # for seed in 111 222; do
  #   export TOTAL_STEPS=20000000
  #   (sleep 1s && python ppo_bc.py \
  #     --exp-name "ppo_bc__gw__h_64" \
  #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
  #     --ent-coef 0.2  \
  #     --gw-size 64 \
  #     --agent-type "gw" \
  #     --save-videos False \
  #     --wandb --wandb-project $WANDB_PROJECT --wandb-entity dosssman \
  #     --logdir-prefix $LOGDIR_PREFIX \
  #     --total-steps $TOTAL_STEPS \
  #     --seed $seed \
  #   ) & # >& /dev/null &
  # done
  # endregion: PPO GW - BC | gw-size: 64

# endregion: GW

# Clean up env vars
# export CUDA_VISIBLE_DEVICES=
export WANDB_PROJECT=
export LOGDIR_PREFIX=
export MASTER_PORT=
export WANDB_DIR=
