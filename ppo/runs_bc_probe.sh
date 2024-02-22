#!/bin/bash
NUM_CORES=$(nproc --all)
export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES

export LOGDIR_PREFIX=~/random/rl/exp-logs/ss-hab-bc-revised-finals-probing
if [ ! -d $LOGDIR_PREFIX ]; then
  mkdir -p $LOGDIR_PREFIX
fi

export WANDB_DIR="$LOGDIR_PREFIX""_wandb"
if [ ! -d $WANDB_DIR ]; then
  mkdir -p $WANDB_DIR
fi

# echo $PATH
# TODO: Fix the issue that requires this kind of hardcoding
export LD_LIBRARY_PATH="/usr/local/cudnn-8.4.1_cuda_11.x:/usr/local/cuda-11.7/lib64:"
echo "${LD_LIBRARY_PATH}"

# Set training hyparams
export TOTAL_STEPS=500000
export N_EPOCHS=10

# region: GRU
  # region: 32
    # 1:
    # 2:
    # (sleep 1s && python ppo_bc_probe_train.py \
    #   --exp-name "ppo_bc__sweep_gru_32__seed_2" \
    #   --num-minibatches 50 \
    #   --agent-type "gru" \
    #   --gw-size 32 \
    #   --pretrained-model-name "ppo_gru__sweep_gw_32__seed_2" \
    #   --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc-revised-final/ppo_bc_seed_2__2024_02_14_23_16_31_981756.musashi/models/ppo_agent.20001000.ckpt.pth" \
    #   --wandb --wandb-project "ss-hab-bc-revised-finals-probing" --wandb-entity dosssman \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --total-steps $TOTAL_STEPS \
    #   --n-epochs $N_EPOCHS \
    #   --seed 42 \
    # ) & # >& /dev/null &
  # endregion: 32

  # region: 64
    # 1:
    # 2:
    # (sleep 1s && python ppo_bc_probe_train.py \
    #   --exp-name "ppo_bc__sweep_gru_64__seed_2" \
    #   --num-minibatches 50 \
    #   --agent-type "gru" \
    #   --gw-size 64 \
    #   --pretrained-model-name "ppo_bc__sweep_gw_64__seed_2" \
    #   --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc-revised-final/ppo_bc_seed_2__2024_02_14_23_16_39_041498.musashi/models/ppo_agent.20001000.ckpt.pth" \
    #   --wandb --wandb-project "ss-hab-bc-revised-finals-probing" --wandb-entity dosssman \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --total-steps $TOTAL_STEPS \
    #   --n-epochs $N_EPOCHS \
    #   --seed 42 \
    # ) & # >& /dev/null &
  # endregion: 64

  # region: 128
    # 1:
    # 2:
    # (sleep 1s && python ppo_bc_probe_train.py \
    #   --exp-name "ppo_bc__sweep_gru_128__seed_2" \
    #   --num-minibatches 50 \
    #   --agent-type "gru" \
    #   --gw-size 128 \
    #   --pretrained-model-name "ppo_bc__sweep_gw_128__seed_2" \
    #   --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc-revised-final/ppo_bc_seed_2__2024_02_14_23_16_39_820446.musashi/models/ppo_agent.20001000.ckpt.pth" \
    #   --wandb --wandb-project "ss-hab-bc-revised-finals-probing" --wandb-entity dosssman \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --total-steps $TOTAL_STEPS \
    #   --n-epochs $N_EPOCHS \
    #   --seed 42 \
    # ) & # >& /dev/null &
  # endregion: 128

  # region: 256
  # endregion: 256

  # region: 512
  # endregion: 512
# endregion: GRU

### ----------------------------------------------- ###

# Clean up env vars
# export CUDA_VISIBLE_DEVICES=
export LOGDIR_PREFIX=
export MASTER_PORT=
export WANDB_DIR=