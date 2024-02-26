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
    export EXPNAME="ppo_bc__gru_32__seed_2"
    (sleep 1s && python ppo_bc_probe_train.py \
      --exp-name $EXPNAME --pretrained-model-name $EXPNAME \
      --agent-type "gru" \
      --gw-size 32 \
      --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc-revised-final/ppo_bc_seed_2__2024_02_14_23_16_31_981756.musashi/models/ppo_agent.20001000.ckpt.pth" \
      --wandb --wandb-project "ss-hab-bc-revised-finals-probing" --wandb-entity dosssman \
      --num-minibatches 50 \
      --logdir-prefix $LOGDIR_PREFIX \
      --total-steps $TOTAL_STEPS \
      --n-epochs $N_EPOCHS \
      --seed 42 \
    ) &&
    # 3:
    export EXPNAME="ppo_bc__gru_32__seed_3"
    (sleep 1s && python ppo_bc_probe_train.py \
      --exp-name $EXPNAME --pretrained-model-name $EXPNAME \
      --agent-type "gru" \
      --gw-size 32 \
      --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc-revised-final/ppo_bc_seed_3__2024_02_18_17_20_56_143826.musashi/models/ppo_agent.20001000.ckpt.pth" \
      --wandb --wandb-project "ss-hab-bc-revised-finals-probing" --wandb-entity dosssman \
      --num-minibatches 50 \
      --logdir-prefix $LOGDIR_PREFIX \
      --total-steps $TOTAL_STEPS \
      --n-epochs $N_EPOCHS \
      --seed 42 \
    ) &&
    # 4:
    export EXPNAME="ppo_bc__gru_32__seed_4"
    (sleep 1s && python ppo_bc_probe_train.py \
      --exp-name $EXPNAME --pretrained-model-name $EXPNAME \
      --agent-type "gru" \
      --gw-size 32 \
      --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc-revised-final/ppo_bc_seed_4__2024_02_22_11_33_12_409229.musashi/models/ppo_agent.20001000.ckpt.pth" \
      --wandb --wandb-project "ss-hab-bc-revised-finals-probing" --wandb-entity dosssman \
      --num-minibatches 50 \
      --logdir-prefix $LOGDIR_PREFIX \
      --total-steps $TOTAL_STEPS \
      --n-epochs $N_EPOCHS \
      --seed 42 \
    ) &&
  # endregion: 32

  # region: 64
    # 1:
    # 2:
    export EXPNAME="ppo_bc__gru_64__seed_2"
    (sleep 1s && python ppo_bc_probe_train.py \
      --exp-name $EXPNAME --pretrained-model-name $EXPNAME \
      --agent-type "gru" \
      --gw-size 64 \
      --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc-revised-final/ppo_bc_seed_2__2024_02_14_23_16_39_041498.musashi/models/ppo_agent.20001000.ckpt.pth" \
      --wandb --wandb-project "ss-hab-bc-revised-finals-probing" --wandb-entity dosssman \
      --num-minibatches 50 \
      --logdir-prefix $LOGDIR_PREFIX \
      --total-steps $TOTAL_STEPS \
      --n-epochs $N_EPOCHS \
      --seed 42 \
    ) &&
    # 3:
    export EXPNAME="ppo_bc__gru_64__seed_3"
    (sleep 1s && python ppo_bc_probe_train.py \
      --exp-name $EXPNAME --pretrained-model-name $EXPNAME \
      --agent-type "gru" \
      --gw-size 64 \
      --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc-revised-final/ppo_bc_seed_3__2024_02_18_17_20_55_505423.musashi/models/ppo_agent.20001000.ckpt.pth" \
      --wandb --wandb-project "ss-hab-bc-revised-finals-probing" --wandb-entity dosssman \
      --num-minibatches 50 \
      --logdir-prefix $LOGDIR_PREFIX \
      --total-steps $TOTAL_STEPS \
      --n-epochs $N_EPOCHS \
      --seed 42 \
    ) &&
    # 4:
    export EXPNAME="ppo_bc__gru_64__seed_4"
    (sleep 1s && python ppo_bc_probe_train.py \
      --exp-name $EXPNAME --pretrained-model-name $EXPNAME \
      --agent-type "gru" \
      --gw-size 64 \
      --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc-revised-final/ppo_bc_seed_4__2024_02_22_11_32_47_249281.musashi/models/ppo_agent.20001000.ckpt.pth" \
      --wandb --wandb-project "ss-hab-bc-revised-finals-probing" --wandb-entity dosssman \
      --num-minibatches 50 \
      --logdir-prefix $LOGDIR_PREFIX \
      --total-steps $TOTAL_STEPS \
      --n-epochs $N_EPOCHS \
      --seed 42 \
    ) &
  # endregion: 64

  # region: 128
    # 2:
    export EXPNAME="ppo_bc__gru_128__seed_2"
    (sleep 1s && python ppo_bc_probe_train.py \
      --exp-name $EXPNAME --pretrained-model-name $EXPNAME \
      --agent-type "gru" \
      --gw-size 128 \
      --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc-revised-final/ppo_bc_seed_2__2024_02_14_23_16_39_820446.musashi/models/ppo_agent.20001000.ckpt.pth" \
      --wandb --wandb-project "ss-hab-bc-revised-finals-probing" --wandb-entity dosssman \
      --num-minibatches 50 \
      --logdir-prefix $LOGDIR_PREFIX \
      --total-steps $TOTAL_STEPS \
      --n-epochs $N_EPOCHS \
      --seed 42 \
    ) &&
    # 3:
    export EXPNAME="ppo_bc__gru_128__seed_3"
    (sleep 1s && python ppo_bc_probe_train.py \
      --exp-name $EXPNAME --pretrained-model-name $EXPNAME \
      --agent-type "gru" \
      --gw-size 128 \
      --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc-revised-final/ppo_bc_seed_3__2024_02_18_17_20_58_861635.musashi/models/ppo_agent.20001000.ckpt.pth" \
      --wandb --wandb-project "ss-hab-bc-revised-finals-probing" --wandb-entity dosssman \
      --num-minibatches 50 \
      --logdir-prefix $LOGDIR_PREFIX \
      --total-steps $TOTAL_STEPS \
      --n-epochs $N_EPOCHS \
      --seed 42 \
    ) &&
    # 4:
    export EXPNAME="ppo_bc__gru_128__seed_4"
    (sleep 1s && python ppo_bc_probe_train.py \
      --exp-name $EXPNAME --pretrained-model-name $EXPNAME \
      --agent-type "gru" \
      --gw-size 128 \
      --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc-revised-final/ppo_bc_seed_4__2024_02_22_11_32_58_100153.musashi/models/ppo_agent.20001000.ckpt.pth" \
      --wandb --wandb-project "ss-hab-bc-revised-finals-probing" --wandb-entity dosssman \
      --num-minibatches 50 \
      --logdir-prefix $LOGDIR_PREFIX \
      --total-steps $TOTAL_STEPS \
      --n-epochs $N_EPOCHS \
      --seed 42 \
    ) &&
  # endregion: 128

  # region: 256
  # endregion: 256

  # region: 512
  # endregion: 512
# endregion: GRU


# region: GW
  # region: 32
    # 1:
    export EXPNAME="ppo_bc__gw_32__seed_1"
    (sleep 1s && python ppo_bc_probe_train.py \
      --exp-name $EXPNAME --pretrained-model-name $EXPNAME \
      --agent-type "gw" \
      --gw-size 32 \
      --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc-revised-final/ppo_bc_seed_1__2024_02_13_10_37_37_585583.Loki/models/ppo_agent.20001000.ckpt.pth" \
      --wandb --wandb-project "ss-hab-bc-revised-finals-probing" --wandb-entity dosssman \
      --num-minibatches 50 \
      --logdir-prefix $LOGDIR_PREFIX \
      --total-steps $TOTAL_STEPS \
      --n-epochs $N_EPOCHS \
      --seed 42 \
    ) &&
    # 2:
    export EXPNAME="ppo_bc__gw_32__seed_2"
    (sleep 1s && python ppo_bc_probe_train.py \
      --exp-name $EXPNAME --pretrained-model-name $EXPNAME \
      --agent-type "gw" \
      --gw-size 32 \
      --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc-revised-final/ppo_bc_seed_2__2024_02_16_01_22_44_875490.Loki/models/ppo_agent.20001000.ckpt.pth" \
      --wandb --wandb-project "ss-hab-bc-revised-finals-probing" --wandb-entity dosssman \
      --num-minibatches 50 \
      --logdir-prefix $LOGDIR_PREFIX \
      --total-steps $TOTAL_STEPS \
      --n-epochs $N_EPOCHS \
      --seed 42 \
    ) &&
    # 3:
    export EXPNAME="ppo_bc__gw_32__seed_3"
    (sleep 1s && python ppo_bc_probe_train.py \
      --exp-name $EXPNAME --pretrained-model-name $EXPNAME \
      --agent-type "gw" \
      --gw-size 32 \
      --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc-revised-final/ppo_bc_seed_3__2024_02_20_19_13_41_898999.Loki/models/ppo_agent.20001000.ckpt.pth" \
      --wandb --wandb-project "ss-hab-bc-revised-finals-probing" --wandb-entity dosssman \
      --num-minibatches 50 \
      --logdir-prefix $LOGDIR_PREFIX \
      --total-steps $TOTAL_STEPS \
      --n-epochs $N_EPOCHS \
      --seed 42 \
    ) &
  # endregion: 32

  # region: 64
    # 1:
    export EXPNAME="ppo_bc__gw_64__seed_1"
    (sleep 1s && python ppo_bc_probe_train.py \
      --exp-name $EXPNAME --pretrained-model-name $EXPNAME \
      --agent-type "gw" \
      --gw-size 64 \
      --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc-revised-final/ppo_bc_seed_1__2024_02_13_10_37_42_215535.Loki/models/ppo_agent.20001000.ckpt.pth" \
      --wandb --wandb-project "ss-hab-bc-revised-finals-probing" --wandb-entity dosssman \
      --num-minibatches 50 \
      --logdir-prefix $LOGDIR_PREFIX \
      --total-steps $TOTAL_STEPS \
      --n-epochs $N_EPOCHS \
      --seed 42 \
    ) &&
    # 4:
    export EXPNAME="ppo_bc__gw_64__seed_4"
    (sleep 1s && python ppo_bc_probe_train.py \
      --exp-name $EXPNAME --pretrained-model-name $EXPNAME \
      --agent-type "gw" \
      --gw-size 64 \
      --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc-revised-final/ppo_bc_seed_4__2024_02_16_01_36_19_264390.Loki/models/ppo_agent.20001000.ckpt.pth" \
      --wandb --wandb-project "ss-hab-bc-revised-finals-probing" --wandb-entity dosssman \
      --num-minibatches 50 \
      --logdir-prefix $LOGDIR_PREFIX \
      --total-steps $TOTAL_STEPS \
      --n-epochs $N_EPOCHS \
      --seed 42 \
    ) &&
    # 5:
    export EXPNAME="ppo_bc__gw_64__seed_5"
    (sleep 1s && python ppo_bc_probe_train.py \
      --exp-name $EXPNAME --pretrained-model-name $EXPNAME \
      --agent-type "gw" \
      --gw-size 64 \
      --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc-revised-final/ppo_bc_seed_5__2024_02_20_22_20_43_504368.Loki/models/ppo_agent.20001000.ckpt.pth" \
      --wandb --wandb-project "ss-hab-bc-revised-finals-probing" --wandb-entity dosssman \
      --num-minibatches 50 \
      --logdir-prefix $LOGDIR_PREFIX \
      --total-steps $TOTAL_STEPS \
      --n-epochs $N_EPOCHS \
      --seed 42 \
    ) &&
  # endregion: 64

  # region: 128
    # 1:
    export EXPNAME="ppo_bc__gw_128__seed_1"
    (sleep 1s && python ppo_bc_probe_train.py \
      --exp-name $EXPNAME --pretrained-model-name $EXPNAME \
      --agent-type "gw" \
      --gw-size 128 \
      --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc-revised-final/ppo_bc_seed_1__2024_02_13_10_37_37_584574.Loki/models/ppo_agent.20001000.ckpt.pth" \
      --wandb --wandb-project "ss-hab-bc-revised-finals-probing" --wandb-entity dosssman \
      --num-minibatches 50 \
      --logdir-prefix $LOGDIR_PREFIX \
      --total-steps $TOTAL_STEPS \
      --n-epochs $N_EPOCHS \
      --seed 42 \
    ) &&
    # 2:
    export EXPNAME="ppo_bc__gw_128__seed_2"
    (sleep 1s && python ppo_bc_probe_train.py \
      --exp-name $EXPNAME --pretrained-model-name $EXPNAME \
      --agent-type "gw" \
      --gw-size 128 \
      --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc-revised-final/ppo_bc_seed_2__2024_02_16_01_22_44_854253.Loki/models/ppo_agent.20001000.ckpt.pth" \
      --wandb --wandb-project "ss-hab-bc-revised-finals-probing" --wandb-entity dosssman \
      --num-minibatches 50 \
      --logdir-prefix $LOGDIR_PREFIX \
      --total-steps $TOTAL_STEPS \
      --n-epochs $N_EPOCHS \
      --seed 42 \
    ) &&
    # 3:
    export EXPNAME="ppo_bc__gw_128__seed_3"
    (sleep 1s && python ppo_bc_probe_train.py \
      --exp-name $EXPNAME --pretrained-model-name $EXPNAME \
      --agent-type "gw" \
      --gw-size 128 \
      --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc-revised-final/ppo_bc_seed_3__2024_02_20_19_13_42_542902.Loki/models/ppo_agent.20001000.ckpt.pth" \
      --wandb --wandb-project "ss-hab-bc-revised-finals-probing" --wandb-entity dosssman \
      --num-minibatches 50 \
      --logdir-prefix $LOGDIR_PREFIX \
      --total-steps $TOTAL_STEPS \
      --n-epochs $N_EPOCHS \
      --seed 42 \
    ) &
  # endregion: 128

  # region: 256
  # endregion: 256

  # region: 512
  # endregion: 512

# endregion: GW
### ----------------------------------------------- ###

# Clean up env vars
# export CUDA_VISIBLE_DEVICES=
export EXPNAME=
export LOGDIR_PREFIX=
export MASTER_PORT=
export WANDB_DIR=
