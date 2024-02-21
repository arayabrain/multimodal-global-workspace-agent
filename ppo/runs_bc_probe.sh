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

# region: GW
  # region: 32
    # 1:
    # 2:
    # (sleep 1s && python ppo_bc_probe_train.py \
    #   --exp-name "ppo_bc__sweep_gw_32__seed_2" \
    #   --num-minibatches 50 \
    #   --agent-type "gru" \
    #   --gw-size 32 \
    #   --pretrained-model-name "ppo_bc__sweep_gw_32__seed_2" \
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
    #   --exp-name "ppo_bc__sweep_gw_64__seed_2__prb_dpth_2" \
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
    #   --exp-name "ppo_bc__sweep_gw_128__seed_2" \
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
# endregion: GW

# GRU

## RNN agents get inputs of shape T * B = 150 * 10, but the input for the probes is broken into 50 batches of size 30. probe depth 2, probe hid size 1024, bias = True
# region: Probing ppo_bc__savi_ss1_rgb_spectro__gwtv3__gw_detach__usenull__grulynrm__entcoef_0.2__no_cew @20M steps, Linear probe with minibatch train
# for seed in 111; do
#     TOTAL_STEPS=500000; N_EPOCHS=10;
#     (sleep 1s && python ppo_bc_probe_train_mb.py \
#       --exp-name "ppo_bc__savi_ss1_rgb_spectro__gwtv3__gw_detach__usenull__grulynrm__entcoef_0.2__no_cew__n_mb_50__prb_dpth_2" \
#       --num-minibatches 50 \
#       --probe-depth 2 \
#       --agent-type "gwtv3" \
#       --gwtv3-use-gw "True" \
#       --gwtv3-enc-gw-detach "True" \
#       --gwtv3-use-null "True" \
#       --gwtv3-gru-type "layernorm" \
#       --obs-center False \
#       --pretrained-model-name "ppo_bc__savi_ss1_rgb_spectro__gwtv3__gw_detach__usenull__grulynrm__entcoef_0.2__no_cew" \
#       --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc/ppo_bc__savi_ss1_rgb_spectro__gwtv3__gw_detach__usenull__grulynrm__entcoef_0.2__no_cew_seed_111__2023_11_08_13_34_19_183492.musashi/models/ppo_agent.19995001.ckpt.pth" \
#       --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
#       --save-videos False \
#       --wandb --wandb-project "ss-hab-bc-probing" --wandb-entity dosssman \
#       --logdir-prefix $LOGDIR_PREFIX \
#       --total-steps $TOTAL_STEPS \
#       --n-epochs $N_EPOCHS \
#       --seed $seed \
#     ) & # >& /dev/null &
# done

# endregion: Probing ppo_bc__savi_ss1_rgb_spectro__gwtv3__gw_detach__usenull__grulynrm__entcoef_0.2__no_cew @20M steps, Linear probe with minibatch train

### ----------------------------------------------- ###

# Clean up env vars
# export CUDA_VISIBLE_DEVICES=
export LOGDIR_PREFIX=
export MASTER_PORT=
export WANDB_DIR=