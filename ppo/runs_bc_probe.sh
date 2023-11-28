#!/bin/bash
NUM_CORES=$(nproc --all)
export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES

export LOGDIR_PREFIX=~/random/rl/exp-logs/ss-hab-bc-probing
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

# Gen v3

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

## Queue up a probe run for each experiment name / seed
# declare EXPNAME_TO_PATH=()
# EXPNAME_TO_PATH+=( "ppo_bc__savi_ss1_rgb_spectro__gruv3__gw_detach__grulynrm__entcoef_0.2__no_cew__h_64_seed_111__2023_11_21_11_36_07_475470.musashi" )
# EXPNAME_TO_PATH+=( "ppo_bc__savi_ss1_rgb_spectro__gruv3__gw_detach__grulynrm__entcoef_0.2__no_cew__h_64_seed_222__2023_11_21_11_36_02_145955.musashi" )
# EXPNAME_TO_PATH+=( "ppo_bc__savi_ss1_rgb_spectro__gruv3__gw_detach__grulynrm__entcoef_0.2__no_cew__h_64_seed_333__2023_11_21_11_36_30_471112.musashi" )

# for expname in "${EXPNAME_TO_PATH[@]}"; do
#   echo "${expname}"

#   for seed in 111; do
#     TOTAL_STEPS=500000; N_EPOCHS=10;
#     (sleep 1s && python ppo_bc_probe_train_mb.py \
#       --exp-name "ppo_bc__savi_ss1_rgb_spectro__gruv3__gw_detach__usenull__grulynrm__entcoef_0.2__no_cew__h_64__n_mb_50__prb_dpth_2" \
#       --num-minibatches 50 \
#       --probe-depth 2 \
#       --agent-type "gruv3" \
#       --hidden-size 64 \
#       --gwtv3-use-gw "True" \
#       --gwtv3-enc-gw-detach "True" \
#       --gwtv3-use-null "True" \
#       --gwtv3-gru-type "layernorm" \
#       --obs-center False \
#       --pretrained-model-name "ppo_bc__savi_ss1_rgb_spectro__gruv3__gw_detach__usenull__grulynrm__entcoef_0.2__no_cew__h_64" \
#       --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc/${expname}/models/ppo_agent.19995001.ckpt.pth" \
#       --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
#       --save-videos False \
#       --ent-coef 0 \
#       --wandb --wandb-project "ss-hab-bc-probing" --wandb-entity dosssman \
#       --logdir-prefix $LOGDIR_PREFIX \
#       --total-steps $TOTAL_STEPS \
#       --n-epochs $N_EPOCHS \
#       --seed $seed \
#     ) & # >& /dev/null &
#   done
# done
# export EXPNAME_TO_PATH=

## RNN agents get inputs of shape T * B = 150 * 10, but the input for the probes is broken into 50 batches of size 30. probe depth 2, probe hid size 1024, bias = True
# region: Probing ppo_bc__savi_ss1_rgb_spectro__gruv3__gw_detach__usenull__grulynrm__entcoef_0.2__no_cew @20M steps, Linear probe with minibatch train
# for seed in 111; do
#     TOTAL_STEPS=500000; N_EPOCHS=10;
#     (sleep 1s && python ppo_bc_probe_train_mb.py \
#       --exp-name "ppo_bc__savi_ss1_rgb_spectro__gruv3__gw_detach__usenull__grulynrm__entcoef_0.2__no_cew__n_mb_50__prb_dpth_2" \
#       --num-minibatches 50 \
#       --probe-depth 2 \
#       --agent-type "gruv3" \
#       --gwtv3-use-gw "True" \
#       --gwtv3-enc-gw-detach "True" \
#       --gwtv3-use-null "True" \
#       --gwtv3-gru-type "layernorm" \
#       --obs-center False \
#       --pretrained-model-name "ppo_bc__savi_ss1_rgb_spectro__gruv3__gw_detach__usenull__grulynrm__entcoef_0.2__no_cew" \
#       --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc/ppo_bc__savi_ss1_rgb_spectro__gruv3__gw_detach__grulynrm__entcoef_0.2__no_cew_seed_111__2023_11_06_11_56_54_073258.conan/models/ppo_agent.19995001.ckpt.pth" \
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


## RNN agents get inputs of shape T * B = 150 * 10, but the input for the probes is broken into 50 batches of size 30. probe depth 2, probe hid size 1024, bias = True
# region: Probing ppo_bc__savi_ss1_rgb_spectro__gruv3__gw_detach__usenull__grulynrm__entcoef_0.2__no_cew @20M steps, Linear probe with minibatch train
# for seed in 111; do
#     TOTAL_STEPS=500000; N_EPOCHS=10;
#     (sleep 1s && python ppo_bc_probe_train_mb.py \
#       --exp-name "ppo_bc__savi_ss1_rgb_spectro__gruv3__gw_detach__usenull__grulynrm__entcoef_0.2__no_cew__n_mb_50__prb_dpth_2" \
#       --num-minibatches 50 \
#       --probe-depth 2 \
#       --agent-type "gruv3" \
#       --gwtv3-use-gw "True" \
#       --gwtv3-enc-gw-detach "True" \
#       --gwtv3-use-null "True" \
#       --gwtv3-gru-type "layernorm" \
#       --obs-center False \
#       --pretrained-model-name "ppo_bc__savi_ss1_rgb_spectro__gruv3__gw_detach__usenull__grulynrm__entcoef_0.2__no_cew" \
#       --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc/ppo_bc__savi_ss1_rgb_spectro__gruv3__gw_detach__grulynrm__entcoef_0.2__no_cew_seed_111__2023_11_06_11_56_54_073258.conan/models/ppo_agent.19995001.ckpt.pth" \
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


## RNN agents get inputs of shape T * B = 150 * 10, but the input for the probes is broken into 50 batches of size 30. probe depth 2, probe hid size 1024, bias = True
# region: Probing ppo_bc__savi_ss1_rgb_spectro__gruv3__gw_detach__usenull__grulynrm__entcoef_0.2__no_cew @20M steps, Linear probe with minibatch train
# declare EXPNAME_TO_PATH=()
# # EXPNAME_TO_PATH+=( ""ppo_bc__savi_ss1_rgb_spectro__gwtv3__gw_detach__usenull__grulynrm__entcoef_0.2__no_cew__h_64_seed_111__2023_11_08_13_34_18_744539.musashi"" )
# EXPNAME_TO_PATH+=( "ppo_bc__savi_ss1_rgb_spectro__gwtv3__gw_detach__usenull__grulynrm__entcoef_0.2__no_cew__h_64_seed_222__2023_11_08_13_37_01_281972.conan" )

# for expname in "${EXPNAME_TO_PATH[@]}"; do
#   echo "${expname}"
#   for seed in 111; do
#       TOTAL_STEPS=500000; N_EPOCHS=10;
#       (sleep 1s && python ppo_bc_probe_train_mb.py \
#         --exp-name "ppo_bc__savi_ss1_rgb_spectro__gwtv3__gw_detach__usenull__grulynrm__entcoef_0.2__no_cew__h_64__n_mb_50__prb_dpth_2" \
#         --num-minibatches 50 \
#         --probe-depth 2 \
#         --agent-type "gwtv3" \
#         --gwtv3-use-gw "True" \
#         --gwtv3-enc-gw-detach "True" \
#         --gwtv3-use-null "True" \
#         --gwtv3-gru-type "layernorm" \
#         --hidden-size 64 \
#         --obs-center False \
#         --pretrained-model-name "ppo_bc__savi_ss1_rgb_spectro__gwt3__gw_detach__usenull__grulynrm__entcoef_0.2__no_cew__h_64" \
#         --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc/${expname}/models/ppo_agent.19995001.ckpt.pth" \
#         --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
#         --save-videos False \
#         --ent-coef 0 \
#         --wandb --wandb-project "ss-hab-bc-probing" --wandb-entity dosssman \
#         --logdir-prefix $LOGDIR_PREFIX \
#         --total-steps $TOTAL_STEPS \
#         --n-epochs $N_EPOCHS \
#         --seed $seed \
#       ) & # >& /dev/null &
#   done
# done
# export EXPNAME_TO_PATH=
# endregion: Probing ppo_bc__savi_ss1_rgb_spectro__gwtv3__gw_detach__usenull__grulynrm__entcoef_0.2__no_cew @20M steps, Linear probe with minibatch train

### ----------------------------------------------- ###

# Clean up env vars
# export CUDA_VISIBLE_DEVICES=
export LOGDIR_PREFIX=
export MASTER_PORT=
export WANDB_DIR=