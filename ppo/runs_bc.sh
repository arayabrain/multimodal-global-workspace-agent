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

# echo $PATH
# TODO: Fix the issue that requires this kind of hardcoding
# export LD_LIBRARY_PATH="/usr/local/cudnn-8.4.1_cuda_11.x:/usr/local/cuda-11.7/lib64:"
# echo "${LD_LIBRARY_PATH}"

# region: PPO BC #

  # region: SAVI

    ## RGB + Spectrogram based section

      # GRUv3

      # region: PPO GRUv3 - BC | gw at rec enc level, detached; GRU Layer Norm, entropy reg 0.2, no ce weights
      # for seed in 111 222; do
      #   export TOTAL_STEPS=20000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_spectro__gruv3__gw_detach__grulynrm__entcoef_0.2__no_cew" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --ent-coef 0.2 \
      #     --dataset-ce-weights "False" \
      #     --agent-type "gruv3" \
      #     --gwtv3-use-gw "True" \
      #     --gwtv3-enc-gw-detach True \
      #     --gwtv3-gru-type "layernorm" \
      #     --save-videos False \h
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --total-steps $TOTAL_STEPS \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: PPO GRUv3 - BC | gw at rec enc level, detached; GRU Layer Norm, entropy reg 0.2, no ce weights

      # region: PPO GRUv3 - BC | gw at rec enc level, detached; GRU Layer Norm, entropy reg 0.2, no ce weights, H=64
      # for seed in 111 222 333; do
      #   export TOTAL_STEPS=20000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_spectro__gruv3__gw_detach__grulynrm__entcoef_0.2__no_cew__h_64" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --ent-coef 0.2 \
      #     --dataset-ce-weights "False" \
      #     --hidden-size 64 \
      #     --agent-type "gruv3" \
      #     --gwtv3-use-gw "True" \
      #     --gwtv3-enc-gw-detach True \
      #     --gwtv3-gru-type "layernorm" \
      #     --save-videos False \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --total-steps $TOTAL_STEPS \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: PPO GRUv3 - BC | gw at rec enc level, detached; GRU Layer Norm, entropy reg 0.2, no ce weights, H=64

      # region: PPO GRUv3 - BC | gw at rec enc level, detached; GRU Layer Norm, entropy reg 0.2, no ce weights, H=32
      # for seed in 111 222 333; do
      #   export TOTAL_STEPS=20000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_spectro__gruv3__gw_detach__grulynrm__entcoef_0.2__no_cew__h_32" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --ent-coef 0.2 \
      #     --dataset-ce-weights "False" \
      #     --hidden-size 32 \
      #     --agent-type "gruv3" \
      #     --gwtv3-use-gw "True" \
      #     --gwtv3-enc-gw-detach True \
      #     --gwtv3-gru-type "layernorm" \
      #     --save-videos False \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --total-steps $TOTAL_STEPS \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: PPO GRUv3 - BC | gw at rec enc level, detached; GRU Layer Norm, entropy reg 0.2, no ce weights, H=64


      # GWTv3

      # region: PPO GWTv3 - BC | gw at rec enc level, detached, use null, GRU Layer Norm, entropy reg 0.2, no ce weights
      # for seed in 111 222 333; do
      #   export TOTAL_STEPS=20000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_spectro__gwtv3__gw_detach__usenull__grulynrm__entcoef_0.2__no_cew" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --ent-coef 0.2 \
      #     --dataset-ce-weights "False" \
      #     --agent-type "gwtv3" \
      #     --gwtv3-use-gw "True" \
      #     --gwtv3-use-null "True" \
      #     --gwtv3-enc-gw-detach True \
      #     --gwtv3-gru-type "layernorm" \
      #     --save-videos False \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --total-steps $TOTAL_STEPS \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: PPO GWTv3 - BC | gw at rec enc level, detached, use null, GRU Layer Norm, entropy reg 0.2, no ce weights

      # region: PPO GWTv3 - BC | gw at rec enc level, detached, use null, GRU Layer Norm, entropy reg 0.2, no ce weights, hidden-size=64
      # for seed in 111 222; do
      #   export TOTAL_STEPS=20000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_spectro__gwtv3__gw_detach__usenull__grulynrm__entcoef_0.2__no_cew__h_64" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --hidden-size 64 \
      #     --ent-coef 0.2 \
      #     --dataset-ce-weights "False" \
      #     --agent-type "gwtv3" \
      #     --gwtv3-use-gw "True" \
      #     --gwtv3-use-null "True" \
      #     --gwtv3-enc-gw-detach True \
      #     --gwtv3-gru-type "layernorm" \
      #     --save-videos False \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --total-steps $TOTAL_STEPS \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: PPO GWTv3 - BC | gw at rec enc level, detached, use null, GRU Layer Norm, entropy reg 0.2, no ce weights, hidden-size=64

      # region: PPO GWTv3 - BC | gw at rec enc level, detached, use null, GRU Layer Norm, entropy reg 0.2, no ce weights, hidden-size=64
      for seed in 111 222 333; do
        export TOTAL_STEPS=20000000
        (sleep 1s && python ppo_bc.py \
          --exp-name "ppo_bc__savi_ss1_rgb_spectro__gwtv3__gw_detach__usenull__grulynrm__entcoef_0.2__no_cew__h_32" \
          --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
          --hidden-size 32 \
          --ent-coef 0.2 \
          --dataset-ce-weights "False" \
          --agent-type "gwtv3" \
          --gwtv3-use-gw "True" \
          --gwtv3-use-null "True" \
          --gwtv3-enc-gw-detach True \
          --gwtv3-gru-type "layernorm" \
          --save-videos False \
          --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
          --logdir-prefix $LOGDIR_PREFIX \
          --total-steps $TOTAL_STEPS \
          --seed $seed \
        ) & # >& /dev/null &
      done
      # endregion: PPO GWTv3 - BC | gw at rec enc level, detached, use null, GRU Layer Norm, entropy reg 0.2, no ce weights, hidden-size=64


      ## GWTv3.1

      # region: PPO GWTv3.1 - BC | gw at rec enc level, detached, use null, GRU Layer Norm, entropy reg 0.2, no ce weights
      # for seed in 111 222; do
      #   export TOTAL_STEPS=20000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_spectro__gwtv3.1__gw_detach__usenull__grulynrm__entcoef_0.2__no_cew" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --ent-coef 0.2 \
      #     --dataset-ce-weights "False" \
      #     --agent-type "gwtv3.1" \
      #     --gwtv3-use-gw "True" \
      #     --gwtv3-use-null "True" \
      #     --gwtv3-enc-gw-detach True \
      #     --gwtv3-gru-type "layernorm" \
      #     --save-videos False \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --total-steps $TOTAL_STEPS \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: PPO GWTv3.1 - BC | gw at rec enc level, detached, use null, GRU Layer Norm, entropy reg 0.2, no ce weights

      # region: PPO GWTv3.1 - BC | gw at rec enc level, detached, use null, GRU Layer Norm, entropy reg 0.2, no ce weights, H=64
      # for seed in 111 222; do
      #   export TOTAL_STEPS=20000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_spectro__gwtv3.1__gw_detach__usenull__grulynrm__entcoef_0.2__no_cew__h_64" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --ent-coef 0.2 \
      #     --dataset-ce-weights "False" \
      #     --hidden-size 64 \
      #     --agent-type "gwtv3.1" \
      #     --gwtv3-use-gw "True" \
      #     --gwtv3-use-null "True" \
      #     --gwtv3-enc-gw-detach True \
      #     --gwtv3-gru-type "layernorm" \
      #     --save-videos False \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --total-steps $TOTAL_STEPS \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: PPO GWTv3.1 - BC | gw at rec enc level, detached, use null, GRU Layer Norm, entropy reg 0.2, no ce weights, H=64


      ## GWTv3.2

      # region: PPO GWTv3.2 - BC | gw at rec enc level, detached, use null, GRU Layer Norm, entropy reg 0.2, no ce weights
      # for seed in 111 222; do
      #   export TOTAL_STEPS=20000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_spectro__gwtv3.2__gw_detach__usenull__grulynrm__entcoef_0.2__no_cew" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --ent-coef 0.2 \
      #     --dataset-ce-weights "False" \
      #     --agent-type "gwtv3.2" \
      #     --gwtv3-use-gw "True" \
      #     --gwtv3-use-null "True" \
      #     --gwtv3-enc-gw-detach True \
      #     --gwtv3-gru-type "layernorm" \
      #     --save-videos False \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --total-steps $TOTAL_STEPS \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: PPO GWTv3.2 - BC | gw at rec enc level, detached, use null, GRU Layer Norm, entropy reg 0.2, no ce weights

  # endregion: SAVI
# endregion: PPO BC #

# Clean up env vars
# export CUDA_VISIBLE_DEVICES=
export LOGDIR_PREFIX=
export MASTER_PORT=
export WANDB_DIR=
