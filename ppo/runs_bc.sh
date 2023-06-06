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
export LD_LIBRARY_PATH="/usr/local/cudnn-8.4.1_cuda_11.x:/usr/local/cuda-11.7/lib64:"
echo "${LD_LIBRARY_PATH}"

# region: PPO BC #

  # region: AvNav
    # region: PPO GRU - BC with RGB and Depth for vision
    # for seed in 111 222; do
    #   # export MASTER_PORT=8738 # Default port is 8738
    #   export TOTAL_STEPS=5000000
    #   (sleep 1s && python ppo_bc.py \
    #       --exp-name "ppo_bc__ss1__rgb_depth_spectro__gru" \
    #       --config-path "env_configs/audiogoal_rgb_depth_ss1.yaml" \
    #       --dataset-path "AvNav_Oracle_Dataset_v0" \
    #       --dataset-ce-weights True \
    #       --save-videos True \
    #       --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
    #       --logdir-prefix $LOGDIR_PREFIX \
    #       --total-steps $TOTAL_STEPS \
    #       --seed $seed \
    #   ) & # >& /dev/null &
    # done
    # endregion: PPO GRU - BC with RGB and Depth for vision

    # region: PPO GRU - BC with RGB only for vision
    # for seed in 111 222; do
    #   # export MASTER_PORT=8738 # Default port is 8738
    #   export TOTAL_STEPS=5000000
    #   (sleep 1s && python ppo_bc.py \
    #       --exp-name "ppo_bc__ss1__rgb_spectro__gru" \
    #       --config-path "env_configs/audiogoal_rgb_nocont.yaml" \
    #       --dataset-path "AvNav_Oracle_Dataset_v0" \
    #       --ent-coef 0 \
    #       --dataset-ce-weights True \
    #       --save-videos True \
    #       --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
    #       --logdir-prefix $LOGDIR_PREFIX \
    #       --total-steps $TOTAL_STEPS \
    #       --seed $seed \
    #   ) & # >& /dev/null &
    # done
    # endregion: PPO GRU - BC with RGB only for vision

    # region: Custom PPO + Perceiver GWT GWWM Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents; RGB + Spectrogram SS1
    # for seed in 111 222; do
    #   # export MASTER_PORT=8748 # Default port is 8738
    #   export TOTAL_STEPS=5000000
    #   (sleep 1s && python ppo_bc.py \
    #       --exp-name "ppo_bc__ss1__rgb_depth_spectro__pgwt__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats" \
    #       --config-path "env_configs/audiogoal_rgb_depth_ss1.yaml" \
    #       --dataset-path "AvNav_Oracle_Dataset_v0" \
    #       --dataset-ce-weights True \
    #       --agent-type "perceiver-gwt-gwwm" \
    #       --pgwt-depth 1 \
    #       --pgwt-num-latents 8 \
    #       --pgwt-latent-dim 64 \
    #       --pgwt-cross-heads 1 \
    #       --pgwt-latent-heads 4 \
    #       --pgwt-use-sa False \
    #       --pgwt-mod-embed 0 \
    #       --pgwt-ca-prev-latents True \
    #       --save-videos True \
    #       --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
    #       --total-steps $TOTAL_STEPS \
    #       --logdir-prefix $LOGDIR_PREFIX \
    #       --seed $seed \
    #   ) & # >& /dev/null &
    # done
    # endregion: Custom PPO + Perceiver GWT GWWM Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents; RGB + Spectrogram SS1

    # Older expeirments scripts runs
    # region: PPO GRU: batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50 (accum grad over 4 mini batches), Entropy coef: 0 (reference), CE Weights 38.6 0.67 0.84 0.78, no grad norm
    # for seed in 111 222; do
    #   export TOTAL_STEPS=5000000
    #   (sleep 1s && python ppo_bc2_ndset.py \
    #     --dataset-path "AvNav_Oracle_Dataset_v0" \
    #     --config-path "env_configs/audiogoal_rgb_nocont.yaml" \
    #     --exp-name "ppo_bc2_ndset__ss1_rgb_spectro__gru__bsize_32_bchnklen_32__nsteps_50__cew_dset_nogradnorm" \
    #     --max-grad-norm 0 \
    #     --ent-coef 0 \
    #     --num-steps 50 \
    #     --num-envs 32 \
    #     --batch-chunk-length 32 \
    #     --save-videos True \
    #     --wandb --wandb-project ss-hab-bc --wandb-entity dosssman \
    #     --logdir-prefix $LOGDIR_PREFIX \
    #     --total-steps $TOTAL_STEPS \
    #     --seed $seed \
    #   ) & # >& /dev/null &
    # done
    # endregion: PPO GRU: batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50 (accum grad over 4 mini batches), Entropy coef: 0 (reference), CE Weights 38.6 0.67 0.84 0.78, no grad norm
  # endregion: AvNav


  # region: SAVI
  
    # region: PPO GRU - BC with default hyparams
    # for seed in 111 222 333; do
    #   # export MASTER_PORT=8738 # Default port is 8738
    #   export TOTAL_STEPS=5000000
    #   (sleep 1s && python ppo_bc.py \
    #       --exp-name "ppo_bc__savi_ss1__rgb_depth_spectro__gru" \
    #       --config-path "env_configs/savi/savi_ss1.yaml" \
    #       --save-videos True \
    #       --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
    #       --logdir-prefix $LOGDIR_PREFIX \
    #       --total-steps $TOTAL_STEPS \
    #       --seed $seed \
    #   ) & # >& /dev/null &
    # done
    # endregion: PPO GRU - BC with default hyparams

    # region: Custom PPO + Perceiver GWT GWWM Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents; RGB + Spectrogram SS1
    # for seed in 111 222 333; do
    #   # export MASTER_PORT=8748 # Default port is 8738
    #   export TOTAL_STEPS=5000000
    #   (sleep 1s && python ppo_bc.py \
    #       --exp-name "ppo_bc__savi_ss1__rgb_depth_spectro__pgwt__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats" \
    #       --agent-type "perceiver-gwt-gwwm" \
    #       --pgwt-depth 1 \
    #       --pgwt-num-latents 8 \
    #       --pgwt-latent-dim 64 \
    #       --pgwt-cross-heads 1 \
    #       --pgwt-latent-heads 4 \
    #       --pgwt-use-sa False \
    #       --pgwt-mod-embed 0 \
    #       --pgwt-ca-prev-latents True \
    #       --config-path "env_configs/savi/savi_ss1.yaml" \
    #       --save-videos True \
    #       --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
    #       --total-steps $TOTAL_STEPS \
    #       --logdir-prefix $LOGDIR_PREFIX \
    #       --seed $seed \
    #   ) & # >& /dev/null &
    # done
    # endregion: Custom PPO + Perceiver GWT GWWM Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents; RGB + Spectrogram SS1

  # endregion: SAVI
# endregion: PPO BC #

# Clean up env vars
# export CUDA_VISIBLE_DEVICES=
export LOGDIR_PREFIX=
export MASTER_PORT=
export WANDB_DIR=
