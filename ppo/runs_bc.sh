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
  # for seed in 111; do
  #   # export MASTER_PORT=8738 # Default port is 8738
  #   export TOTAL_STEPS=10000000
  #   (sleep 1s && python ppo_bc.py \
  #       --exp-name "ppo_bc__ss1_rgb_spectro__gru" \
  #       --config-path "env_configs/audiogoal_rgb_nocont.yaml" \
  #       --wandb --wandb-project ss-hab-bc --wandb-entity dosssman \
  #       --logdir-prefix $LOGDIR_PREFIX \
  #       --total-steps $TOTAL_STEPS \
  #       --seed $seed \
  #   ) & # >& /dev/null &
  # done
  # endregion: PPO GRU - BC with default hyparams

  # region: Custom PPO + Perceiver GWT GWWM Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents; RGB + Spectrogram SS1
  # for seed in 111; do
  #   # export MASTER_PORT=8748 # Default port is 8738
  #   export TOTAL_STEPS=10000000
  #   (sleep 1s && python ppo_bc.py \
  #       --exp-name "ppo_bc__ss1_rgb_spectro__pgwt_gwwm__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats" \
  #       --agent-type "perceiver-gwt-gwwm" \
  #       --pgwt-depth 1 \
  #       --pgwt-num-latents 8 \
  #       --pgwt-latent-dim 64 \
  #       --pgwt-cross-heads 1 \
  #       --pgwt-latent-heads 4 \
  #       --pgwt-use-sa False \
  #       --pgwt-mod-embed 0 \
  #       --pgwt-ca-prev-latents True \
  #       --config-path "env_configs/audiogoal_rgb_nocont.yaml" \
  #       --save-videos True \
  #       --wandb --wandb-project ss-hab-bc --wandb-entity dosssman \
  #       --total-steps $TOTAL_STEPS \
  #       --logdir-prefix $LOGDIR_PREFIX \
  #       --seed $seed \
  #   ) & # >& /dev/null &
  # done
  # endregion: Custom PPO + Perceiver GWT GWWM Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents; RGB + Spectrogram SS1

  # region: PPO GRU | Perceiver - BC with default hyparams, search over batch -size
  for seed in 111; do
    for bsize in 32; do
      export TOTAL_STEPS=1000000
      
      ## PPO GRU
      # (sleep 1s && python ppo_bc.py \
      #   --config-path "env_configs/audiogoal_rgb_nocont.yaml" \
      #   --wandb --wandb-project ss-hab-bc --wandb-entity dosssman \
      #   --exp-name "ppo_bc__ss1_rgb_spectro__gru__bsize_$bsize" \
      #   --num-envs $bsize \
      #   --logdir-prefix $LOGDIR_PREFIX \
      #   --total-steps $TOTAL_STEPS \
      #   --seed $seed \
      # ) & # >& /dev/null &

      ## PPO Perceiver
      # (sleep 1s && python ppo_bc.py \
      #   --agent-type "perceiver-gwt-gwwm" \
      #   --pgwt-depth 1 \
      #   --pgwt-num-latents 8 \
      #   --pgwt-latent-dim 64 \
      #   --pgwt-cross-heads 1 \
      #   --pgwt-latent-heads 4 \
      #   --pgwt-use-sa False \
      #   --pgwt-mod-embed 0 \
      #   --pgwt-ca-prev-latents True \
      #   --config-path "env_configs/audiogoal_rgb_nocont.yaml" \
      #   --save-videos True \
      #   --wandb --wandb-project ss-hab-bc --wandb-entity dosssman \
      #   --exp-name "ppo_bc__ss1_rgb_spectro__pgwt_gwwm__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats__bsize_$bsize" \
      #   --num-envs $bsize \
      #   --total-steps $TOTAL_STEPS \
      #   --logdir-prefix $LOGDIR_PREFIX \
      #   --seed $seed \
      # ) & # >& /dev/null &
    done

    ## ppo_bc
    
    # region: PPO GRU: batch_size 64; batch_chunk_len: 32;  seq_length == chunk_length: 80 (accum grad over 4 mini batches)
    # (sleep 1s && python ppo_bc.py \
    #   --config-path "env_configs/audiogoal_rgb_nocont.yaml" \
    #   --wandb --wandb-project ss-hab-bc --wandb-entity dosssman \
    #   --exp-name "ppo_bc__ss1_rgb_spectro__gru__bsize_10_bchnklen_10__nsteps_150_chnklen_150" \
    #   --num-steps 150 \
    #   --chunk-length 150 \
    #   --num-envs 10 \
    #   --batch-chunk-length 10 \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --total-steps $TOTAL_STEPS \
    #   --seed $seed \
    # ) & # >& /dev/null &
    # endregion: PPO GRU: batch_size 64; batch_chunk_len: 32;  seq_length == chunk_length: 80 (accum grad over 4 mini batches)

    # region: PPO Perceiver batch_size 64; batch_chunk_len: 32;  seq_length == chunk_length: 80
    # (sleep 1s && python ppo_bc.py \
    #   --agent-type "perceiver-gwt-gwwm" \
    #   --pgwt-depth 1 \
    #   --pgwt-num-latents 8 \
    #   --pgwt-latent-dim 64 \
    #   --pgwt-cross-heads 1 \
    #   --pgwt-latent-heads 4 \
    #   --pgwt-use-sa False \
    #   --pgwt-mod-embed 0 \
    #   --pgwt-ca-prev-latents True \
    #   --config-path "env_configs/audiogoal_rgb_nocont.yaml" \
    #   --save-videos True \
    #   --wandb --wandb-project ss-hab-bc --wandb-entity dosssman \
    #   --exp-name "ppo_bc__ss1_rgb_spectro__pgwt_gwwm__dpth_1_nlats_8_latdim_64_noSA_CAprevlats__bsize_10_bchnklen_10__nsteps_150_chnklen_150" \
    #   --num-steps 150 \
    #   --chunk-length 150 \
    #   --num-envs 10 \
    #   --batch-chunk-length 10 \
    #   --total-steps $TOTAL_STEPS \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --seed $seed \
    # ) & # >& /dev/null &
    # endregion: PPO Perceiver batch_size 128; batch_chunk_len: 32;  seq_length == chunk_length: 80

    ## ppo_bc2

    # region: PPO GRU: batch_size 64; batch_chunk_len: 32;  seq_length == chunk_length: 80 (accum grad over 4 mini batches)
    # (sleep 1s && python ppo_bc2.py \
    #   --config-path "env_configs/audiogoal_rgb_nocont.yaml" \
    #   --wandb --wandb-project ss-hab-bc --wandb-entity dosssman \
    #   --exp-name "ppo_bc2__ss1_rgb_spectro__gru__bsize_128_bchnklen_32__nsteps_80" \
    #   --num-steps 80 \
    #   --num-envs 128 \
    #   --batch-chunk-length 32 \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --total-steps $TOTAL_STEPS \
    #   --seed $seed \
    # ) & # >& /dev/null &
    # endregion: PPO GRU: batch_size 64; batch_chunk_len: 32;  seq_length == chunk_length: 80 (accum grad over 4 mini batches)

    # region: PPO Perceiver batch_size 64; batch_chunk_len: 32;  seq_length == chunk_length: 80
    # (sleep 1s && python ppo_bc2.py \
    #   --agent-type "perceiver-gwt-gwwm" \
    #   --pgwt-depth 1 \
    #   --pgwt-num-latents 8 \
    #   --pgwt-latent-dim 64 \
    #   --pgwt-cross-heads 1 \
    #   --pgwt-latent-heads 4 \
    #   --pgwt-use-sa False \
    #   --pgwt-mod-embed 0 \
    #   --pgwt-ca-prev-latents True \
    #   --config-path "env_configs/audiogoal_rgb_nocont.yaml" \
    #   --save-videos True \
    #   --wandb --wandb-project ss-hab-bc --wandb-entity dosssman \
    #   --exp-name "ppo_bc2__ss1_rgb_spectro__pgwt_gwwm__dpth_1_nlats_8_latdim_64_noSA_CAprevlats__bsize_128_bchnklen_32__nsteps_80" \
    #   --num-steps 80 \
    #   --num-envs 128 \
    #   --batch-chunk-length 32 \
    #   --total-steps $TOTAL_STEPS \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --seed $seed \
    # ) & # >& /dev/null &
    # endregion: PPO Perceiver batch_size 128; batch_chunk_len: 32;  seq_length == chunk_length: 80

  done
  # region: PPO GRU | Perceiver - BC with default hyparams, search over batch -size

# endregion: PPO BC #

# Clean up env vars
export CUDA_VISIBLE_DEVICES=
export LOGDIR_PREFIX=
export MASTER_PORT=
export WANDB_DIR=