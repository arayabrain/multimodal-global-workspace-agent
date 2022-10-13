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

    export TOTAL_STEPS=4000000
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

    # region: PPO GRU: batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50 (accum grad over 4 mini batches)
    # (sleep 1s && python ppo_bc2.py \
    #   --config-path "env_configs/audiogoal_rgb_nocont.yaml" \
    #   --wandb --wandb-project ss-hab-bc --wandb-entity dosssman \
    #   --exp-name "ppo_bc2__ss1_rgb_spectro__gru__bsize_32_bchnklen_32__nsteps_50" \
    #   --num-steps 50 \
    #   --num-envs 32 \
    #   --batch-chunk-length 32 \
    #   --save-videos True \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --total-steps $TOTAL_STEPS \
    #   --seed $seed \
    # ) & # >& /dev/null &
    # endregion: PPO GRU: batch_size 64; batch_chunk_len: 32;  seq_length == chunk_length: 50 (accum grad over 4 mini batches)

    # region: PPO GRU: batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50 (accum grad over 4 mini batches), no Grad norm
    # (sleep 1s && python ppo_bc2.py \
    #   --config-path "env_configs/audiogoal_rgb_nocont.yaml" \
    #   --wandb --wandb-project ss-hab-bc --wandb-entity dosssman \
    #   --exp-name "ppo_bc2__ss1_rgb_spectro__gru__bsize_32_bchnklen_32__nsteps_50__maxgradnorm_0" \
    #   --num-steps 50 \
    #   --num-envs 32 \
    #   --batch-chunk-length 32 \
    #   --save-videos True \
    #   --max-grad-norm 0 \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --total-steps $TOTAL_STEPS \
    #   --seed $seed \
    # ) & # >& /dev/null &
    # endregion: PPO GRU: batch_size 64; batch_chunk_len: 32;  seq_length == chunk_length: 50 (accum grad over 4 mini batches)

    # region: PPO GRU: batch_size 128; batch_chunk_len: 32;  seq_length == chunk_length: 50 (accum grad over 4 mini batches)
    # (sleep 1s && python ppo_bc2.py \
    #   --config-path "env_configs/audiogoal_rgb_nocont.yaml" \
    #   --wandb --wandb-project ss-hab-bc --wandb-entity dosssman \
    #   --exp-name "ppo_bc2__ss1_rgb_spectro__gru__bsize_128_bchnklen_32__nsteps_50" \
    #   --num-steps 50 \
    #   --num-envs 128 \
    #   --batch-chunk-length 32 \
    #   --save-videos True \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --total-steps $TOTAL_STEPS \
    #   --seed $seed \
    # ) & # >& /dev/null &
    # endregion: PPO GRU: batch_size 64; batch_chunk_len: 32;  seq_length == chunk_length: 50 (accum grad over 4 mini batches)

    # region: PPO GRU: batch_size 256; batch_chunk_len: 32;  seq_length == chunk_length: 50 (accum grad over 4 mini batches)
    # (sleep 1s && python ppo_bc2.py \
    #   --config-path "env_configs/audiogoal_rgb_nocont.yaml" \
    #   --wandb --wandb-project ss-hab-bc --wandb-entity dosssman \
    #   --exp-name "ppo_bc2__ss1_rgb_spectro__gru__bsize_256_bchnklen_32__nsteps_50" \
    #   --num-steps 50 \
    #   --num-envs 256 \
    #   --batch-chunk-length 32 \
    #   --save-videos True \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --total-steps $TOTAL_STEPS \
    #   --seed $seed \
    # ) & # >& /dev/null &
    # endregion: PPO GRU: batch_size 256; batch_chunk_len: 32;  seq_length == chunk_length: 50 (accum grad over 4 mini batches)

    # region: PPO GRU: batch_size 128; batch_chunk_len: 32;  seq_length == chunk_length: 10 (accum grad over 4 mini batches)
    # (sleep 1s && python ppo_bc2.py \
    #   --config-path "env_configs/audiogoal_rgb_nocont.yaml" \
    #   --wandb --wandb-project ss-hab-bc --wandb-entity dosssman \
    #   --exp-name "ppo_bc2__ss1_rgb_spectro__gru__bsize_32_bchnklen_32__nsteps_10" \
    #   --num-steps 10 \
    #   --num-envs 128 \
    #   --batch-chunk-length 32 \
    #   --save-videos True \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --total-steps $TOTAL_STEPS \
    #   --seed $seed \
    # ) & # >& /dev/null &
    # endregion: PPO GRU: batch_size 128; batch_chunk_len: 32;  seq_length == chunk_length: 10 (accum grad over 4 mini batches)

    # region: PPO GRU: batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50 (accum grad over 4 mini batches), Entropy coef: 0 (reference)
    # (sleep 1s && python ppo_bc2.py \
    #   --config-path "env_configs/audiogoal_rgb_nocont.yaml" \
    #   --wandb --wandb-project ss-hab-bc --wandb-entity dosssman \
    #   --exp-name "ppo_bc2__ss1_rgb_spectro__gru__bsize_32_bchnklen_32__nsteps_50__entcoef_0.0" \
    #   --ent-coef 0.0 \
    #   --num-steps 50 \
    #   --num-envs 32 \
    #   --batch-chunk-length 32 \
    #   --save-videos True \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --total-steps $TOTAL_STEPS \
    #   --seed $seed \
    # ) & # >& /dev/null &
    # endregion: PPO GRU: batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50 (accum grad over 4 mini batches), Entropy coef: 0 (reference)

    # region: PPO GRU: batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50 (accum grad over 4 mini batches), Entropy coef: 0.2
    # (sleep 1s && python ppo_bc2.py \
    #   --config-path "env_configs/audiogoal_rgb_nocont.yaml" \
    #   --wandb --wandb-project ss-hab-bc --wandb-entity dosssman \
    #   --exp-name "ppo_bc2__ss1_rgb_spectro__gru__bsize_32_bchnklen_32__nsteps_50__entcoef_0.2" \
    #   --ent-coef 0.2 \
    #   --num-steps 50 \
    #   --num-envs 32 \
    #   --batch-chunk-length 32 \
    #   --save-videos True \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --total-steps $TOTAL_STEPS \
    #   --seed $seed \
    # ) & # >& /dev/null &
    # endregion: PPO GRU: batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50 (accum grad over 4 mini batches), Entropy coef: 0.2

    # region: PPO GRU: batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50 (accum grad over 4 mini batches), Entropy coef: 0 (reference), CE Weights 58 1 1 1
    # (sleep 1s && python ppo_bc2.py \
    #   --config-path "env_configs/audiogoal_rgb_nocont.yaml" \
    #   --wandb --wandb-project ss-hab-bc --wandb-entity dosssman \
    #   --exp-name "ppo_bc2__ss1_rgb_spectro__gru__bsize_32_bchnklen_32__nsteps_50__entcoef_0.0__cew_58_1_1_1" \
    #   --ce-weights 58 1 1 1 \
    #   --ent-coef 0.0 \
    #   --num-steps 50 \
    #   --num-envs 32 \
    #   --batch-chunk-length 32 \
    #   --save-videos True \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --total-steps $TOTAL_STEPS \
    #   --seed $seed \
    # ) & # >& /dev/null &
    # endregion: PPO GRU: batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50 (accum grad over 4 mini batches), Entropy coef: 0 (reference), CE Weights 58 1 1 1

    # region: PPO GRU: batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50 (accum grad over 4 mini batches), Entropy coef: 0.2, CE Weights 58 1 1 1
    # (sleep 1s && python ppo_bc2.py \
    #   --config-path "env_configs/audiogoal_rgb_nocont.yaml" \
    #   --wandb --wandb-project ss-hab-bc --wandb-entity dosssman \
    #   --exp-name "ppo_bc2__ss1_rgb_spectro__gru__bsize_32_bchnklen_32__nsteps_50__entcoef_0.2__cew_58_1_1_1" \
    #   --ce-weights 58 1 1 1 \
    #   --ent-coef 0.2 \
    #   --num-steps 50 \
    #   --num-envs 32 \
    #   --batch-chunk-length 32 \
    #   --save-videos True \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --total-steps $TOTAL_STEPS \
    #   --seed $seed \
    # ) & # >& /dev/null &
    # endregion: PPO GRU: batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50 (accum grad over 4 mini batches), Entropy coef: 0.2, CE Weights 58 1 1 1
    
    # region: PPO GRU: batch_size 128; batch_chunk_len: 32;  seq_length == chunk_length: 50 (accum grad over 4 mini batches), Entropy coef: 0 (reference), CE Weights 58 1 1 1
    # (sleep 1s && python ppo_bc2.py \
    #   --config-path "env_configs/audiogoal_rgb_nocont.yaml" \
    #   --wandb --wandb-project ss-hab-bc --wandb-entity dosssman \
    #   --exp-name "ppo_bc2__ss1_rgb_spectro__gru__bsize_128_bchnklen_32__nsteps_50__entcoef_0.0__cew_58_1_1_1" \
    #   --ce-weights 58 1 1 1 \
    #   --ent-coef 0.0 \
    #   --num-steps 50 \
    #   --num-envs 128 \
    #   --batch-chunk-length 32 \
    #   --save-videos True \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --total-steps $TOTAL_STEPS \
    #   --seed $seed \
    # ) & # >& /dev/null &
    # endregion: PPO GRU: batch_size 128; batch_chunk_len: 32;  seq_length == chunk_length: 50 (accum grad over 4 mini batches), Entropy coef: 0 (reference), CE Weights 58 1 1 1

    # region: PPO GRU: batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50 (accum grad over 4 mini batches), Entropy coef: 0 (reference), CE Weights 38.6 0.67 0.84 0.78
    # (sleep 1s && python ppo_bc2.py \
    #   --config-path "env_configs/audiogoal_rgb_nocont.yaml" \
    #   --wandb --wandb-project ss-hab-bc --wandb-entity dosssman \
    #   --exp-name "ppo_bc2__ss1_rgb_spectro__gru__bsize_32_bchnklen_32__nsteps_50__entcoef_0.0__cew_38.6_0.67_0.84_0.78" \
    #   --ce-weights 38.6 0.67 0.84 0.78 \
    #   --ent-coef 0 \
    #   --num-steps 50 \
    #   --num-envs 32 \
    #   --batch-chunk-length 32 \
    #   --save-videos True \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --total-steps $TOTAL_STEPS \
    #   --seed $seed \
    # ) & # >& /dev/null &
    # endregion: PPO GRU: batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50 (accum grad over 4 mini batches), Entropy coef: 0 (reference), CE Weights 38.6 0.67 0.84 0.78

    # region: PPO GRU: batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50 (accum grad over 4 mini batches), Entropy coef: 0 (reference), CE Weights 38.6 0.67 0.84 0.78, no grad norm
    # (sleep 1s && python ppo_bc2.py \
    #   --config-path "env_configs/audiogoal_rgb_nocont.yaml" \
    #   --wandb --wandb-project ss-hab-bc --wandb-entity dosssman \
    #   --exp-name "ppo_bc2__ss1_rgb_spectro__gru__bsize_32_bchnklen_32__nsteps_50__cew_38.6_0.67_0.84_0.78_nogradnorm" \
    #   --max-grad-norm 0 \
    #   --ce-weights 38.6 0.67 0.84 0.78 \
    #   --ent-coef 0 \
    #   --num-steps 50 \
    #   --num-envs 32 \
    #   --batch-chunk-length 32 \
    #   --save-videos True \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --total-steps $TOTAL_STEPS \
    #   --seed $seed \
    # ) & # >& /dev/null &
    # endregion: PPO GRU: batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50 (accum grad over 4 mini batches), Entropy coef: 0 (reference), CE Weights 38.6 0.67 0.84 0.78, no grad norm

    # region: PPO GRU: batch_size 10; batch_chunk_len: 10;  seq_length == chunk_length: 150 (accum grad over 4 mini batches), Entropy coef: 0 (reference), CE Weights 38.6 0.67 0.84 0.78, no grad norm
    # (sleep 1s && python ppo_bc2.py \
    #   --config-path "env_configs/audiogoal_rgb_nocont.yaml" \
    #   --wandb --wandb-project ss-hab-bc --wandb-entity dosssman \
    #   --exp-name "ppo_bc2__ss1_rgb_spectro__gru__bsize_10_bchnklen_10__nsteps_150__cew_38.6_0.67_0.84_0.78_nogradnorm" \
    #   --max-grad-norm 0 \
    #   --ce-weights 38.6 0.67 0.84 0.78 \
    #   --ent-coef 0 \
    #   --num-steps 150 \
    #   --num-envs 10 \
    #   --batch-chunk-length 10 \
    #   --save-videos True \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --total-steps $TOTAL_STEPS \
    #   --seed $seed \
    # ) & # >& /dev/null &
    # endregion: PPO GRU: batch_size 10; batch_chunk_len: 10;  seq_length == chunk_length: 150 (accum grad over 4 mini batches), Entropy coef: 0 (reference), CE Weights 38.6 0.67 0.84 0.78, no grad norm

    # region: PPO GRU: batch_size 256; batch_chunk_len: 32;  seq_length == chunk_length: 50 (accum grad over 4 mini batches), Entropy coef: 0 (reference), CE Weights 38.6 0.67 0.84 0.78, no grad norm
    # (sleep 1s && python ppo_bc2.py \
    #   --config-path "env_configs/audiogoal_rgb_nocont.yaml" \
    #   --wandb --wandb-project ss-hab-bc --wandb-entity dosssman \
    #   --exp-name "ppo_bc2__ss1_rgb_spectro__gru__bsize_256_bchnklen_32__nsteps_50__cew_38.6_0.67_0.84_0.78_nogradnorm" \
    #   --max-grad-norm 0 \
    #   --ce-weights 38.6 0.67 0.84 0.78 \
    #   --ent-coef 0 \
    #   --num-steps 50 \
    #   --num-envs 256 \
    #   --batch-chunk-length 32 \
    #   --save-videos True \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --total-steps $TOTAL_STEPS \
    #   --seed $seed \
    # ) & # >& /dev/null &
    # endregion: PPO GRU: batch_size 256; batch_chunk_len: 32;  seq_length == chunk_length: 50 (accum grad over 4 mini batches), Entropy coef: 0 (reference), CE Weights 38.6 0.67 0.84 0.78, no grad norm

    # region: PPO GRU: batch_size 64; batch_chunk_len: 32;  seq_length == chunk_length: 50 (accum grad over 4 mini batches), Entropy coef: 0 (reference), CE Weights 38.6 0.67 0.84 0.78
    # (sleep 1s && python ppo_bc2.py \
    #   --config-path "env_configs/audiogoal_rgb_nocont.yaml" \
    #   --wandb --wandb-project ss-hab-bc --wandb-entity dosssman \
    #   --exp-name "ppo_bc2__ss1_rgb_spectro__gru__bsize_64_bchnklen_32__nsteps_50__entcoef_0.0__cew_38.6_0.67_0.84_0.78" \
    #   --ce-weights 38.6 0.67 0.84 0.78 \
    #   --ent-coef 0 \
    #   --num-steps 50 \
    #   --num-envs 64 \
    #   --batch-chunk-length 32 \
    #   --save-videos True \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --total-steps $TOTAL_STEPS \
    #   --seed $seed \
    # ) & # >& /dev/null &
    # endregion: PPO GRU: batch_size 64; batch_chunk_len: 32;  seq_length == chunk_length: 50 (accum grad over 4 mini batches), Entropy coef: 0 (reference), CE Weights 38.6 0.67 0.84 0.78

    # region: PPO Perceiver batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50
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
    #   --exp-name "ppo_bc2__ss1_rgb_spectro__pgwt_gwwm__dpth_1_nlats_8_latdim_64_noSA_CAprevlats__bsize_32_bchnklen_32__nsteps_50" \
    #   --num-steps 50 \
    #   --num-envs 32 \
    #   --batch-chunk-length 32 \
    #   --total-steps $TOTAL_STEPS \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --seed $seed \
    # ) & # >& /dev/null &
    # endregion: PPO Perceiver batch_size 128; batch_chunk_len: 32;  seq_length == chunk_length: 50

    # region: PPO Perceiver batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50; entcoef 0 (reference)
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
    #   --exp-name "ppo_bc2__ss1_rgb_spectro__pgwt_gwwm__dpth_1_nlats_8_latdim_64_noSA_CAprevlats__bsize_32_bchnklen_32__nsteps_50__entcoef_0" \
    #   --ent-coef 0 \
    #   --num-steps 50 \
    #   --num-envs 32 \
    #   --batch-chunk-length 32 \
    #   --total-steps $TOTAL_STEPS \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --seed $seed \
    # ) & # >& /dev/null &
    # endregion: PPO Perceiver batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50; entcoef 0 (reference)

    # region: PPO Perceiver batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50; entcoef 0.2
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
    #   --exp-name "ppo_bc2__ss1_rgb_spectro__pgwt_gwwm__dpth_1_nlats_8_latdim_64_noSA_CAprevlats__bsize_32_bchnklen_32__nsteps_50__entcoef_0.2" \
    #   --ent-coef 0.2 \
    #   --num-steps 50 \
    #   --num-envs 32 \
    #   --batch-chunk-length 32 \
    #   --total-steps $TOTAL_STEPS \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --seed $seed \
    # ) & # >& /dev/null &
    # endregion: PPO Perceiver batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50; entcoef 0.2

    # region: PPO Perceiver batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50; entcoef 0 (reference) __cew_58_1_1_1
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
    #   --exp-name "ppo_bc2__ss1_rgb_spectro__pgwt_gwwm__dpth_1_nlats_8_latdim_64_noSA_CAprevlats__bsize_32_bchnklen_32__nsteps_50__entcoef_0__cew_58_1_1_1" \
    #   --ce-weights 58 1 1 1 \
    #   --ent-coef 0 \
    #   --num-steps 50 \
    #   --num-envs 32 \
    #   --batch-chunk-length 32 \
    #   --total-steps $TOTAL_STEPS \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --seed $seed \
    # ) & # >& /dev/null &
    # endregion: PPO Perceiver batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50; entcoef 0 (reference )__cew_58_1_1_1

    # region: PPO Perceiver batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50; entcoef 0.2 __cew_58_1_1_1
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
    #   --exp-name "ppo_bc2__ss1_rgb_spectro__pgwt_gwwm__dpth_1_nlats_8_latdim_64_noSA_CAprevlats__bsize_32_bchnklen_32__nsteps_50__entcoef_0.2__cew_58_1_1_1" \
    #   --ce-weights 58 1 1 1 \
    #   --ent-coef 0.2 \
    #   --num-steps 50 \
    #   --num-envs 32 \
    #   --batch-chunk-length 32 \
    #   --total-steps $TOTAL_STEPS \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --seed $seed \
    # ) & # >& /dev/null &
    # endregion: PPO Perceiver batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50; entcoef 0.2 __cew_58_1_1_1

    # region: PPO Perceiver batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50; entcoef 0 (reference) __cew_38.6_0.67_0.84_0.78
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
    #   --exp-name "ppo_bc2__ss1_rgb_spectro__pgwt_gwwm__dpth_1_nlats_8_latdim_64_noSA_CAprevlats__bsize_32_bchnklen_32__nsteps_50__entcoef_0__cew_38.6_0.67_0.84_0.78" \
    #   --ce-weights 38.6 0.67 0.84 0.78 \
    #   --ent-coef 0 \
    #   --num-steps 50 \
    #   --num-envs 32 \
    #   --batch-chunk-length 32 \
    #   --total-steps $TOTAL_STEPS \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --seed $seed \
    # ) & # >& /dev/null &
    # endregion: PPO Perceiver batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50; entcoef 0 (reference )__cew_38.6_0.67_0.84_0.78

    # region: PPO Perceiver batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50; entcoef 0 (reference) __cew_38.6_0.67_0.84_0.78, no grad norm
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
    #   --exp-name "ppo_bc2__ss1_rgb_spectro__pgwt_gwwm__dpth_1_nlats_8_latdim_64_noSA_CAprevlats__bsize_32_bchnklen_32__nsteps_50__cew_38.6_0.67_0.84_0.78_nogradnorm" \
    #   --max-grad-norm 0 \
    #   --ce-weights 38.6 0.67 0.84 0.78 \
    #   --ent-coef 0 \
    #   --num-steps 50 \
    #   --num-envs 32 \
    #   --batch-chunk-length 32 \
    #   --total-steps $TOTAL_STEPS \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --seed $seed \
    # ) & # >& /dev/null &
    # endregion: PPO Perceiver batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50; entcoef 0 (reference) __cew_38.6_0.67_0.84_0.78, no grad norm

    # region: PPO Perceiver batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50; Modeality embedings 128 entcoef 0 (reference) __cew_38.6_0.67_0.84_0.78, no grad norm
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
    #   --exp-name "ppo_bc2__ss1_rgb_spectro__pgwt_gwwm__dpth_1_nlats_8_latdim_64_noSA_CAprevlats__bsize_32_bchnklen_32__nsteps_50__modembed_128__cew_38.6_0.67_0.84_0.78_nogradnorm" \
    #   --max-grad-norm 0 \
    #   --pgwt-mod-embed 128 \
    #   --ce-weights 38.6 0.67 0.84 0.78 \
    #   --ent-coef 0 \
    #   --num-steps 50 \
    #   --num-envs 32 \
    #   --batch-chunk-length 32 \
    #   --total-steps $TOTAL_STEPS \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --seed $seed \
    # ) & # >& /dev/null &
    # endregion: PPO Perceiver batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50; Modeality embedings 128 entcoef 0 (reference) __cew_38.6_0.67_0.84_0.78, no grad norm

    # region: PPO Perceiver batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50; entcoef 0 (reference) __cew_38.6_0.67_0.84_0.78, no grad norm
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
    #   --exp-name "ppo_bc2__ss1_rgb_spectro__pgwt_gwwm__dpth_1_nlats_8_latdim_64_noSA_CAprevlats__bsize_10_bchnklen_10__nsteps_150__cew_38.6_0.67_0.84_0.78_nogradnorm" \
    #   --max-grad-norm 0 \
    #   --ce-weights 38.6 0.67 0.84 0.78 \
    #   --ent-coef 0 \
    #   --num-steps 150 \
    #   --num-envs 10 \
    #   --batch-chunk-length 10 \
    #   --total-steps $TOTAL_STEPS \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --seed $seed \
    # ) & # >& /dev/null &
    # endregion: PPO Perceiver batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50; entcoef 0 (reference) __cew_38.6_0.67_0.84_0.78, no grad norm

    # region: PPO Perceiver batch_size 256; batch_chunk_len: 32;  seq_length == chunk_length: 50; entcoef 0 (reference) __cew_38.6_0.67_0.84_0.78, no grad norm
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
    #   --exp-name "ppo_bc2__ss1_rgb_spectro__pgwt_gwwm__dpth_1_nlats_8_latdim_64_noSA_CAprevlats__bsize_256_bchnklen_32__nsteps_50__cew_38.6_0.67_0.84_0.78_nogradnorm" \
    #   --max-grad-norm 0 \
    #   --ce-weights 38.6 0.67 0.84 0.78 \
    #   --ent-coef 0 \
    #   --num-steps 50 \
    #   --num-envs 256 \
    #   --batch-chunk-length 32 \
    #   --total-steps $TOTAL_STEPS \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --seed $seed \
    # ) & # >& /dev/null &
    # endregion: PPO Perceiver batch_size 256; batch_chunk_len: 32;  seq_length == chunk_length: 50; entcoef 0 (reference) __cew_38.6_0.67_0.84_0.78, no grad norm

    # region: PPO Perceiver batch_size 64; batch_chunk_len: 32;  seq_length == chunk_length: 50; entcoef 0 (reference) __cew_38.6_0.67_0.84_0.78
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
    #   --exp-name "ppo_bc2__ss1_rgb_spectro__pgwt_gwwm__dpth_1_nlats_8_latdim_64_noSA_CAprevlats__bsize_64_bchnklen_32__nsteps_50__entcoef_0__cew_38.6_0.67_0.84_0.78" \
    #   --ce-weights 38.6 0.67 0.84 0.78 \
    #   --ent-coef 0 \
    #   --num-steps 50 \
    #   --num-envs 64 \
    #   --batch-chunk-length 32 \
    #   --total-steps $TOTAL_STEPS \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --seed $seed \
    # ) & # >& /dev/null &
    # endregion: PPO Perceiver batch_size 64; batch_chunk_len: 32;  seq_length == chunk_length: 50; entcoef 0 (reference )__cew_38.6_0.67_0.84_0.78


    # region: PPO Perceiver batch_size 128; batch_chunk_len: 32;  seq_length == chunk_length: 50
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
    #   --exp-name "ppo_bc2__ss1_rgb_spectro__pgwt_gwwm__dpth_1_nlats_8_latdim_64_noSA_CAprevlats__bsize_64_bchnklen_32__nsteps_50" \
    #   --num-steps 50 \
    #   --num-envs 128 \
    #   --batch-chunk-length 32 \
    #   --total-steps $TOTAL_STEPS \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --seed $seed \
    # ) & # >& /dev/null &
    # endregion: PPO Perceiver batch_size 64; batch_chunk_len: 32;  seq_length == chunk_length: 50

    # region: PPO GRU with Aux Vloss varaint: batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50 (accum grad over 4 mini batches)
    # (sleep 1s && python ppo_bc2_vfnloss.py \
    #   --config-path "env_configs/audiogoal_rgb_nocont.yaml" \
    #   --wandb --wandb-project ss-hab-bc --wandb-entity dosssman \
    #   --exp-name "ppo_bc2_vfnloss__ss1_rgb_spectro__gru__bsize_32_bchnklen_32__nsteps_50" \
    #   --num-steps 50 \
    #   --num-envs 32 \
    #   --batch-chunk-length 32 \
    #   --save-videos True \
    #   --logdir-prefix $LOGDIR_PREFIX \
    #   --total-steps $TOTAL_STEPS \
    #   --seed $seed \
    # ) & # >& /dev/null &
    # endregion: PPO GRU: batch_size 64; batch_chunk_len: 32;  seq_length == chunk_length: 50 (accum grad over 4 mini batches)


  done
  # region: PPO GRU | Perceiver - BC with default hyparams, search over batch -size

# endregion: PPO BC #

# Clean up env vars
export CUDA_VISIBLE_DEVICES=
export LOGDIR_PREFIX=
export MASTER_PORT=
export WANDB_DIR=