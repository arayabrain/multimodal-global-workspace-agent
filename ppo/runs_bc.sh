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

  # region: AvNav

    # region: PPO GRU - BC with RGB only for vision, cv2 resize variant for eval envs
    # for seed in 111 222; do
    #   # export MASTER_PORT=8738 # Default port is 8738
    #   export TOTAL_STEPS=2500000
    #   (sleep 1s && python ppo_bc_cv2resize.py \
    #       --exp-name "ppo_bc_cv2resize__ss1__rgb_spectro__gru" \
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
    # endregion: PPO GRU - BC with RGB only for vision, cv2 resize variant for eval envs

    # region: Custom PPO + Perceiver GWT GWWM Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents; RGB + Spectrogram SS1
    # for seed in 111 222; do
    #   # export MASTER_PORT=8748 # Default port is 8738
    #   export TOTAL_STEPS=2500000
    #   (sleep 1s && python ppo_bc_cv2resize.py \
    #       --exp-name "ppo_bc_cv2resize__ss1__rgb_depth_spectro__pgwt__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats" \
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


    ## Deprecated runs below

    # region: PPO GRU - BC with RGB and Depth for vision
    # Reason for deprecation: was not properly cv2.resizing the observation.
    # The eval metrics where therefore uunreliable.
    # for seed in 111 222; do
    #   # export MASTER_PORT=8738 # Default port is 8738
    #   export TOTAL_STEPS=2500000
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
    # Reason for deprecation: was not properly cv2.resizing the observation.
    # The eval metrics where therefore uunreliable.
    # for seed in 111 222; do
    #   # export MASTER_PORT=8738 # Default port is 8738
    #   export TOTAL_STEPS=2500000
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


    # Older experiments scripts runs
    # region: PPO GRU: batch_size 32; batch_chunk_len: 32;  seq_length == chunk_length: 50 (accum grad over 4 mini batches), Entropy coef: 0 (reference), CE Weights 38.6 0.67 0.84 0.78, no grad norm
    # Reason for deprecation: was not properly cv2.resizing the observation.
    # The eval metrics where therefore uunreliable.
    # for seed in 111 222; do
    #   export TOTAL_STEPS=2500000
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

    ## RGB + Spectrogram based section

      # region: PPO GRU - BC with default hyparams
      # for seed in 111 222; do
      #   # export MASTER_PORT=8738 # Default port is 8738
      #   export TOTAL_STEPS=10000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_spectro__gru" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --save-videos False \
      #     --ent-coef 0 \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --total-steps $TOTAL_STEPS \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: PPO GRU - BC with default hyparams

      # region: PPO GRU 2 - BC with default hyparams, with pose
      # for seed in 111 222; do
      # # for seed in 111 333; do
      #   # export MASTER_PORT=8738 # Default port is 8738
      #   export TOTAL_STEPS=20000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_spectro__gru2" \
      #     --agent-type "custom-gru" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --save-videos False \
      #     --ent-coef 0 \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --total-steps $TOTAL_STEPS \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: PPO GRU2 - BC with default hyparams

      # region: PPO Custom GWT with GRU - BC with default hyparams
      # for seed in 111; do
      # for seed in 222; do
      # for seed in 333; do
      # for seed in 111 222; do
      #   # export MASTER_PORT=8738 # Default port is 8738
      #   export TOTAL_STEPS=20000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_spectro__gwt_bu_td" \
      #     --agent-type "custom-gwt" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --save-videos False \
      #     --ent-coef 0 \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --total-steps $TOTAL_STEPS \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: PPO Custom GWT with GRU - BC with default hyparams

      # GWTv2
      # region: PPO Custom GWT with GRU BU - BC with default hyparams
      # for seed in 111 222; do
      #   # export MASTER_PORT=8738 # Default port is 8738
      #   export TOTAL_STEPS=10000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_spectro__gwt_bu" \
      #     --agent-type "custom-gwt-bu" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --save-videos False \
      #     --ent-coef 0 \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --total-steps $TOTAL_STEPS \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: PPO Custom GWT with GRU BU - BC with default hyparams

      # region: PPO Custom GWT with GRU BU - BC with default hyparams
      # for seed in 111 222; do
      #   # export MASTER_PORT=8738 # Default port is 8738
      #   export TOTAL_STEPS=10000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_spectro__gwt_td" \
      #     --agent-type "custom-gwt-td" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --save-videos False \
      #     --ent-coef 0 \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --total-steps $TOTAL_STEPS \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: PPO Custom GWT with GRU BU - BC with default hyparams

      # GWTv1
      # region: Custom PPO + Perceiver GWT GWWM Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents; RGB + Spectrogram SS1
      # for seed in 111 222; do
      # for seed in 111 333; do
      #   # export MASTER_PORT=8748 # Default port is 8738
      #   export TOTAL_STEPS=10000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb__spectro__pgwt__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --agent-type "perceiver-gwt-gwwm" \
      #     --ent-coef 0 \
      #     --pgwt-depth 1 \
      #     --pgwt-num-latents 8 \
      #     --pgwt-latent-dim 64 \
      #     --pgwt-cross-heads 1 \
      #     --pgwt-latent-heads 4 \
      #     --pgwt-use-sa False \
      #     --pgwt-mod-embed 0 \
      #     --pgwt-ca-prev-latents True \
      #     --save-videos False \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --total-steps $TOTAL_STEPS \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: Custom PPO + Perceiver GWT GWWM Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents; RGB + Spectrogram SS1

      # region: Custom PPO + Perceiver GWT GWWM Basic Arch. NoSA Cross Heads 3 SA Heads 4 mod_emb 0 CA Prev Latents; RGB + Spectrogram SS1
      # for seed in 111 222; do
      #   # export MASTER_PORT=8748 # Default port is 8738
      #   export TOTAL_STEPS=10000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb__spectro__pgwt__dpth_1_nlats_8_latdim_64_noSA_CAnheads_4_SAnheads_4_modembed_0_CAprevlats" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --agent-type "perceiver-gwt-gwwm" \
      #     --ent-coef 0 \
      #     --pgwt-depth 1 \
      #     --pgwt-num-latents 8 \
      #     --pgwt-latent-dim 64 \
      #     --pgwt-cross-heads 4 \
      #     --pgwt-latent-heads 4 \
      #     --pgwt-use-sa False \
      #     --pgwt-mod-embed 0 \
      #     --pgwt-ca-prev-latents True \
      #     --save-videos False \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --total-steps $TOTAL_STEPS \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: Custom PPO + Perceiver GWT GWWM Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents; RGB + Spectrogram SS1

      # GWTv3

      # region: PPO GWTv3 - BC | gw at rec enc level, detached, use null, GRU Layer Norm, entropy reg 0.2, no ce weights
      # for seed in 111 222; do
      #   export TOTAL_STEPS=20000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_spectro__gwtv3__gw_detach__usenull__grulynrm__gwfix__entcoef_0.2__no_cew" \
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
      for seed in 111; do
        export TOTAL_STEPS=20000000
        (sleep 1s && python ppo_bc.py \
          --exp-name "ppo_bc__savi_ss1_rgb_spectro__gwtv3__gw_detach__usenull__grulynrm__gwfix__entcoef_0.2__no_cew__h_64" \
          --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
          --hidden-size 64 \
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
      #     --save-videos False \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --total-steps $TOTAL_STEPS \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: PPO GRUv3 - BC | gw at rec enc level, detached; GRU Layer Norm, entropy reg 0.2, no ce weights

      # region: PPO GWTv3 - BC | gw at rec enc level, detached, use null, GRU Layer Norm
      # for seed in 111 222; do
      #   export TOTAL_STEPS=20000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_spectro__gwtv3__gw_detach__usenull__grulynrm__gwfix" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --agent-type "gwtv3" \
      #     --gwtv3-use-gw "True" \
      #     --gwtv3-use-null "True" \
      #     --gwtv3-enc-gw-detach True \
      #     --gwtv3-gru-type "layernorm" \
      #     --save-videos False \
      #     --ent-coef 0 \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --total-steps $TOTAL_STEPS \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: PPO GWTv3 - BC | gw at rec enc level, detached, use null, GRU Layer Norm

    ## RGB + Spectrogram based section, with RGB obs centered at [-0.5, 0.5] instead of [0, 1]

      # region: PPO GRU - BC with default hyparams
      # for seed in 111 222; do
      #   # export MASTER_PORT=8738 # Default port is 8738
      #   export TOTAL_STEPS=10000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_cntr_spectro__gru" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --obs-center True \
      #     --save-videos False \
      #     --ent-coef 0 \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --total-steps $TOTAL_STEPS \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: PPO GRU - BC with default hyparams

      # region: Custom PPO + Perceiver GWT GWWM Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents; RGB + Spectrogram SS1
      # for seed in 111 222; do
      #   # export MASTER_PORT=8748 # Default port is 8738
      #   export TOTAL_STEPS=10000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_cntr__spectro__pgwt__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --obs-center True \
      #     --agent-type "perceiver-gwt-gwwm" \
      #     --ent-coef 0 \
      #     --pgwt-depth 1 \
      #     --pgwt-num-latents 8 \
      #     --pgwt-latent-dim 64 \
      #     --pgwt-cross-heads 1 \
      #     --pgwt-latent-heads 4 \
      #     --pgwt-use-sa False \
      #     --pgwt-mod-embed 0 \
      #     --pgwt-ca-prev-latents True \
      #     --save-videos False \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --total-steps $TOTAL_STEPS \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: Custom PPO + Perceiver GWT GWWM Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents; RGB + Spectrogram SS1


    ## RGB + Spectrogram based section, SSL : rec-rgb

      # region: PPO GRU - BC with default hyparams, rec-rgb
      # for seed in 111; do
      #   # export MASTER_PORT=8738 # Default port is 8738
      #   export TOTAL_STEPS=10000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_cntr_spectro__gru__rec_rgb" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --obs-center True \
      #     --ssl-tasks "rec-rgb" \
      #     --save-videos False \
      #     --ent-coef 0 \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --total-steps $TOTAL_STEPS \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: PPO GRU - BC with default hyparams, rec-rgb


      # region: PPO GRU v2, rec-rgb-vis-ae-4 , detach rec features
      # By detaching the features for the rec-rgb SSL task, the reconstruction
      # just allows us to qualitatively evaluate the quality of the latent
      # SSL in this case will not contribute to learning better features.
      # for seed in 111; do
      #   # export MASTER_PORT=8738 # Default port is 8738
      #   export TOTAL_STEPS=10000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_cntr_spectro__gru2__rec_rgb_vis_ae_4_sslfeat_detach" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --num-envs 5 \
      #     --agent-type "custom-gru" \
      #     --obs-center True \
      #     --ssl-tasks "rec-rgb-vis-ae-4" \
      #     --ssl-rec-rgb-detach True \
      #     --save-videos False \
      #     --ent-coef 0 \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --total-steps $TOTAL_STEPS \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: PPO GRU v2, rec-rgb-vis-ae-4 , detach rec features

      # region: PPO GRU v2, rec-rgb-vis-ae-4 , backprop rec-rgb 
      # Backprops from the rec-rgb loss through decoder -> vision features -> encoder
      # Should contribute to learning better features ?
      # for seed in 111; do
      #   # export MASTER_PORT=8738 # Default port is 8738
      #   export TOTAL_STEPS=10000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_cntr_spectro__gru2__rec_rgb_vis_ae_4_sslfeat_nodetach" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --num-envs 5 \
      #     --agent-type "custom-gru" \
      #     --obs-center True \
      #     --ssl-tasks "rec-rgb-vis-ae-4" \
      #     --ssl-rec-rgb-detach False \
      #     --save-videos False \
      #     --ent-coef 0 \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --total-steps $TOTAL_STEPS \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: PPO GRU v2, rec-rgb-vis-ae-4 , backprop rec-rgb 

      # region: PPO GRU v2, rec-rgb-vis-ae-4 , backprop rec-rgb, H=1024
      # Backprops from the rec-rgb loss through decoder -> vision features -> encoder
      # Should contribute to learning better features ?
      # Preliminary run with H=512 suggest the latent size might not be large enough
      # How about just using 1024 ?
      # for seed in 111; do
      #   # export MASTER_PORT=8738 # Default port is 8738
      #   export TOTAL_STEPS=10000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_cntr_spectro__gru2__rec_rgb_vis_ae_4_sslfeat_nodetach_H_1024" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --num-envs 5 \
      #     --hidden-size 1024 \
      #     --agent-type "custom-gru" \
      #     --obs-center True \
      #     --ssl-tasks "rec-rgb-vis-ae-4" \
      #     --ssl-rec-rgb-detach False \
      #     --save-videos False \
      #     --ent-coef 0 \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --total-steps $TOTAL_STEPS \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: PPO GRU v2, rec-rgb-vis-ae-4 , backprop rec-rgb, H=1024


      # region: PPO GRU v2, rec-rgb-vis-ae-5 , backprop rec-rgb , use vision features
      # Backprops from the rec-rgb loss through decoder -> vision features -> encoder
      # Should contribute to learning better features ?
      # for seed in 111; do
      #   # export MASTER_PORT=8738 # Default port is 8738
      #   export TOTAL_STEPS=10000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_cntr_spectro__gru2__rec_rgb_vis_ae_5_sslfeat_nodetach" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --num-envs 10 \
      #     --num-steps 50 \
      #     --agent-type "custom-gru" \
      #     --obs-center True \
      #     --ssl-tasks "rec-rgb-vis-ae-5" \
      #     --ssl-rec-rgb-mid-size 1536 \
      #     --ssl-rec-rgb-mid-feat False \
      #     --ssl-rec-rgb-detach False \
      #     --save-videos False \
      #     --ent-coef 0 \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --total-steps $TOTAL_STEPS \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: PPO GRU v2, rec-rgb-vis-ae-5 , backprop rec-rgb , use vision features

      # region: PPO GRU v2, rec-rgb-vis-ae-5 , backprop rec-rgb , use intermediate, larger vis. features
      # Backprops from the rec-rgb loss through decoder -> vision features -> encoder
      # Should contribute to learning better features ?
      # for seed in 111; do
      #   # export MASTER_PORT=8738 # Default port is 8738
      #   export TOTAL_STEPS=10000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_cntr_spectro__gru2__rec_rgb_vis_ae_5_sslfeat_nodetach__vis_mid_feats" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --num-envs 10 \
      #     --num-steps 50 \
      #     --agent-type "custom-gru" \
      #     --obs-center True \
      #     --ssl-tasks "rec-rgb-vis-ae-5" \
      #     --ssl-rec-rgb-mid-size 1536 \
      #     --ssl-rec-rgb-mid-feat True \
      #     --ssl-rec-rgb-detach False \
      #     --save-videos False \
      #     --ent-coef 0 \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --total-steps $TOTAL_STEPS \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: PPO GRU v2, rec-rgb-vis-ae-5 , backprop rec-rgb , use intermediate, larger vis. features

      # region: PPO GRU v2, rec-rgb-vis-ae-5 , backprop rec-rgb , use intermediate, larger vis. features, no max grad norm
      # Backprops from the rec-rgb loss through decoder -> vision features -> encoder
      # Should contribute to learning better features ?
      # for seed in 111; do
      #   # export MASTER_PORT=8738 # Default port is 8738
      #   export TOTAL_STEPS=10000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_cntr_spectro__gru2__rec_rgb_vis_ae_5_sslfeat_nodetach__vis_mid_feats_gradnorm_100" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --num-envs 10 \
      #     --num-steps 50 \
      #     --max-grad-norm 100 \
      #     --agent-type "custom-gru" \
      #     --obs-center True \
      #     --ssl-tasks "rec-rgb-vis-ae-5" \
      #     --ssl-rec-rgb-mid-size 1536 \
      #     --ssl-rec-rgb-mid-feat True \
      #     --ssl-rec-rgb-detach False \
      #     --save-videos False \
      #     --ent-coef 0 \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --total-steps $TOTAL_STEPS \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: PPO GRU v2, rec-rgb-vis-ae-5 , backprop rec-rgb , use intermediate, larger vis. features, no max grad norm


      # region: Custom PPO + Perceiver GWT GWWM Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents; RGB + Spectrogram SS1, rec-rgb
      # for seed in 111; do
      #   # export MASTER_PORT=8748 # Default port is 8738
      #   export TOTAL_STEPS=10000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_cntr__spectro__pgwt__rec_rgb__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --obs-center True \
      #     --ssl-tasks "rec-rgb" \
      #     --agent-type "perceiver-gwt-gwwm" \
      #     --ent-coef 0 \
      #     --pgwt-depth 1 \
      #     --pgwt-num-latents 8 \
      #     --pgwt-latent-dim 64 \
      #     --pgwt-cross-heads 1 \
      #     --pgwt-latent-heads 4 \
      #     --pgwt-use-sa False \
      #     --pgwt-mod-embed 0 \
      #     --pgwt-ca-prev-latents True \
      #     --save-videos False \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --total-steps $TOTAL_STEPS \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: Custom PPO + Perceiver GWT GWWM Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents; RGB + Spectrogram SS1, rec-rgb

      # region: Custom PPO + Perceiver GWT GWWM Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents; RGB + Spectrogram SS1, rec-rgb
      # for seed in 111; do
      #   # export MASTER_PORT=8748 # Default port is 8738
      #   export TOTAL_STEPS=10000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_cntr__spectro__pgwt__rec_rgb__nogrdnrm__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --obs-center True \
      #     --ssl-tasks "rec-rgb" \
      #     --max-grad-norm 0 \
      #     --agent-type "perceiver-gwt-gwwm" \
      #     --ent-coef 0 \
      #     --pgwt-depth 1 \
      #     --pgwt-num-latents 8 \
      #     --pgwt-latent-dim 64 \
      #     --pgwt-cross-heads 1 \
      #     --pgwt-latent-heads 4 \
      #     --pgwt-use-sa False \
      #     --pgwt-mod-embed 0 \
      #     --pgwt-ca-prev-latents True \
      #     --save-videos False \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --total-steps $TOTAL_STEPS \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: Custom PPO + Perceiver GWT GWWM Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents; RGB + Spectrogram SS1, rec-rgb

      # region: Custom PPO + Perceiver GWT GWWM ; RGB + Spectrogram SS1, rec-rgb-vis-ae
      # Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents
      # for seed in 111; do
      #   # export MASTER_PORT=8748 # Default port is 8738
      #   export TOTAL_STEPS=10000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_cntr__spectro__pgwt__rec_rgb_vis_ae__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --obs-center True \
      #     --ssl-tasks "rec-rgb-vis-ae" \
      #     --agent-type "perceiver-gwt-gwwm" \
      #     --ent-coef 0 \
      #     --pgwt-depth 1 \
      #     --pgwt-num-latents 8 \
      #     --pgwt-latent-dim 64 \
      #     --pgwt-cross-heads 1 \
      #     --pgwt-latent-heads 4 \
      #     --pgwt-use-sa False \
      #     --pgwt-mod-embed 0 \
      #     --pgwt-ca-prev-latents True \
      #     --save-videos False \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --total-steps $TOTAL_STEPS \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: Custom PPO + Perceiver GWT GWWM ; RGB + Spectrogram SS1, rec-rgb-vis-ae
      # Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents

      # region: Custom PPO + Perceiver GWT GWWM ; RGB + Spectrogram SS1, rec-rgb-ae, H=1024 eq. to Latents: 16 * 64
      # Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents
      # for seed in 111; do
      #   # export MASTER_PORT=8748 # Default port is 8738
      #   export TOTAL_STEPS=10000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_cntr__spectro__pgwt__rec_rgb_ae_H_1024__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --obs-center True \
      #     --ssl-tasks "rec-rgb-ae" \
      #     --hidden-size 1024 \
      #     --agent-type "perceiver-gwt-gwwm" \
      #     --ent-coef 0 \
      #     --pgwt-depth 1 \
      #     --pgwt-num-latents 16 \
      #     --pgwt-latent-dim 64 \
      #     --pgwt-cross-heads 1 \
      #     --pgwt-latent-heads 4 \
      #     --pgwt-use-sa False \
      #     --pgwt-mod-embed 0 \
      #     --pgwt-ca-prev-latents True \
      #     --save-videos False \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --total-steps $TOTAL_STEPS \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: Custom PPO + Perceiver GWT GWWM ; RGB + Spectrogram SS1, rec-rgb-ae, H=1024 eq. to Latents: 16 * 64
      # Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents

      # region: Custom PPO + Perceiver GWT GWWM ; RGB + Spectrogram  SS1, rec-rgb-vis-ae, H=1024 eq. to Latents: 16 * 64
      # Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents
      # for seed in 111; do
      #   # export MASTER_PORT=8748 # Default port is 8738
      #   export TOTAL_STEPS=10000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_cntr__spectro__pgwt__rec_rgb_vis_ae_H_1024__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --obs-center True \
      #     --ssl-tasks "rec-rgb-vis-ae" \
      #     --hidden-size 1024 \
      #     --agent-type "perceiver-gwt-gwwm" \
      #     --ent-coef 0 \
      #     --pgwt-depth 1 \
      #     --pgwt-num-latents 16 \
      #     --pgwt-latent-dim 64 \
      #     --pgwt-cross-heads 1 \
      #     --pgwt-latent-heads 4 \
      #     --pgwt-use-sa False \
      #     --pgwt-mod-embed 0 \
      #     --pgwt-ca-prev-latents True \
      #     --save-videos False \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --total-steps $TOTAL_STEPS \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: Custom PPO + Perceiver GWT GWWM ; RGB + Spectrogram SS1, rec-rgb-vis-ae, H=1024 eq. to Latents: 16 * 64
      # Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents

      # region: Custom PPO + Perceiver GWT GWWM ; RGB + Spectrogram SS1, rec-rgb-vis-ae-2
      # Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents
      # for seed in 111; do
      #   # export MASTER_PORT=8748 # Default port is 8738
      #   export TOTAL_STEPS=10000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_cntr__spectro__pgwt__rec_rgb_ae_2__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --obs-center True \
      #     --ssl-tasks "rec-rgb-ae-2" \
      #     --agent-type "perceiver-gwt-gwwm" \
      #     --ent-coef 0 \
      #     --pgwt-depth 1 \
      #     --pgwt-num-latents 8 \
      #     --pgwt-latent-dim 64 \
      #     --pgwt-cross-heads 1 \
      #     --pgwt-latent-heads 4 \
      #     --pgwt-use-sa False \
      #     --pgwt-mod-embed 0 \
      #     --pgwt-ca-prev-latents True \
      #     --save-videos False \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --total-steps $TOTAL_STEPS \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: Custom PPO + Perceiver GWT GWWM ; RGB + Spectrogram SS1, rec-rgb-vis-ae-2
      # Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents

      # region: Custom PPO + Perceiver GWT GWWM ; RGB + Spectrogram SS1, rec-rgb-vis-ae-3
      # Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents
      # for seed in 111 222; do
      #   # export MASTER_PORT=8748 # Default port is 8738
      #   export TOTAL_STEPS=10000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_cntr__spectro__pgwt__rec_rgb_vis_ae_3__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --obs-center True \
      #     --ssl-tasks "rec-rgb-vis-ae-3" \
      #     --agent-type "perceiver-gwt-gwwm" \
      #     --ent-coef 0 \
      #     --pgwt-depth 1 \
      #     --pgwt-num-latents 8 \
      #     --pgwt-latent-dim 64 \
      #     --pgwt-cross-heads 1 \
      #     --pgwt-latent-heads 4 \
      #     --pgwt-use-sa False \
      #     --pgwt-mod-embed 0 \
      #     --pgwt-ca-prev-latents True \
      #     --save-videos False \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --total-steps $TOTAL_STEPS \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: Custom PPO + Perceiver GWT GWWM ; RGB + Spectrogram SS1, rec-rgb-vis-ae-3
      # Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents

      # region: Custom PPO + Perceiver GWT GWWM ; RGB + Spectrogram SS1, rec-rgb-vis-ae-3, no grad norm
      # Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents
      # for seed in 111; do
      #   # export MASTER_PORT=8748 # Default port is 8738
      #   export TOTAL_STEPS=10000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_cntr__spectro__pgwt__rec_rgb_vis_ae_3__nogrdnrm__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --obs-center True \
      #     --max-grad-norm 0 \
      #     --ssl-tasks "rec-rgb-vis-ae-3" \
      #     --agent-type "perceiver-gwt-gwwm" \
      #     --ent-coef 0 \
      #     --pgwt-depth 1 \
      #     --pgwt-num-latents 8 \
      #     --pgwt-latent-dim 64 \
      #     --pgwt-cross-heads 1 \
      #     --pgwt-latent-heads 4 \
      #     --pgwt-use-sa False \
      #     --pgwt-mod-embed 0 \
      #     --pgwt-ca-prev-latents True \
      #     --save-videos False \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --total-steps $TOTAL_STEPS \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: Custom PPO + Perceiver GWT GWWM ; RGB + Spectrogram SS1, rec-rgb-vis-ae-3, no grad norm
      # Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents

      # region: Custom PPO + Perceiver GWT GWWM ; RGB + Spectrogram SS1, rec-rgb-vis-ae-3
      # Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents
      # for seed in 111; do
      #   # export MASTER_PORT=8748 # Default port is 8738
      #   export TOTAL_STEPS=10000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_rgb_cntr__spectro__pgwt__rec_rgb_vis_ae_mse__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats" \
      #     --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      #     --obs-center True \
      #     --ssl-tasks "rec-rgb-vis-ae-mse" \
      #     --agent-type "perceiver-gwt-gwwm" \
      #     --ent-coef 0 \
      #     --pgwt-depth 1 \
      #     --pgwt-num-latents 8 \
      #     --pgwt-latent-dim 64 \
      #     --pgwt-cross-heads 1 \
      #     --pgwt-latent-heads 4 \
      #     --pgwt-use-sa False \
      #     --pgwt-mod-embed 0 \
      #     --pgwt-ca-prev-latents True \
      #     --save-videos False \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --total-steps $TOTAL_STEPS \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: Custom PPO + Perceiver GWT GWWM ; RGB + Spectrogram SS1, rec-rgb-vis-ae-3
      # Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents

    ## Spectrogram only
    
      # region: PPO GRU - BC with default hyparams
      # for seed in 111 222; do
      #   # export MASTER_PORT=8738 # Default port is 8738
      #   export TOTAL_STEPS=10000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_spectro__gru" \
      #     --config-path "env_configs/savi/savi_ss1_spectro.yaml" \
      #     --save-videos False \
      #     --ent-coef 0 \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --total-steps $TOTAL_STEPS \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: PPO GRU - BC with default hyparams

      # region: Custom PPO + Perceiver GWT GWWM Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents; RGB + Spectrogram SS1
      # for seed in 111 222; do
      #   # export MASTER_PORT=8748 # Default port is 8738
      #   export TOTAL_STEPS=10000000
      #   (sleep 1s && python ppo_bc.py \
      #     --exp-name "ppo_bc__savi_ss1_spectro__pgwt__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats" \
      #     --config-path "env_configs/savi/savi_ss1_spectro.yaml" \
      #     --agent-type "perceiver-gwt-gwwm" \
      #     --ent-coef 0 \
      #     --pgwt-depth 1 \
      #     --pgwt-num-latents 8 \
      #     --pgwt-latent-dim 64 \
      #     --pgwt-cross-heads 1 \
      #     --pgwt-latent-heads 4 \
      #     --pgwt-use-sa False \
      #     --pgwt-mod-embed 0 \
      #     --pgwt-ca-prev-latents True \
      #     --save-videos False \
      #     --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #     --total-steps $TOTAL_STEPS \
      #     --logdir-prefix $LOGDIR_PREFIX \
      #     --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: Custom PPO + Perceiver GWT GWWM Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents; RGB + Spectrogram SS1


    ## RGB + Depth + Spectrogram

      # region: PPO GRU - BC with RGBD + Spectrogram
      # for seed in 111 222; do
      #   # export MASTER_PORT=8738 # Default port is 8738
      #   export TOTAL_STEPS=5000000
      #   (sleep 1s && python ppo_bc.py \
      #       --exp-name "ppo_bc__savi_ss1_rgbd_spectro__gru" \
      #       --config-path "env_configs/savi/savi_ss1_rgbd_spectro.yaml" \
      #       --save-videos False \
      #       --ent-coef 0 \
      #       --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
      #       --logdir-prefix $LOGDIR_PREFIX \
      #       --total-steps $TOTAL_STEPS \
      #       --seed $seed \
      #   ) & # >& /dev/null &
      # done
      # endregion: PPO GRU - BC with RGBD + Spectrogram

      # region: Custom PPO + Perceiver GWT GWWM Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents; RGBD + Spectrogram SS1
    # for seed in 111 222; do
    #   # export MASTER_PORT=8748 # Default port is 8738
    #   export TOTAL_STEPS=5000000
    #   (sleep 1s && python ppo_bc.py \
    #       --exp-name "ppo_bc__savi_ss1_rgbd__spectro__pgwt__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats" \
    #       --config-path "env_configs/savi/savi_ss1_rgbd_spectro.yaml" \
    #       --agent-type "perceiver-gwt-gwwm" \
    #       --ent-coef 0 \
    #       --pgwt-depth 1 \
    #       --pgwt-num-latents 8 \
    #       --pgwt-latent-dim 64 \
    #       --pgwt-cross-heads 1 \
    #       --pgwt-latent-heads 4 \
    #       --pgwt-use-sa False \
    #       --pgwt-mod-embed 0 \
    #       --pgwt-ca-prev-latents True \
    #       --save-videos False \
    #       --wandb --wandb-project "ss-hab-bc" --wandb-entity dosssman \
    #       --total-steps $TOTAL_STEPS \
    #       --logdir-prefix $LOGDIR_PREFIX \
    #       --seed $seed \
    #   ) & # >& /dev/null &
    # done
    # endregion: Custom PPO + Perceiver GWT GWWM Basic Arch. NoSA Cross Heads 1 SA Heads 4 mod_emb 0 CA Prev Latents; RGBD + Spectrogram SS1

  # endregion: SAVI
# endregion: PPO BC #

# Clean up env vars
# export CUDA_VISIBLE_DEVICES=
export LOGDIR_PREFIX=
export MASTER_PORT=
export WANDB_DIR=
