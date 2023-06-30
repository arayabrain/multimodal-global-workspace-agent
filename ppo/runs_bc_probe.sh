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

## RNN agents get inputs of shape T * B = 150 * 10, but the input for the probes is broken into batches of size 30.
# region: Probing ppo_bc__rgb_spectro__gru__SAVi (No depth) @10M steps, Linear probe with minibatch train
# for seed in 111; do
#     TOTAL_STEPS=500000; N_EPOCHS=10;
#     (sleep 1s && python ppo_bc_probe_train_mb.py \
#         --exp-name "ppo_bc__rgb_spectro__gru__SAVi__9990001__n_mb_50" \
#         --num-minibatches 50 \
#         --pretrained-model-name "ppo_bc__rgb_spectro__gru__SAVi" \
#         --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc/ppo_bc__savi_ss1_rgb_spectro__gru_seed_222__2023_06_17_21_24_12_718867.musashi/models/ppo_agent.9990001.ckpt.pth" \
#         --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
#         --save-videos False \
#         --ent-coef 0 \
#         --wandb --wandb-project "ss-hab-bc-probing" --wandb-entity dosssman \
#         --logdir-prefix $LOGDIR_PREFIX \
#         --total-steps $TOTAL_STEPS \
#         --n-epochs $N_EPOCHS \
#         --seed $seed \
#     ) & # >& /dev/null &
# done
# endregion: Probing ppo_bc__rgb_spectro__gru__SAVi (No depth) @10M steps, Linear probe with minibatch train

# region: Probing ppo_bc__rgb_spectro__pgwt__SAVi (No depth) @10M steps, Linear probe with minibatch train
# for seed in 111; do
#     TOTAL_STEPS=500000; N_EPOCHS=10;
#     (sleep 1s && python ppo_bc_probe_train_mb.py \
#         --exp-name "ppo_bc__rgb_spectro__pgwt__SAVi__9990001__n_mb_50" \
#         --num-minibatches 50 \
#         --pretrained-model-name "ppo_bc__rgb_spectro__pgwt__SAVi" \
#         --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc/ppo_bc__savi_ss1_rgb__spectro__pgwt__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats_seed_222__2023_06_17_21_24_10_884437.musashi/models/ppo_agent.9990001.ckpt.pth" \
#         --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
#         --save-videos False \
#         --ent-coef 0 \
#         --wandb --wandb-project "ss-hab-bc-probing" --wandb-entity dosssman \
#         --logdir-prefix $LOGDIR_PREFIX \
#         --total-steps $TOTAL_STEPS \
#         --n-epochs $N_EPOCHS \
#         --seed $seed \
#     ) & # >& /dev/null &
# done
# endregion: Probing ppo_bc__rgb_spectro__pgwt__SAVi (No depth) @10M steps, Linear probe with minibatch train


## RNN agents get inputs of shape T * B = 150 * 10, but the input for the probes is broken into batches of size 30, probe depth 2
# region: Probing ppo_bc__rgb_spectro__gru__SAVi (No depth) @10M steps, Linear probe with minibatcdh train
# for seed in 111; do
#     TOTAL_STEPS=500000; N_EPOCHS=10;
#     (sleep 1s && python ppo_bc_probe_train_mb.py \
#         --exp-name "ppo_bc__rgb_spectro__gru__SAVi__9990001__n_mb_50__prb_dpth_2" \
#         --num-minibatches 50 \
#         --probe-depth 2 \
#         --pretrained-model-name "ppo_bc__rgb_spectro__gru__SAVi" \
#         --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc/ppo_bc__savi_ss1_rgb_spectro__gru_seed_222__2023_06_17_21_24_12_718867.musashi/models/ppo_agent.9990001.ckpt.pth" \
#         --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
#         --save-videos False \
#         --ent-coef 0 \
#         --wandb --wandb-project "ss-hab-bc-probing" --wandb-entity dosssman \
#         --logdir-prefix $LOGDIR_PREFIX \
#         --total-steps $TOTAL_STEPS \
#         --n-epochs $N_EPOCHS \
#         --seed $seed \
#     ) & # >& /dev/null &
# done
# endregion: Probing ppo_bc__rgb_spectro__gru__SAVi (No depth) @10M steps, Linear probe with minibatch train

# region: Probing ppo_bc__rgb_spectro__pgwt__SAVi (No depth) @10M steps, Linear probe with minibatch train
# for seed in 111; do
#     TOTAL_STEPS=500000; N_EPOCHS=10;
#     (sleep 1s && python ppo_bc_probe_train_mb.py \
#         --exp-name "ppo_bc__rgb_spectro__pgwt__SAVi__9990001__n_mb_50__prb_dpth_2" \
#         --num-minibatches 50 \
#         --probe-depth 2 \
#         --pretrained-model-name "ppo_bc__rgb_spectro__pgwt__SAVi" \
#         --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc/ppo_bc__savi_ss1_rgb__spectro__pgwt__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats_seed_222__2023_06_17_21_24_10_884437.musashi/models/ppo_agent.9990001.ckpt.pth" \
#         --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
#         --save-videos False \
#         --ent-coef 0 \
#         --wandb --wandb-project "ss-hab-bc-probing" --wandb-entity dosssman \
#         --logdir-prefix $LOGDIR_PREFIX \
#         --total-steps $TOTAL_STEPS \
#         --n-epochs $N_EPOCHS \
#         --seed $seed \
#     ) & # >& /dev/null &
# done
# endregion: Probing ppo_bc__rgb_spectro__pgwt__SAVi (No depth) @10M steps, Linear probe with minibatch train


## RNN agents get inputs of shape T * B = 150 * 10, but the input for the probes is broken into batches of size 30. probe depth 2, probe hid size 102, bias = True
# region: Probing ppo_bc__rgb_spectro__gru__SAVi (No depth) @10M steps, Linear probe with minibatcdh train
for seed in 111; do
    TOTAL_STEPS=500000; N_EPOCHS=10;
    (sleep 1s && python ppo_bc_probe_train_mb.py \
      --exp-name "ppo_bc__rgb_spectro__gru__SAVi__9990001__n_mb_50__prb_dpth_2_hsize_1024_bias" \
      --num-minibatches 50 \
      --probe-depth 2 \
      --probe-hid-size 1024 \
      --probe-bias True \
      --pretrained-model-name "ppo_bc__rgb_spectro__gru__SAVi" \
      --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc/ppo_bc__savi_ss1_rgb_spectro__gru_seed_222__2023_06_17_21_24_12_718867.musashi/models/ppo_agent.9990001.ckpt.pth" \
      --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      --save-videos False \
      --ent-coef 0 \
      --wandb --wandb-project "ss-hab-bc-probing" --wandb-entity dosssman \
      --logdir-prefix $LOGDIR_PREFIX \
      --total-steps $TOTAL_STEPS \
      --n-epochs $N_EPOCHS \
      --seed $seed \
    ) & # >& /dev/null &
done
# endregion: Probing ppo_bc__rgb_spectro__gru__SAVi (No depth) @10M steps, Linear probe with minibatch train

# region: Probing ppo_bc__rgb_spectro__pgwt__SAVi (No depth) @10M steps, Linear probe with minibatch train
for seed in 111; do
    TOTAL_STEPS=500000; N_EPOCHS=10;
    (sleep 1s && python ppo_bc_probe_train_mb.py \
      --exp-name "ppo_bc__rgb_spectro__pgwt__SAVi__9990001__n_mb_50__prb_dpth_2_hsize_1024_bias" \
      --num-minibatches 50 \
      --probe-depth 2 \
      --probe-hid-size 1024 \
      --probe-bias True \
      --pretrained-model-name "ppo_bc__rgb_spectro__pgwt__SAVi" \
      --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc/ppo_bc__savi_ss1_rgb__spectro__pgwt__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats_seed_222__2023_06_17_21_24_10_884437.musashi/models/ppo_agent.9990001.ckpt.pth" \
      --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
      --save-videos False \
      --ent-coef 0 \
      --wandb --wandb-project "ss-hab-bc-probing" --wandb-entity dosssman \
      --logdir-prefix $LOGDIR_PREFIX \
      --total-steps $TOTAL_STEPS \
      --n-epochs $N_EPOCHS \
      --seed $seed \
    ) & # >& /dev/null &
done
# endregion: Probing ppo_bc__rgb_spectro__pgwt__SAVi (No depth) @10M steps, Linear probe with minibatch train



## Old / deprecated

## RNN agents get inputs of shape T * B = 150 * 10, and batch of size 1500 is passed to the probe for learning / acc. computation.
# region: Probing ppo_bc__rgb_spectro__gru__SAVi (No depth) @10M steps, Linear probe
# for seed in 111; do
#     TOTAL_STEPS=500000; N_EPOCHS=10;
#     (sleep 1s && python ppo_bc_probe_train.py \
#         --exp-name "ppo_bc__rgb_spectro__gru__SAVi__9990001" \
#         --pretrained-model-name "ppo_bc__rgb_spectro__gru__SAVi" \
#         --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc/ppo_bc__savi_ss1_rgb_spectro__gru_seed_222__2023_06_17_21_24_12_718867.musashi/models/ppo_agent.9990001.ckpt.pth" \
#         --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
#         --save-videos False \
#         --ent-coef 0 \
#         --wandb --wandb-project "ss-hab-bc-probing" --wandb-entity dosssman \
#         --logdir-prefix $LOGDIR_PREFIX \
#         --total-steps $TOTAL_STEPS \
#         --n-epochs $N_EPOCHS \
#         --seed $seed \
#     ) & # >& /dev/null &
# done
# endregion: Probing ppo_bc__rgb_spectro__gru__SAVi (No depth) @10M steps, Linear probe

# region: Probing ppo_bc__rgb_spectro__pgwt__SAVi (No depth) @10M steps, Linear probe
# for seed in 111; do
#     TOTAL_STEPS=500000; N_EPOCHS=10;
#     (sleep 1s && python ppo_bc_probe_train.py \
#         --exp-name "ppo_bc__rgb_spectro__pgwt__SAVi__9990001" \
#         --pretrained-model-name "ppo_bc__rgb_spectro__pgwt__SAVi" \
#         --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc/ppo_bc__savi_ss1_rgb__spectro__pgwt__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats_seed_222__2023_06_17_21_24_10_884437.musashi/models/ppo_agent.9990001.ckpt.pth" \
#         --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
#         --save-videos False \
#         --ent-coef 0 \
#         --wandb --wandb-project "ss-hab-bc-probing" --wandb-entity dosssman \
#         --logdir-prefix $LOGDIR_PREFIX \
#         --total-steps $TOTAL_STEPS \
#         --n-epochs $N_EPOCHS \
#         --seed $seed \
#     ) & # >& /dev/null &
# done
# endregion: Probing ppo_bc__rgb_spectro__pgwt__SAVi (No depth) @10M steps, Linear probe


## THe follwoign two runs are based on RGBD runs, with depth modality that is not necessarily relevant to this task
## Futhermore, might want to use depth as a separated modality that is not tied.
# region: Probing ppo_gru__random
# for seed in 111; do
#     TOTAL_STEPS=500000; N_EPOCHS=10;
#     (sleep 1s && python ppo_bc_probe_train.py \
#         --exp-name "ppo_gru__random__fixTB" \
#         --pretrained-model-name "ppo_gru__random" \
#         --config-path "env_configs/savi/savi_ss1_rgbd_spectro.yaml" \
#         --save-videos False \
#         --ent-coef 0 \
#         --wandb --wandb-project "ss-hab-bc-probing" --wandb-entity dosssman \
#         --logdir-prefix $LOGDIR_PREFIX \
#         --total-steps $TOTAL_STEPS \
#         --n-epochs $N_EPOCHS \
#         --seed $seed \
#     ) & # >& /dev/null &
# done
# endregion: Probing ppo_gru__random

# region: Probing ppo_pgwt__random
# for seed in 111; do
#     TOTAL_STEPS=500000; N_EPOCHS=10;
#     (sleep 1s && python ppo_bc_probe_train.py \
#         --exp-name "ppo_pgwt__random__fixTB" \
#         --pretrained-model-name "ppo_pgwt__random" \
#         --config-path "env_configs/savi/savi_ss1_rgbd_spectro.yaml" \
#         --save-videos False \
#         --ent-coef 0 \
#         --wandb --wandb-project "ss-hab-bc-probing" --wandb-entity dosssman \
#         --logdir-prefix $LOGDIR_PREFIX \
#         --total-steps $TOTAL_STEPS \
#         --n-epochs $N_EPOCHS \
#         --seed $seed \
#     ) & # >& /dev/null &
# done
# endregion: Probing ppo_pgwt__random


## Following runs are based on RGBD runs, etc...
# region: Probing ppo_bc__rgbd_spectro__gru__SAVi
# for seed in 111; do
#     TOTAL_STEPS=500000; N_EPOCHS=10;
#     (sleep 1s && python ppo_bc_probe_train.py \
#         --exp-name "ppo_bc__rgbd_spectro__gru__SAVi__fixTB" \
#         --pretrained-model-name "ppo_bc__rgbd_spectro__gru__SAVi" \
#         --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc/ppo_bc__savi_ss1_rgbd_spectro__gru_seed_111__2023_06_10_16_05_39_999286.musashi/models/ppo_agent.4995001.ckpt.pth" \
#         --config-path "env_configs/savi/savi_ss1_rgbd_spectro.yaml" \
#         --save-videos False \
#         --ent-coef 0 \
#         --wandb --wandb-project "ss-hab-bc-probing" --wandb-entity dosssman \
#         --logdir-prefix $LOGDIR_PREFIX \
#         --total-steps $TOTAL_STEPS \
#         --n-epochs $N_EPOCHS \
#         --seed $seed \
#     ) & # >& /dev/null &
# done
# endregion: Probing ppo_bc__rgbd_spectro__gru__SAVi

# region: Probing ppo_bc__rgbd_spectro__pgwt__SAVi
# for seed in 111; do
#     TOTAL_STEPS=500000; N_EPOCHS=10;
#     (sleep 1s && python ppo_bc_probe_train.py \
#         --exp-name "ppo_bc__rgbd_spectro__pgwt__SAVi__fixTB" \
#         --pretrained-model-name "ppo_bc__rgbd_spectro__pgwt__SAVi" \
#         --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc/ppo_bc__savi_ss1_rgbd__spectro__pgwt__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats_seed_111__2023_06_10_16_05_37_098602.musashi/models/ppo_agent.4995001.ckpt.pth" \
#         --config-path "env_configs/savi/savi_ss1_rgbd_spectro.yaml" \
#         --save-videos False \
#         --ent-coef 0 \
#         --wandb --wandb-project "ss-hab-bc-probing" --wandb-entity dosssman \
#         --logdir-prefix $LOGDIR_PREFIX \
#         --total-steps $TOTAL_STEPS \
#         --n-epochs $N_EPOCHS \
#         --seed $seed \
#     ) & # >& /dev/null &
# done
# endregion: Probing ppo_bc__rgbd_spectro__pgwt__SAVi


## 
# region: Probing ppo_bc__rgb_spectro__gru__SAVi (No depth), Linear probe
# for seed in 111; do
#     TOTAL_STEPS=500000; N_EPOCHS=10;
#     (sleep 1s && python ppo_bc_probe_train.py \
#         --exp-name "ppo_bc__rgb_spectro__gru__SAVi__4995001" \
#         --pretrained-model-name "ppo_bc__rgb_spectro__gru__SAVi" \
#         --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc/ppo_bc__savi_ss1_rgb_spectro__gru_seed_222__2023_06_17_21_24_12_718867.musashi/models/ppo_agent.4995001.ckpt.pth" \
#         --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
#         --save-videos False \
#         --ent-coef 0 \
#         --wandb --wandb-project "ss-hab-bc-probing" --wandb-entity dosssman \
#         --logdir-prefix $LOGDIR_PREFIX \
#         --total-steps $TOTAL_STEPS \
#         --n-epochs $N_EPOCHS \
#         --seed $seed \
#     ) & # >& /dev/null &
# done
# endregion: Probing ppo_bc__rgb_spectro__gru__SAVi (No depth), Linear probe

# region: Probing ppo_bc__rgb_spectro__pgwt__SAVi (No depth), Linear probe
# for seed in 111; do
#     TOTAL_STEPS=500000; N_EPOCHS=10;
#     (sleep 1s && python ppo_bc_probe_train.py \
#         --exp-name "ppo_bc__rgb_spectro__pgwt__SAVi__4995001" \
#         --pretrained-model-name "ppo_bc__rgb_spectro__pgwt__SAVi" \
#         --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc/ppo_bc__savi_ss1_rgb__spectro__pgwt__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats_seed_222__2023_06_17_21_24_10_884437.musashi/models/ppo_agent.4995001.ckpt.pth" \
#         --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
#         --save-videos False \
#         --ent-coef 0 \
#         --wandb --wandb-project "ss-hab-bc-probing" --wandb-entity dosssman \
#         --logdir-prefix $LOGDIR_PREFIX \
#         --total-steps $TOTAL_STEPS \
#         --n-epochs $N_EPOCHS \
#         --seed $seed \
#     ) & # >& /dev/null &
# done
# endregion: Probing ppo_bc__rgb_spectro__pgwt__SAVi (No depth), Linear probe


## 
# region: Probing ppo_bc__rgb_spectro__gru__SAVi (No depth) @10M steps, beefier linear probe
# for seed in 111; do
#     TOTAL_STEPS=500000; N_EPOCHS=10;
#     (sleep 1s && python ppo_bc_probe_train.py \
#         --exp-name "ppo_bc__rgb_spectro__gru__SAVi__9990001__prb_depth_2" \
#         --probe-depth 2 \
#         --pretrained-model-name "ppo_bc__rgb_spectro__gru__SAVi" \
#         --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc/ppo_bc__savi_ss1_rgb_spectro__gru_seed_222__2023_06_17_21_24_12_718867.musashi/models/ppo_agent.9990001.ckpt.pth" \
#         --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
#         --save-videos False \
#         --ent-coef 0 \
#         --wandb --wandb-project "ss-hab-bc-probing" --wandb-entity dosssman \
#         --logdir-prefix $LOGDIR_PREFIX \
#         --total-steps $TOTAL_STEPS \
#         --n-epochs $N_EPOCHS \
#         --seed $seed \
#     ) & # >& /dev/null &
# done
# endregion: Probing ppo_bc__rgb_spectro__gru__SAVi (No depth) @10M steps, beefier linear probe

# region: Probing ppo_bc__rgb_spectro__pgwt__SAVi (No depth) @10M steps, beefier linear probe
# for seed in 111; do
#     TOTAL_STEPS=500000; N_EPOCHS=10;
#     (sleep 1s && python ppo_bc_probe_train.py \
#         --exp-name "ppo_bc__rgb_spectro__pgwt__SAVi__9990001__prb_depth_2" \
#         --probe-depth 2 \
#         --pretrained-model-name "ppo_bc__rgb_spectro__pgwt__SAVi" \
#         --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc/ppo_bc__savi_ss1_rgb__spectro__pgwt__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats_seed_222__2023_06_17_21_24_10_884437.musashi/models/ppo_agent.9990001.ckpt.pth" \
#         --config-path "env_configs/savi/savi_ss1_rgb_spectro.yaml" \
#         --save-videos False \
#         --ent-coef 0 \
#         --wandb --wandb-project "ss-hab-bc-probing" --wandb-entity dosssman \
#         --logdir-prefix $LOGDIR_PREFIX \
#         --total-steps $TOTAL_STEPS \
#         --n-epochs $N_EPOCHS \
#         --seed $seed \
#     ) & # >& /dev/null &
# done
# endregion: Probing ppo_bc__rgb_spectro__pgwt__SAVi (No depth) @10M steps, beefier linear probe


## 
# region: Probing ppo_bc__rgbd_spectro__gru__SAVi, affine probe, depth = 1
# for seed in 111; do
#     TOTAL_STEPS=500000; N_EPOCHS=10;
#     (sleep 1s && python ppo_bc_probe_train.py \
#         --exp-name "ppo_bc__rgbd_spectro__gru__SAVi__prb_bias" \
#         --pretrained-model-name "ppo_bc__rgbd_spectro__gru__SAVi" \
#         --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc/ppo_bc__savi_ss1_rgbd_spectro__gru_seed_111__2023_06_10_16_05_39_999286.musashi/models/ppo_agent.4995001.ckpt.pth" \
#         --config-path "env_configs/savi/savi_ss1_rgbd_spectro.yaml" \
#         --probe-bias True \
#         --save-videos False \
#         --ent-coef 0 \
#         --wandb --wandb-project "ss-hab-bc-probing" --wandb-entity dosssman \
#         --logdir-prefix $LOGDIR_PREFIX \
#         --total-steps $TOTAL_STEPS \
#         --n-epochs $N_EPOCHS \
#         --seed $seed \
#     ) & # >& /dev/null &
# done
# endregion: Probing ppo_bc__rgbd_spectro__gru__SAVi, affine probe, depth = 1

# region: Probing ppo_bc__rgbd_spectro__pgwt__SAVi, affine probe, depth = 1
# for seed in 111; do
#     TOTAL_STEPS=500000; N_EPOCHS=10;
#     (sleep 1s && python ppo_bc_probe_train.py \
#         --exp-name "ppo_bc__rgbd_spectro__pgwt__SAVi__prb_bias" \
#         --pretrained-model-name "ppo_bc__rgbd_spectro__pgwt__SAVi" \
#         --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc/ppo_bc__savi_ss1_rgbd__spectro__pgwt__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats_seed_111__2023_06_10_16_05_37_098602.musashi/models/ppo_agent.4995001.ckpt.pth" \
#         --config-path "env_configs/savi/savi_ss1_rgbd_spectro.yaml" \
#         --probe-bias True \
#         --save-videos False \
#         --ent-coef 0 \
#         --wandb --wandb-project "ss-hab-bc-probing" --wandb-entity dosssman \
#         --logdir-prefix $LOGDIR_PREFIX \
#         --total-steps $TOTAL_STEPS \
#         --n-epochs $N_EPOCHS \
#         --seed $seed \
#     ) & # >& /dev/null &
# done
# endregion: Probing ppo_bc__rgbd_spectro__pgwt__SAVi, affine probe, depth = 1


## 
# region: Probing ppo_bc__rgbd_spectro__gru__SAVi, beefier probes
# for seed in 111; do
#     TOTAL_STEPS=500000; N_EPOCHS=10;
#     (sleep 1s && python ppo_bc_probe_train.py \
#         --exp-name "ppo_bc__rgbd_spectro__gru__SAVi__prb_depth_2" \
#         --probe-depth 2 \
#         --pretrained-model-name "ppo_bc__rgbd_spectro__gru__SAVi" \
#         --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc/ppo_bc__savi_ss1_rgbd_spectro__gru_seed_111__2023_06_10_16_05_39_999286.musashi/models/ppo_agent.4995001.ckpt.pth" \
#         --config-path "env_configs/savi/savi_ss1_rgbd_spectro.yaml" \
#         --save-videos False \
#         --ent-coef 0 \
#         --wandb --wandb-project "ss-hab-bc-probing" --wandb-entity dosssman \
#         --logdir-prefix $LOGDIR_PREFIX \
#         --total-steps $TOTAL_STEPS \
#         --n-epochs $N_EPOCHS \
#         --seed $seed \
#     ) & # >& /dev/null &
# done
# endregion: Probing ppo_bc__rgbd_spectro__gru__SAVi, beefier probes

# region: Probing ppo_bc__rgbd_spectro__pgwt__SAVi, beefier probes
# for seed in 111; do
#     TOTAL_STEPS=500000; N_EPOCHS=10;
#     (sleep 1s && python ppo_bc_probe_train.py \
#         --exp-name "ppo_bc__rgbd_spectro__pgwt__SAVi__prb_depth_2" \
#         --probe-depth 2 \
#         --pretrained-model-name "ppo_bc__rgbd_spectro__pgwt__SAVi" \
#         --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc/ppo_bc__savi_ss1_rgbd__spectro__pgwt__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats_seed_111__2023_06_10_16_05_37_098602.musashi/models/ppo_agent.4995001.ckpt.pth" \
#         --config-path "env_configs/savi/savi_ss1_rgbd_spectro.yaml" \
#         --save-videos False \
#         --ent-coef 0 \
#         --wandb --wandb-project "ss-hab-bc-probing" --wandb-entity dosssman \
#         --logdir-prefix $LOGDIR_PREFIX \
#         --total-steps $TOTAL_STEPS \
#         --n-epochs $N_EPOCHS \
#         --seed $seed \
#     ) & # >& /dev/null &
# done
# endregion: Probing ppo_bc__rgbd_spectro__pgwt__SAVi, beefier probes

# Clean up env vars
# export CUDA_VISIBLE_DEVICES=
export LOGDIR_PREFIX=
export MASTER_PORT=
export WANDB_DIR=