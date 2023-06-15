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


# region: Probing ppo_gru__random
for seed in 111; do
    TOTAL_STEPS=500000; N_EPOCHS=10;
    (sleep 1s && python ppo_bc_probe_train.py \
        --exp-name "ppo_gru__random" \
        --pretrained-model-name "ppo_gru__random" \
        --config-path "env_configs/savi/savi_ss1_rgbd_spectro.yaml" \
        --save-videos False \
        --ent-coef 0 \
        --wandb --wandb-project "ss-hab-bc-probing" --wandb-entity dosssman \
        --logdir-prefix $LOGDIR_PREFIX \
        --total-steps $TOTAL_STEPS \
        --n-epochs $N_EPOCHS \
        --seed $seed \
    ) & # >& /dev/null &
done
# endregion: Probing ppo_gru__random

# region: Probing ppo_pgwt__random
for seed in 111; do
    TOTAL_STEPS=500000; N_EPOCHS=10;
    (sleep 1s && python ppo_bc_probe_train.py \
        --exp-name "ppo_pgwt__random" \
        --pretrained-model-name "ppo_pgwt__random" \
        --config-path "env_configs/savi/savi_ss1_rgbd_spectro.yaml" \
        --save-videos False \
        --ent-coef 0 \
        --wandb --wandb-project "ss-hab-bc-probing" --wandb-entity dosssman \
        --logdir-prefix $LOGDIR_PREFIX \
        --total-steps $TOTAL_STEPS \
        --n-epochs $N_EPOCHS \
        --seed $seed \
    ) & # >& /dev/null &
done
# endregion: Probing ppo_pgwt__random

# region: Probing ppo_bc__rgbd_spectro__gru__SAVi
for seed in 111; do
    TOTAL_STEPS=500000; N_EPOCHS=10;
    (sleep 1s && python ppo_bc_probe_train.py \
        --exp-name "ppo_bc__rgbd_spectro__gru__SAVi" \
        --pretrained-model-name "ppo_bc__rgbd_spectro__gru__SAVi" \
        --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc/ppo_bc__savi_ss1_rgbd_spectro__gru_seed_111__2023_06_10_16_05_39_999286.musashi/models/ppo_agent.4995001.ckpt.pth" \
        --config-path "env_configs/savi/savi_ss1_rgbd_spectro.yaml" \
        --save-videos False \
        --ent-coef 0 \
        --wandb --wandb-project "ss-hab-bc-probing" --wandb-entity dosssman \
        --logdir-prefix $LOGDIR_PREFIX \
        --total-steps $TOTAL_STEPS \
        --n-epochs $N_EPOCHS \
        --seed $seed \
    ) & # >& /dev/null &
done
# endregion: Probing ppo_bc__rgbd_spectro__gru__SAVi

# region: Probing ppo_bc__rgbd_spectro__pgwt__SAVi
for seed in 111; do
    TOTAL_STEPS=500000; N_EPOCHS=10;
    (sleep 1s && python ppo_bc_probe_train.py \
        --exp-name "ppo_bc__rgbd_spectro__pgwt__SAVi" \
        --pretrained-model-name "ppo_bc__rgbd_spectro__pgwt__SAVi" \
        --pretrained-model-path "/home/rousslan/random/rl/exp-logs/ss-hab-bc/ppo_bc__savi_ss1_rgbd__spectro__pgwt__dpth_1_nlats_8_latdim_64_noSA_CAnheads_1_SAnheads_4_modembed_0_CAprevlats_seed_111__2023_06_10_16_05_37_098602.musashi/models/ppo_agent.4995001.ckpt.pth" \
        --config-path "env_configs/savi/savi_ss1_rgbd_spectro.yaml" \
        --save-videos False \
        --ent-coef 0 \
        --wandb --wandb-project "ss-hab-bc-probing" --wandb-entity dosssman \
        --logdir-prefix $LOGDIR_PREFIX \
        --total-steps $TOTAL_STEPS \
        --n-epochs $N_EPOCHS \
        --seed $seed \
    ) & # >& /dev/null &
done
# endregion: Probing ppo_bc__rgbd_spectro__pgwt__SAVi

# Clean up env vars
# export CUDA_VISIBLE_DEVICES=
export LOGDIR_PREFIX=
export MASTER_PORT=
export WANDB_DIR=