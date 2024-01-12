export LOGDIR_PREFIX=~/random/rl/exp-logs/ss-hab-sweep-test
if [ ! -d $LOGDIR_PREFIX ]; then
  mkdir -p $LOGDIR_PREFIX
fi

export WANDB_DIR="$LOGDIR_PREFIX""_wandb"
if [ ! -d $WANDB_DIR ]; then
  mkdir -p $WANDB_DIR
fi

# Init sweep
# wandb sweep --project ss-hab-sweep-test ppo_bc_wandb_sweep.yml

# Execute sweep runs
# NOTE: set a specific WANDB_DIR folder for the logging
# WANDB_DIR="/home/rousslan/random/rl/exp-logs/ss-hab-sweep-test-2_wandb" wandb agent dosssman/ss-hab-sweep-test/gwzoasps

# Clean up env vars
# export CUDA_VISIBLE_DEVICES=
export LOGDIR_PREFIX=
export WANDB_DIR=