export WANDB_PROJECT="ss-hab-bc-revised-sweep"

export LOGDIR_PREFIX=~/random/rl/exp-logs/$WANDB_PROJECT
if [ ! -d $LOGDIR_PREFIX ]; then
  mkdir -p $LOGDIR_PREFIX
fi

export WANDB_DIR="$LOGDIR_PREFIX""_wandb"
if [ ! -d $WANDB_DIR ]; then
  mkdir -p $WANDB_DIR
fi

## Running agents
# Init sweep
# wandb sweep --project ss-hab-sweep-test ppo_bc_wandb_sweep.yml

# 2024-01-15: Bayes sweep 5 seeds, 4 hid size: 
## wandb agent dosssman/ss-hab-sweep-test/2hma5226

# Execute sweep runs
# NOTE: set a specific WANDB_DIR folder for the logging
# Even better, just set is as env variable before starting the run.
# WANDB_DIR="/home/rousslan/random/rl/exp-logs/ss-hab-sweep-test-2_wandb" wandb agent dosssman/ss-hab-sweep-test/2hma5226

# Clean up env vars
# export CUDA_VISIBLE_DEVICES=
# export LOGDIR_PREFIX=
# export WANDB_DIR=