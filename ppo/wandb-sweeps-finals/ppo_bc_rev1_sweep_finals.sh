exit 1 # This script is not supposed to be ran, but serves as a memo of the sweep runs.

conda activate ss-hab-headless-py39

export WANDB_PROJECT="ss-hab-bc-revised-finals"

export LOGDIR_PREFIX=~/random/rl/exp-logs/$WANDB_PROJECT
if [ ! -d $LOGDIR_PREFIX ]; then
  mkdir -p $LOGDIR_PREFIX
fi

export WANDB_DIR="$LOGDIR_PREFIX""_wandb"
if [ ! -d $WANDB_DIR ]; then
  mkdir -p $WANDB_DIR
fi

echo $WANDB_DIR
echo $LOGDIR_PREFIX

# 2024-02-13 Init sweeps for GW agent
wandb sweep --project "ss-hab-bc-revised-finals" ppo_bc_rev1_final__gw_32.yml
# Sweep cmd:
wandb agent dosssman/ss-hab-bc-revised-finals/8xi8qzb1

wandb sweep --project "ss-hab-bc-revised-finals" ppo_bc_rev1_final__gw_64.yml
wandb sweep --project "ss-hab-bc-revised-finals" ppo_bc_rev1_final__gw_64__makeup.yml
# Sweep cmd:
wandb agent dosssman/ss-hab-bc-revised-finals/1hh364va
wandb agent dosssman/ss-hab-bc-revised-finals/4rol21qd # Makeup

wandb sweep --project "ss-hab-bc-revised-finals" ppo_bc_rev1_final__gw_128.yml
# Sweep cmd:
wandb agent dosssman/ss-hab-bc-revised-finals/kat63jyo

wandb sweep --project "ss-hab-bc-revised-finals" ppo_bc_rev1_final__gw_256.yml
# Sweep cmd:
wandb agent dosssman/ss-hab-bc-revised-finals/x1h6o421

wandb sweep --project "ss-hab-bc-revised-finals" ppo_bc_rev1_final__gw_512.yml
# Sweep cmd:
wandb agent dosssman/ss-hab-bc-revised-finals/ubpu7vgx


# 2024-02-13 Init sweeps for GRU agent
wandb sweep --project "ss-hab-bc-revised-finals" ppo_bc_rev1_final__gru_32.yml
wandb sweep --project "ss-hab-bc-revised-finals" ppo_bc_rev1_final__gru_32__makeup.yml
# Sweep cmd:
wandb agent dosssman/ss-hab-bc-revised-finals/hfzyln4n
wandb agent dosssman/ss-hab-bc-revised-finals/2hrqrv6m # Makeup

wandb sweep --project "ss-hab-bc-revised-finals" ppo_bc_rev1_final__gru_64.yml
wandb sweep --project "ss-hab-bc-revised-finals" ppo_bc_rev1_final__gru_64__makeup.yml
# Sweep cmd:
wandb agent dosssman/ss-hab-bc-revised-finals/4u34tx79
wandb agent dosssman/ss-hab-bc-revised-finals/hcd4yr3k # Makeup

wandb sweep --project "ss-hab-bc-revised-finals" ppo_bc_rev1_final__gru_128.yml
wandb sweep --project "ss-hab-bc-revised-finals" ppo_bc_rev1_final__gru_128__makeup.yml
# Sweep cmd:
wandb agent dosssman/ss-hab-bc-revised-finals/qixvi9fi
wandb agent dosssman/ss-hab-bc-revised-finals/ex3p8nt0 # Makeup

wandb sweep --project "ss-hab-bc-revised-finals" ppo_bc_rev1_final__gru_256.yml
# Sweep cmd:
wandb agent dosssman/ss-hab-bc-revised-finals/gd8fn58f

wandb sweep --project "ss-hab-bc-revised-finals" ppo_bc_rev1_final__gru_512.yml
# Sweep cmd:
wandb agent dosssman/ss-hab-bc-revised-finals/3ovbyg5t


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
export LOGDIR_PREFIX=
export WANDB_DIR=