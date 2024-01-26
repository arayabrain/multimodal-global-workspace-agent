exit 1 # Not supposed to be ran

conda activate ss-hab-headless-py39

export WANDB_PROJECT="ss-hab-bc-revised-sweep"

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

# 2024-01-19
## Init sweeps for GW agent
wandb sweep --project "ss-hab-bc-revised-sweep" ppo_bc_rev1_sweep__gw_32.yml
# Sweep cmd: 
wandb agent dosssman/ss-hab-bc-revised-sweep/7d73e9pg --count 10

wandb sweep --project "ss-hab-bc-revised-sweep" ppo_bc_rev1_sweep__gw_64.yml
# Sweep cmd:
wandb agent dosssman/ss-hab-bc-revised-sweep/jk68e46c --count 10

wandb sweep --project "ss-hab-bc-revised-sweep" ppo_bc_rev1_sweep__gw_128.yml
# Sweep cmd:
wandb agent dosssman/ss-hab-bc-revised-sweep/altdwxen --count 10

wandb sweep --project "ss-hab-bc-revised-sweep" ppo_bc_rev1_sweep__gw_256.yml
# Sweep cmd:
wandb agent dosssman/ss-hab-bc-revised-sweep/874isgkh --count 10

wandb sweep --project "ss-hab-bc-revised-sweep" ppo_bc_rev1_sweep__gw_512.yml
# Sweep cmd:
wandb agent dosssman/ss-hab-bc-revised-sweep/s5eec2x4 --count 5


# 2024-01-26
## Init sweeps for GRU
wandb sweep --project "ss-hab-bc-revised-sweep" ppo_bc_rev1_sweep__gru_32.yml
# Sweep cmd: 
wandb agent dosssman/ss-hab-bc-revised-sweep/06vgm81a --count 10

wandb sweep --project "ss-hab-bc-revised-sweep" ppo_bc_rev1_sweep__gru_64.yml
# Sweep cmd: 
wandb agent dosssman/ss-hab-bc-revised-sweep/36x1e4mn --count 10

wandb sweep --project "ss-hab-bc-revised-sweep" ppo_bc_rev1_sweep__gru_128.yml
# Sweep cmd:
wandb agent dosssman/ss-hab-bc-revised-sweep/4xcj7wvk --count 5 # run twice over diff. workstations

wandb sweep --project "ss-hab-bc-revised-sweep" ppo_bc_rev1_sweep__gru_256.yml
# Sweep cmd:
wandb agent dosssman/ss-hab-bc-revised-sweep/p6do3c7l --count 10

wandb sweep --project "ss-hab-bc-revised-sweep" ppo_bc_rev1_sweep__gru_512.yml
# Sweep cmd:  
wandb agent dosssman/ss-hab-bc-revised-sweep/lyt9qfa1 --count 10

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