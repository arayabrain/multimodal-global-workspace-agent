# Automatic GPU selection if needs be
import re
import nvsmi
import torch as th

# Logging helpers
# Helper to faciliatate periodic action
class Every:

  def __init__(self, every):
    self._every = every
    self._last = None

  def __call__(self, step):
    if not self._every:
      return False
    if self._last is None:
      self._last = step
      return True
    if step >= self._last + self._every:
      self._last += self._every
      return True
    return False

# GPU selection helpers
def get_device(config):
	use_gpu = config.gpu_device

	if use_gpu == "auto":
		gpu_dev = get_avail_gpu(
			gpu_auto_rev=config.gpu_auto_rev,
			gpu_auto_max_n_procs=config.gpu_auto_max_n_procs)
		device = th.device(f"cuda:{gpu_dev}"
			if th.cuda.is_available() and not config.cpu else "cpu")
	elif use_gpu == "":
		device = th.device("cuda"
			if th.cuda.is_available() and not config.cpu else "cpu")
	else:
		try:
			device = th.device(f"cuda:{int(use_gpu)}"
				if th.cuda.is_available() and not config.cpu else "cpu")
		except ValueError:
			gpu_dev = int(re.match(r"\d+", use_gpu))
			device = th.device(f"cuda:{int(gpu_dev)}"
				if th.cuda.is_available() and not config.cpu else "cpu")
	
	return device

def get_avail_gpu(gpu_auto_rev=False, gpu_auto_max_n_procs=2):
	# TODO: paremterize the GPU selection criterions ?
	available_gpus = list(nvsmi.get_available_gpus(gpu_util_max=51, mem_util_max=55, mem_free_min=9000))
	
	if not len(available_gpus):
		raise Exception(f"No available GPU to be automatically assigned, aborting experiment.")
	
	def count_procs_per_gpu(gpu_id):
		n_procs = 0
		for gpu_proc in nvsmi.get_gpu_processes():
			if gpu_id == gpu_proc.gpu_id:
				n_procs += 1
		return n_procs
	
	avail_gpus = [] # Additional filter based on process count on the GPU
	# If a GPU has more than 2 process, likely to be inconvenient to HWM exp.
	MAX_PROCS_PER_GPU = min(gpu_auto_max_n_procs, 8) # Hard limit of 8 for now

	for gpu in available_gpus:
		if count_procs_per_gpu(gpu.id) < MAX_PROCS_PER_GPU:
			avail_gpus.append(gpu)

	if gpu_auto_rev:
		avail_gpus = avail_gpus[::-1]

	return int(avail_gpus[0].id)