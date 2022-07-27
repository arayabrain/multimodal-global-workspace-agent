# A wrapper around both the TensorboardX logger tool and the Wandb experiment manager logger
# also handlese creation of logging folders for convenience
import os
import git
import sys
import json
import time
import wandb
import shutil
import torch as th

# Pytorch's TensorboardX Support
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from datetime import datetime

class TBXLogger(object):
    def __init__( self, logdir = None, logdir_prefix = None, exp_name = 'default',
        graph = None, args=None):

        self.logdir_prefix = args.logdir_prefix
        self.logdir = logdir
        # TODO: args is affect in this class, but the parent class also depends on it
        # This is bad.
        self.args = args

        # Path for resume ckeckpoint dict path
        self.wandb_id = ""

        if args.resume != "":
            # NOTE
            ## 1. 'resume' should hold the stopped experiment log dir path relative to the logdir_prefix
            ## 2. 'logdir_prefix' should be the same as the original experiment by default.
            ## Other use case don't really make sense any way.

            self.full_exp_name = args.resume
            self.logdir_prefix = args.logdir_prefix
            self.logdir = os.path.join(self.logdir_prefix, self.full_exp_name)
            self.resume_ckpt_path = os.path.join(self.logdir, "resume_ckpt.json")

            if not os.path.exists(self.logdir) and not os.path.isdir(self.logdir):
                raise Exception(f"Attempted 'resume': {self.full_exp_name} not found.")
            
            # TODO: read resume wandb ID
            self.wandb_id = self.get_resume_ckpt()["wandb_id"]

            # Overwrite all the hyper parameters based on what is in the resumed run folder
            resume_args = self.get_resume_args()
            for k, v in resume_args.items():
                # Params related to resume and GPU should not be overriden, in case
                # the hardware setting for resume requires different GPU config.
                if k not in ["resume", "gpu_device", "gpu_auto_max_n_procs", "gpu_auto_rev",]:
                    setattr(args, k ,v)
                
            if args.wandb:
                monitor_gym = True if args.save_videos else False
                wandb.init(id=self.wandb_id, project=args.wandb_project,
                        config=vars(args), name=self.full_exp_name,
                        monitor_gym=monitor_gym, sync_tensorboard=True, save_code=True, resume=True)
            
            self.tb_writer = SummaryWriter(self.logdir)

            # TODO: reload all the hyper parameters just to be sure
        else:
            if self.logdir_prefix is None:
                self.logdir_prefix = os.getcwd()
                self.logdir_prefix = os.path.join( self.logdir_prefix, 'logs')

                if not os.path.exists( self.logdir_prefix):
                    os.makedirs( self.logdir_prefix)
            
            if self.logdir is None:
                strfiedDatetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
                self.logdir = exp_name

                # TODO: Dynamically generate the log name based on what was passed to the args.
                if hasattr( args, "env_id"):
                    self.logdir +=  "_%s_seed_%d__%s.%s" % ( args.env_id, args.seed, strfiedDatetime, os.uname()[1])
                else:
                    # This is for support of non RL experiments
                    self.logdir +=  "_seed_%d__%s.%s" % ( args.seed, strfiedDatetime, os.uname()[1])

                self.full_exp_name = self.logdir

                self.logdir = os.path.join( self.logdir_prefix, self.logdir)

            if not os.path.exists( self.logdir):
                os.makedirs( self.logdir)

            self.resume_ckpt_path = os.path.join(self.logdir, "resume_ckpt.json")

            # Enable Wandb if needed.
            # Call wandb.init beforethe tensorboard writer is created
            # https://docs.wandb.ai/guides/integrations/tensorboard#how-do-i-configure-tensorboard-when-im-using-it-with-wandb
            if args is not None and args.wandb:
                monitor_gym = True if args.save_videos else False
                self.wandb_id = wandb.util.generate_id()

                wandb.init(id=self.wandb_id, project=args.wandb_project,
                    config=vars(args), name=self.full_exp_name,
                    monitor_gym=monitor_gym, sync_tensorboard=True, save_code=True, resume="allow")
            # Call after wandb.init()
            self.tb_writer = SummaryWriter(self.logdir)

            if args is not None:
                hyparams ="|Parameter|Value|\n|-|-|\n%s" \
                    % ('\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()]))

                self.tb_writer.add_text( "Hyparams", hyparams, 0 )

                # Also dumping hyparams to JSON file
                with open( os.path.join( self.logdir, "hyparams.json"), "w") as fp:
                    json.dump( vars( args), fp)
            

        # Folder for weights saving
        self.models_save_dir = os.path.join( self.logdir, "models")

        if not os.path.exists( self.models_save_dir):
            os.makedirs( self.models_save_dir)

        # Creates folder for videos saving
        self.videos_save_dir = os.path.join( self.logdir, "videos")

        if not os.path.exists( self.videos_save_dir):
            os.makedirs( self.videos_save_dir)

        # Folders for images and plots saving
        self.images_save_dir = os.path.join( self.logdir, "images")

        if not os.path.exists( self.images_save_dir):
            os.makedirs( self.images_save_dir)

        # Tracking some training stats
        self.tracked_stats = {}

    def get_logdir(self):
        return self.logdir

    # Just access it instead ?
    def get_models_savedir(self):
        return self.models_save_dir

    def get_videos_savedir(self):
        return self.videos_save_dir

    def get_images_savedir(self):
        return self.images_save_dir

    # A helper to track training states such as FPS, 
    # number of updates per step, etc...
    def track_duration(self, name, step, inverse=False):
        if name not in list(self.tracked_stats.keys()):
            self.tracked_stats[name] = {
                "last_step": step,
                "last_time": time.time(),
            }
            return 0
        last_step, last_time = self.tracked_stats[name]["last_step"], \
            self.tracked_stats[name]["last_time"]
        elapsed_steps = step - last_step
        duration = time.time() - last_time

        self.tracked_stats[name]["last_time"] += duration
        self.tracked_stats[name]["last_step"] = step

        if inverse:
            if elapsed_steps == 0:
                return 0
            else:
                return duration / elapsed_steps

        return elapsed_steps / duration

    # Simple scalar logging
    def log_stats( self, stats_data, step, prefix=None):
        ''' Expects a dictionary with key names and values'''

        for tag, val in stats_data.items():
            fulltag = prefix + "/" + tag if prefix is not None else tag
            self.tb_writer.add_scalar(fulltag, val, global_step=step)

    # Simple histogram logging
    def log_histogram(self, name, hist_data, step, prefix=None):
        fieldname = name if prefix == None else prefix + "/" + name

        self.tb_writer.add_histogram( fieldname, hist_data, step)
    
    # Log resume.json for resume support
    def get_git_hash(self):
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha

        return sha
    
    def get_git_branch(self):
        repo = git.Repo(search_parent_directories=True)

        return repo.active_branch

    def log_resume_ckpt(self, resume_dict):
        resume_ckpt_path_tmp = ''.join([
            self.resume_ckpt_path, ".tmp"])
        resume_ckpt_path_bak = ''.join([
            self.resume_ckpt_path, ".bak"])

        with open(resume_ckpt_path_tmp, 'w') as ckpt_file:
            json.dump(resume_dict, ckpt_file)
        
        # Back up the previous checkpoint file
        if os.path.exists(self.resume_ckpt_path):
            os.rename(self.resume_ckpt_path,
                resume_ckpt_path_bak)

        # Rename the freshly written file to the definite
        # resume_cpkt file path
        os.rename(resume_ckpt_path_tmp,
            self.resume_ckpt_path)

    def get_resume_ckpt(self, model_ckpt_name=None):
        resume_dict = None
        if not os.path.exists(self.resume_ckpt_path):
            raise Exception(f"Attempting to resume {self.logdir}, but could not find the 'resume_ckpt_path'")

        if model_ckpt_name is not None:
            model_savepath = os.path.join(
                self.get_models_savedir(),
                model_ckpt_name)
            
            try:
                # If this succeeds, no need to use the backup
                th.load(model_savepath, map_location=th.device("cpu"))
            
            except Exception as e:
                # Restore the backup as the main ckpt file
                model_savepath_bak = ''.join([
                    model_savepath,
                    ".bak"
                ])

                # Restore the previous model file from the backup
                if os.path.exists(model_savepath_bak):
                    shutil.copy2(model_savepath_bak,
                        model_savepath)
                else:
                    # TODO: abort the experiment
                    pass
                
                # Restore the previous resume_ckpt from the backpu
                resume_cpkt_bak_path = ''.join([
                    self.resume_ckpt_path, ".bak"])
                
                if os.path.exists(resume_cpkt_bak_path):
                    shutil.copy2(resume_cpkt_bak_path,
                        self.resume_ckpt_path)
                else:
                    # TODO: abort the experiment ?
                    pass
                
                print(e)
        
        # Load the latest resume_ckpt dict
        with open(self.resume_ckpt_path, 'r') as ckpt_path:
            resume_dict = json.load(ckpt_path)

        if not isinstance(resume_dict, dict):
            raise Exception(f"Loaded 'resume_ckpt' is not a dict")

        return resume_dict

    def get_resume_args(self):
        resume_args_filepath = os.path.join(self.logdir, "hyparams.json")
        with open(resume_args_filepath, "r") as f:
            resume_args = json.load(f)
        
        return resume_args

    def get_script_filename(self):
        script_filename = sys.argv[0] # Broke on 2021-05-05 for some reason ...
        
        return script_filename.split('/')[-1]
    
    def get_resume_ckpt_base(self):
        if self.args.resume == "":
            RESUME_CKPT_BASE = {
                # "git_branch": f"{self.get_git_branch()}",
                # "git_hash": f"{self.get_git_hash()}",
                "script_filename": f"{self.get_script_filename()}",
                "wandb_id": f"{self.wandb_id}",
                "total_steps": self.args.total_steps,
                "finished": False
            }
        else:
            resume_ckpt_dict = self.get_resume_ckpt()
            # DO not try to read git_branch when it might 
            # have been detached to previous commit already
            # due to resume operation. Instead, read it from
            # the resume ckpt, since this is not expected to
            # change.
            RESUME_CKPT_BASE = {
                # "git_branch": resume_ckpt_dict["git_branch"],
                # "git_hash": resume_ckpt_dict["git_hash"],
                "script_filename": resume_ckpt_dict["script_filename"],
                "wandb_id": resume_ckpt_dict["wandb_id"],
                "total_steps": resume_ckpt_dict["total_steps"],
                "finished": resume_ckpt_dict["finished"]
            }
        
        return RESUME_CKPT_BASE

    def save_model_dict(self, state_dict, model_savename):
        # A safer model saving method that has one level of 
        # backup ot handle the case when experiment interrupted
        # during model writing
        model_savepath = os.path.join(
            self.get_models_savedir(),
            model_savename)
            
        model_savepath_tmp = ''.join([
            model_savepath, ".tmp"
        ])
        model_savepath_bak = ''.join([
            model_savepath, ".bak"
        ])

        # Saves model weights, and optimizer states
        th.save(state_dict, model_savepath_tmp)
        
        # Backup the previously existing model file
        if os.path.exists(model_savepath):
            os.rename(model_savepath, model_savepath_bak)
        
        # Rename the .tmp written file to the latest one
        os.rename(model_savepath_tmp, model_savepath)

    def close(self):
        if hasattr( self, '_tb_writer'):
            if self.tb_writer is not None:
                self.tb_writer.close()

    # Note: by using Wandb tensorboard sync, no need to call wandb.log()
    def log_image(self, name, image_data, step, prefix=None, nrows=None):
        fulltag = prefix + "/" + name if prefix is not None else name
        self.tb_writer.add_image(fulltag, image_data, step)
        # TODO: save to disk logic

    def log_pyplot(self, name, plt_object, step, prefix=None):
        fulltag = prefix + "/" + name if prefix is not None else name
        self.tb_writer.add_figure(fulltag, plt_object, step)
        # TODO: save to disk logic. Can be disk space expensive depending on what is logged
        # fig_savepath = os.path.join(self.get_images_savedir(), f"{name}_at_step_{step}.jpg")
        # plt_object.savefig(fig_savepath)

    def log_video(self, name, video_data, step, fps=60, prefix=None):
        fulltag = prefix + "/" + name if prefix is not None else name
        self.tb_writer.add_video(tag=fulltag, vid_tensor=video_data,
            global_step=step, fps=fps)
    
    def log_wandb_video_audio(self, name, filepath):
        if self.args.wandb:
            # The experiment appears to be using Wandb.
            # By default, it syncs with the tensorboard, so the 'step' cannot be overriden
            # to exactly match the scalar metrics.
            # NOTE: step and fps are probably useless
            wandb.log({name: wandb.Video(filepath)})
