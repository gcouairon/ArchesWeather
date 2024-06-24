import pytorch_lightning as pl
import torch
from utils.load_config import load_config
import submitit
from pathlib import Path
import shutil
from functools import partial
import subprocess
import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from hydra.utils import instantiate
import os
from datetime import datetime
import time

from omegaconf import OmegaConf
from pytorch_lightning.plugins.environments import SLURMEnvironment
import signal


try:
    from pytorch_lightning.callbacks import Callback, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
except:
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger



def get_random_code():
    import string
    import random
    # generate random code that alternates letters and numbers
    l = random.choices(string.ascii_lowercase, k=3)
    n = random.choices(string.digits, k=3)
    return ''.join([f'{a}{b}' for a, b in zip(l, n)])


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        dirpath='./',
        save_step_frequency=50000,
        prefix="checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename
        self.dirpath = dirpath

    def on_train_batch_end(self, trainer: pl.Trainer, *args, **kwargs):
        """ Check if we should save a checkpoint after every train batch """
        if not hasattr(self, 'trainer'):
            self.trainer = trainer

        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            self.save()
       
    def save(self, *args, trainer=None, **kwargs):
        if trainer is None and not hasattr(self, 'trainer'):
            print('No trainer !')
            return
        if trainer is None:
            trainer = self.trainer

        global_step = trainer.global_step
        if self.use_modelcheckpoint_filename:
            filename = trainer.checkpoint_callback.filename
        else:
            filename = f"{self.prefix}_{global_step=}.ckpt"
        ckpt_path = Path(self.dirpath) / 'checkpoints'
        ckpt_path.mkdir(exist_ok=True, parents=True)
        trainer.save_checkpoint(ckpt_path / filename)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    try:
        OmegaConf.register_new_resolver("eval", eval)
    except:
        pass
    
    main_node = int(os.environ.get('SLURM_PROCID', 0)) == 0
    print('is main node', main_node)

    # init some variables
    logger = None
    ckpt_path = None
    # delete submitit handler to let PL take care of resuming
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


    # first, check if exp exists
    if Path(cfg.exp_dir).exists():
        print('Experiment already exists. Trying to resume it.')
        exp_cfg = OmegaConf.load(Path(cfg.exp_dir) / 'config.yaml')
        if cfg.resume:
            cfg = exp_cfg
        else:
            # check that new config and old config match
            if OmegaConf.to_yaml(cfg.module, resolve=True) != OmegaConf.to_yaml(exp_cfg.module):
                print('Module config mismatch. Exiting')
                print('Old config', OmegaConf.to_yaml(exp_cfg.module))
                print('New config', OmegaConf.to_yaml(cfg.module))
                
            if OmegaConf.to_yaml(cfg.dataloader, resolve=True) != OmegaConf.to_yaml(exp_cfg.dataloader):
                print('Dataloader config mismatch. Exiting.')
                print('Old config', OmegaConf.to_yaml(exp_cfg.dataloader))
                print('New config', OmegaConf.to_yaml(cfg.dataloader))
                return
            
        # trying to find checkpoints
        ckpt_dir = Path(cfg.exp_dir).joinpath('checkpoints')
        if ckpt_dir.exists():
            ckpts = list(sorted(ckpt_dir.iterdir(), key=os.path.getmtime))
            if len(ckpts):
                print('Found checkpoints', ckpts)
                ckpt_path = ckpts[-1]  


    if cfg.log:
        os.environ['WANDB_DISABLE_SERVICE'] = 'True'
        print('wandb mode', cfg.cluster.wandb_mode)
        print('wandb service', os.environ.get('WANDB_DISABLE_SERVICE', 'variable unset'))
        run_id = cfg.name + '-'+get_random_code() if cfg.cluster.manual_requeue else cfg.name
        logger = pl.loggers.WandbLogger(project=cfg.project,
                                        name=cfg.name,
                                        id=run_id,
                                        save_dir=cfg.cluster.wandb_dir,
                                        offline=(cfg.cluster.wandb_mode != 'online'))
        

    if cfg.log and main_node and not Path(cfg.exp_dir).exists():
        print('registering exp on main node')
        hparams = OmegaConf.to_container(cfg, resolve=True)
        print(hparams)
        logger.log_hyperparams(hparams)
        Path(cfg.exp_dir).mkdir(parents=True)
        with open(Path(cfg.exp_dir) / 'config.yaml', 'w') as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))
            
    
    valset = instantiate(cfg.dataloader.dataset, domain='val')
    val_loader = torch.utils.data.DataLoader(valset, 
                                              batch_size=cfg.batch_size,
                                              num_workers=cfg.cluster.cpus,
                                              shuffle=True) # to viz shuffle samples
    
    trainset = instantiate(cfg.dataloader.dataset, domain='train')

    train_loader = torch.utils.data.DataLoader(trainset, 
                                                   batch_size=cfg.batch_size,
                                                   num_workers=cfg.cluster.cpus,
                                                   shuffle=True)

    backbone = instantiate(cfg.module.backbone)
    pl_module = instantiate(cfg.module.module, backbone=backbone,
                                        dataset=trainset)
    
    
    if hasattr(cfg, 'load_ckpt'):
        # load weights w/o resuming run
        pl_module.init_from_ckpt(cfg.load_ckpt)

    
    checkpointer = CheckpointEveryNSteps(dirpath=cfg.exp_dir,
                                         save_step_frequency=cfg.save_step_frequency)

    print('Manual submitit Requeuing')

    def handler(*args, **kwargs):
        print('GCO: SIGTERM signal received. Requeueing job on main node.')
        if main_node:
            checkpointer.save()
            from submit import main as submit_main
            if cfg.cluster.manual_requeue:
                submit_main(cfg)
        exit()
        
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, handler)



    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(cfg.seed)
    trainer = pl.Trainer(
                devices="auto",
                accelerator="auto",
                #strategy="auto",
                strategy="ddp_find_unused_parameters_true",
                precision=cfg.cluster.precision,
                log_every_n_steps=cfg.log_freq, 
                profiler=getattr(cfg, 'profiler', None),
                gradient_clip_val=1,
                max_steps=cfg.max_steps,
                callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=100),
                           checkpointer], 
                logger=logger,
                plugins=[],
                limit_val_batches=cfg.limit_val_batches, # max 5 samples
                accumulate_grad_batches=cfg.accumulate_grad_batches,
                )

    if cfg.debug:
        breakpoint()

    trainer.fit(pl_module, train_loader, val_loader, 
                ckpt_path=ckpt_path)

if __name__ == '__main__':
    main()