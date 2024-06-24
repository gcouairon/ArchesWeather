#submitit file
import submitit
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf
from omegaconf import open_dict

try:
    OmegaConf.register_new_resolver("eval", eval)
except:
    pass

from train_hydra import main as train_main

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    aex = submitit.AutoExecutor(folder=cfg.cluster.folder, cluster='slurm')
    aex.update_parameters(**cfg.cluster.launcher) # original launcher
    aex.submit(train_main, cfg)

if __name__ == '__main__':
    main()