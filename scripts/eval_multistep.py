# script to evaluate a prediction made by two models
from omegaconf import OmegaConf
import hydra
from hydra.utils import instantiate
from pathlib import Path
import os
import torch
from tqdm import tqdm
import sys
import pandas as pd
import xarray as xr
import argparse
import numpy as np
from datetime import timedelta

sys.path.append(str(Path(__file__).parent.parent))

from evaluation.deterministic_metrics import acc, wrmse, headline_wrmse
from dataloaders.preprocessors import reverse_default_preprocessor

parser = argparse.ArgumentParser()
parser.add_argument('--force', action='store_true', help='whether to recompute with model')
parser.add_argument('--debug', action='store_true', help='whether to debug')
parser.add_argument('--max-lead-time', type=int, default=10, help='max lead time')
parser.add_argument('--uid', default="", type=str, help='model uid')
parser.add_argument('--step', default="", type=str, help='model uid')

parser.add_argument('--multistep', default=10, type=int, help='model uid')


args = parser.parse_args()


torch.set_grad_enabled(False)


device = 'cuda:0'
model_uid = args.uid
step= args.step or '50000'
save_path = f'evalstore/{model_uid}'

if Path(save_path).joinpath(f'test2020-step={step}-multistep.zarr').exists():
    if not args.force:
        print('output already exists. Exiting..')
        exit()
    else:
        import shutil
        shutil.rmtree(Path(save_path).joinpath(f'test2020-step={step}-multistep.zarr'))

Path(save_path).mkdir(parents=True, exist_ok=True)
cfg = OmegaConf.load(f'modelstore/{model_uid}/config.yaml')
ds_test = instantiate(cfg.dataloader.dataset, 
                    path='data/era5_240/full/', 
                    domain='test', 
                    z0012=True)
backbone = instantiate(cfg.module.backbone)
paths = sorted(Path('modelstore/{model_uid}/checkpoints').iterdir(), key=os.path.getmtime)
path = [x for x in paths if step in x.name][0]
print('using path', path)

module = instantiate(cfg.module.module, backbone=backbone, dataset=ds_test)
module.load_state_dict(torch.load(path, map_location='cpu')['state_dict'], strict=False)
module = module.to(device).eval()


outs = dict(traj_surface=[], traj_level=[], timestamps=[])

write_frequency = 10

raw_ds = instantiate(cfg.dataloader.dataset, 
                     input_norm_scheme=None,
                     output_norm_scheme=None,
                    path='data/era5_240/full/', 
                    domain='test', 
                    multistep=args.multistep,
                    z0012=True)

lt_dl = torch.utils.data.DataLoader(raw_ds, 
                                batch_size=1, 
                                num_workers=4, 
                                shuffle=False)

def get_trajectory(module, batch):
    batch = {k:(v.to(device) if hasattr(v, 'to') else v) for k, v in batch.items()}
    traj = dict(traj_surface=[], traj_level=[])
    norm_batch = ds_test.normalize(batch)
    for i in range(args.multistep):
        pred = module.forward(norm_batch)
        denorm_pred = ds_test.denormalize(pred, norm_batch)

        norm_batch = ds_test.normalize_next_batch(pred, norm_batch)

        traj['traj_surface'].append(denorm_pred['next_state_surface'].cpu().detach())
        traj['traj_level'].append(denorm_pred['next_state_level'].cpu().detach())

        #batch = ds_test.normalize(new_denorm_batch)
    
    traj = {k:torch.stack(v, dim=1) for k, v in traj.items()}

    return traj

err_log = []
for i, batch in tqdm(enumerate(lt_dl)):
    traj = get_trajectory(module, batch)
    outs['timestamps'].extend(batch['time'])
    outs['traj_surface'].append(traj['traj_surface'])
    outs['traj_level'].append(traj['traj_level'])
    # compute error
    gt = dict(traj_level=torch.cat([batch['next_state_level'][:, None], batch['future_state_level']], dim=1),
              traj_surface=torch.cat([batch['next_state_surface'][:, None], batch['future_state_surface']], dim=1))
    
    err_log.append(headline_wrmse(traj, gt, prefix='traj'))
    if args.debug:
        breakpoint()

    if not i % write_frequency or i == len(raw_ds)-1:
        level_outs = torch.cat(outs['traj_level']).cpu().detach()
        level_outs = reverse_default_preprocessor(level_outs).numpy()
        surface_outs = torch.cat(outs['traj_surface']).cpu().detach()
        surface_outs = reverse_default_preprocessor(surface_outs).numpy()

        # it should have the time dimension so that we can put it back in netcdf
        xds = xr.Dataset(
            data_vars=dict(
                **{lvl:(["time", "prediction_timedelta", "level", "latitude", "longitude"], level_outs[:, :, i]) for (i, lvl) in enumerate(ds_test.variables['state_level'])},
                **{sf:(["time", "prediction_timedelta", "latitude", "longitude"], surface_outs[:, :, i, 0]) for (i, sf) in enumerate(ds_test.variables['state_surface'])},
            ),
            coords=dict(time=pd.to_datetime(outs['timestamps'], utc=True, format='%Y-%m-%d-%H-%M').tz_localize(None), 
                        prediction_timedelta=[timedelta(i) for i in range(1, args.multistep+1)],
                        latitude=np.arange(-90, 90+1e-6, 180/120),
                        longitude=np.arange(0, 360, 360/240),
                        level=[50,  100,  150,  200,  250,  300,  400,  500,  600,  700,  850,  925,
       1000]
        ))
        xds.to_zarr(Path(save_path) / f'test2020-step={step}-multistep.zarr', append_dim='time' if i else None, 
                    encoding=dict(time=dict(units='hours since 2000-01-01')) if not i else None)
        outs = dict(traj_surface=[], traj_level=[], timestamps=[])

        avg_err = {k:torch.cat([e[k] for e in err_log], dim=0).mean(0) for k in err_log[0].keys()}
        print('errors', avg_err)

    if args.debug:
        breakpoint()

avg_err = {k:torch.cat([e[k] for e in err_log], dim=0).mean(0) for k in err_log[0].keys()} # avg on first dimension which is batch.
torch.save(avg_err, Path(save_path).joinpath(f'allmetrics-multistep-step={step}metrics.pt'))