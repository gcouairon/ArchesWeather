<p align='center'>
<svg version="1.1" viewBox="0.0 0.0 310.6062992125984 165.1522309711286" fill="none" stroke="none" stroke-linecap="square" stroke-miterlimit="10" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns="http://www.w3.org/2000/svg"><clipPath id="p.0"><path d="m0 0l310.6063 0l0 165.15224l-310.6063 0l0 -165.15224z" clip-rule="nonzero"/></clipPath><g clip-path="url(#p.0)"><path fill="#000000" fill-opacity="0.0" d="m0 0l310.6063 0l0 165.15224l-310.6063 0z" fill-rule="evenodd"/><path fill="#000000" fill-opacity="0.0" d="m23.283464 135.3673c6.92651 -18.50763 27.706476 -111.04579 41.55906 -111.04579c13.852577 0 34.630356 92.53816 41.556427 111.04579" fill-rule="evenodd"/><path stroke="#741b47" stroke-width="24.0" stroke-linejoin="round" stroke-linecap="butt" d="m23.283464 135.3673c6.92651 -18.50763 27.706476 -111.04579 41.55906 -111.04579c13.852577 0 34.630356 92.53816 41.556427 111.04579" fill-rule="evenodd"/><path fill="#000000" fill-opacity="0.0" d="m127.57193 26.391031c6.5494385 18.079615 25.820648 107.93044 39.296585 108.4777c13.475952 0.5472412 27.70604 -104.978134 41.559067 -105.19423c13.853012 -0.21609688 27.541992 104.87227 41.55905 103.89763c14.017059 -0.97462463 35.45276 -91.4545 42.543304 -109.7454" fill-rule="evenodd"/><path stroke="#1c4587" stroke-width="24.0" stroke-linejoin="round" stroke-linecap="butt" d="m127.57193 26.391031c6.5494385 18.079615 25.820648 107.93044 39.296585 108.4777c13.475952 0.5472412 27.70604 -104.978134 41.559067 -105.19423c13.853012 -0.21609688 27.541992 104.87227 41.55905 103.89763c14.017059 -0.97462463 35.45276 -91.4545 42.543304 -109.7454" fill-rule="evenodd"/><path fill="#741b47" d="m11.786552 133.84822l0 0c0 -6.6796036 5.4148865 -12.09449 12.094488 -12.09449l0 0c3.2076569 0 6.2839375 1.2742386 8.552095 3.5423965c2.2681541 2.2681503 3.5423927 5.3444366 3.5423927 8.5520935l0 0c0 6.679596 -5.4148865 12.094482 -12.094488 12.094482l0 0c-6.6796017 0 -12.094488 -5.4148865 -12.094488 -12.094482z" fill-rule="evenodd"/><path fill="#741b47" d="m94.26063 135.53874l0 0c0 -6.679596 5.4148865 -12.094482 12.094482 -12.094482l0 0c3.2076569 0 6.283943 1.2742386 8.5520935 3.542389c2.268158 2.268158 3.5423965 5.3444366 3.5423965 8.5520935l0 0c0 6.679611 -5.4148865 12.094498 -12.09449 12.094498l0 0c-6.679596 0 -12.094482 -5.4148865 -12.094482 -12.094498z" fill-rule="evenodd"/><path fill="#741b47" d="m52.93566 92.74347l0 0c0 -6.5752335 5.3302803 -11.90551 11.90551 -11.90551l0 0c3.1575394 0 6.185753 1.2543259 8.418472 3.4870453c2.2327118 2.2327118 3.4870453 5.2609253 3.4870453 8.418465l0 0c0 6.5752335 -5.330284 11.90551 -11.905518 11.90551l0 0c-6.5752296 0 -11.90551 -5.3302765 -11.90551 -11.90551z" fill-rule="evenodd"/><path fill="#1c4587" d="m115.668564 26.723763l0 0c0 -6.6796017 5.4148865 -12.094488 12.09449 -12.094488l0 0c3.2076492 0 6.283943 1.2742376 8.552086 3.5423946c2.268158 2.268156 3.5424042 5.3444366 3.5424042 8.5520935l0 0c0 6.6796 -5.4148865 12.09449 -12.09449 12.09449l0 0c-6.6796036 0 -12.09449 -5.4148903 -12.09449 -12.09449z" fill-rule="evenodd"/><path fill="#1c4587" d="m280.4835 23.714651l0 0c0 -6.6796017 5.4148865 -12.094489 12.094513 -12.094489l0 0c3.2076416 0 6.2839355 1.2742376 8.5520935 3.5423937c2.268158 2.268156 3.542389 5.3444366 3.542389 8.552095l0 0c0 6.6796017 -5.4148865 12.094488 -12.094482 12.094488l0 0c-6.6796265 0 -12.094513 -5.4148865 -12.094513 -12.094488z" fill-rule="evenodd"/></g></svg>
</p>

## General present![logo2](https://github.com/gcouairon/ArchesWeather/assets/24316340/bad8bf80-7528-4e08-8072-ad84fc29d987)
ation

This codebase is the code for running and training [ArchesWeather](https://arxiv.org/abs/2405.14527).

Below is an exemple of a 10-day rollout for the ArchesWeather-M model initialized on January 1st, 2020 (with rollout steps of 24h).

<p align='center'>

https://github.com/gcouairon/ArchesWeather/assets/24316340/bd775f39-0d98-4b26-acf7-9b72f01625fd

</p>


## Installation

### Environment

```sh
conda create --name weather python=3.10
conda activate weather
pip install -r requirements.txt

mkdir sblogs
```

We recommend making the following symlinks in the codebase folder:
```sh
ln -s /path/to/data/ data
ln -s /path/to/models/ modelstore
ln -s /path/to/evaluation/ evalstore
ln -s /path/to/wandb/ wandblogs
```
Where `/path/to/models/` is where the trained models are stored, and `/path/to/evaluation/` is a folder used to store intermediate outputs from evaluating models. You can also simply create folders if you want to store data in the same folder.


### Data

The ``dl_era.py`` scripts downloads data from WeatherBench as netcdf files, because it was originally used on a system that could not handle the many files of the zarr storage system.
You can download the full dataset sequentially via `python dl_era.py`. If you wish to download the dataset in parrallel using multiple workers, you can download specific years with the script, e.g. via

```sh
python dl_era.py --clim # to download climatology for ACC metrics
python dl_era.py --year 2019,2020,2021 # to download specific years
```
You should download data from Weatherbench for years 1979 to 2021 (included). By default the dataset will be downloaded to `data/era5_240/`.


### Download model

```sh
mkdir modelstore/archesweather-M
src=https://huggingface.co/gcouairon/ArchesWeather/resolve/main
tgt=modelstore/archesweather-M
wget -O $tgt/archesweather-M_weights.pt $src/archesweather-M_weights.pt 
wget -O $tgt/archesweather-M_config.yaml $src/archesweather-M_config.yaml 
```

You can run a similar command to download the ArchesWeather-S model.


## ArchesWeather Inference

Here is a quick snippet on how to load an ArchesWeather model and perform inference:

```python
from omegaconf import OmegaConf
from hydra.utils import instantiate
import matplotlib.pyplot as plt
import torch

torch.set_grad_enabled(False)

# load model and dataset
device = 'cuda:0'
cfg = OmegaConf.load('modelstore/archesweather-M/archesweather-M_config.yaml')

ds = instantiate(cfg.dataloader.dataset, 
                    path='data/era5_240/full/', 
                    domain='test') # the test domain is year 2020

backbone = instantiate(cfg.module.backbone)
module = instantiate(cfg.module.module, backbone=backbone, dataset=ds)

ckpt = torch.load('modelstore/archesweather-M/archesweather-M_weights.pt', map_location='cpu')
module.load_state_dict(ckpt)
module = module.to(device).eval()


# make a batch
batch = {k:(v[None].to(device) if hasattr(v, 'to') else [v]) for k, v in ds[0].items()}
output = module.forward(batch)

# denormalize output
denorm_pred = ds.denormalize(output, batch)

# get per-sample main metrics from WeatherBench
from evaluation.deterministic_metrics import headline_wrmse
denorm_batch = ds.denormalize(batch)
metrics = headline_wrmse(denorm_pred, denorm_batch, prefix='next_state')

# average metrics
metrics_mean = {k:v.mean(0) for k, v in metrics.items()}

#plot prediction
plt.imshow(denorm_pred['next_state_surface'][0, 2, 0].detach().cpu().numpy())

```

Multistep inference:

```python

multistep = 10
norm_batch = {k:(v.to(device) if hasattr(v, 'to') else v) for k, v in ds[0].items()}


#alternatively
traj = dict(traj_surface=[], traj_level=[])
for i in range(multistep):
    pred = module.forward(norm_batch)
    denorm_pred = ds.denormalize(pred, norm_batch)
    norm_batch = ds.normalize_next_batch(pred, norm_batch)

    traj['traj_surface'].append(denorm_pred['next_state_surface'].cpu().detach())
    traj['traj_level'].append(denorm_pred['next_state_level'].cpu().detach())


```

## Codebase logic

The codebase uses pytorch lightning, hydra, and logs data to Weights and Biases by default. For submission to SLURM it uses the submitit package. 

the configs are stored in `configs` folder. 
On each computing infrastructure, you can define the following alias
```sh
alias train='python submit.py cluster=example-slurm'
alias debug='python train_hydra.py cluster=example-slurm'

```
where `example-slurm` is the file in `configs/cluster` that contains information about how jobs should be started.

*train* submits the job to SLURM while *debug* starts the job directly. *train* will log to Weights and Biases by default, unlike *debug*.

## Training ArchesWeather

Example command on how to train ArchesWeather:

```python
train module=forecast-archesweather dataloader=era5-w
```

The target module is `lightning_modules.forecast.ForecastModule`, which is initialized with a backbone model defined in `backbones/archesweather`.

To override parameters:

```python
train module=forecast-gco dataloader=era5-w \
"++name=archesweather-s" \
"++module.backbone.depth_multiplier=1" \
```

The training script handles SLURM pre-emption: when a job is pre-empted, the script saves a checkpoint and requeues a job that will resume the current run.

By default, if you try to start a run that has the same name as a previous run, the script will check if the configurations for module and datasets are the same. If yes, it will resume the previous run, if not it will issue an error message and exit.


## External resources

Many thanks to the authors of WeatherLearn for adapting the Pangu-Weather pseudocode to pytorch. The code for our model is mostly based on their codebase.

[WeatherBench](https://sites.research.google/weatherbench/)

[WeatherLearn](https://github.com/lizhuoq/WeatherLearn/tree/master)

