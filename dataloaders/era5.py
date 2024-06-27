# dataloader on CLEPS
import apache_beam   # Needs to be imported separately to avoid TypingError
import xarray as xr
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import random
import os
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm
import random
from .preprocessors import default_preprocessor
from functools import partial

default_filename_filter = dict(all= (lambda _: True),
                               last_train= lambda x:('2018' in x),
                               train= lambda x: not('2019' in x or '2020' in x or '2021' in x),
                               val= lambda x: ('2018' in x or '2019' in x or '2020' in x), # to handle a bit before and after.
                               test= lambda x: ('2019' in x or '2020' in x or '2021' in x))

class NetcdfDataset(torch.utils.data.Dataset):
    '''
    dataset to read a list of netcdf files and iterate through it
    '''
    def __init__(self,
                 path,
                 filename_filter=lambda _:True, # condition to keep file in dataset
                 engine='netcdf4',
                 variables = None,
                prep=lambda x: x):
        
        if variables is None:
            self.variables = dict(state_level=['geopotential', 'u_component_of_wind', 'v_component_of_wind', 'temperature', 'specific_humidity'], 
                                  state_surface=['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'mean_sea_level_pressure'])
        else:
            self.variables = variables
        self.prep = prep
        self.filename_filter = filename_filter
        
        if '.nc' in path:
            self.files = [path]
        else:
            self.files = sorted([str(x) for x in Path(path).glob('*.nc') if filename_filter(x.name)], key=lambda x: x.replace('6h', '06h').replace('0h','00h'))
        self.nfiles = len(self.files)
        self.xr_options = dict(engine=engine, cache=True)
        
        self.timestamps = []

        for fid, f in tqdm(enumerate(self.files)):
            with xr.open_dataset(f, **self.xr_options) as obs:
                file_stamps = [(fid, i, t) for (i, t) in enumerate(obs.time.to_numpy())]
                self.timestamps.extend(file_stamps)

        self.timestamps = sorted(self.timestamps, key=lambda x: x[-1]) # sort by timestamp
        #self.id2pt = {i:(file_id, line_id) for (i, (file_id, line_id, s)) in enumerate(self.timestamps)}
        self.id2pt = dict(enumerate(self.timestamps))

        self.cached_xrdataset = None
        self.cached_fileid = None

    def __len__(self):
        return len(self.id2pt)
        
    def __getitem__(self, i):
        out = {}
        file_id, line_id, timestamp = self.id2pt[i]

        if self.cached_fileid != file_id:
            if self.cached_xrdataset is not None:
                self.cached_xrdataset.close()
            self.cached_xrdataset = xr.open_dataset(self.files[file_id], **self.xr_options)
            self.cached_fileid = file_id

        obsi = self.cached_xrdataset.isel(time=line_id)

        out = dict()
        out['time'] = obsi.time.dt.strftime('%Y-%m-%d-%H-%M').item()
        
        for vcat, vnames in self.variables.items():
            data_np = obsi[vnames].to_array().to_numpy()
            out[vcat] = torch.from_numpy(data_np)
            out[vcat] = self.prep(out[vcat])
            if out[vcat].isnan().any().item():
                print(i, file_id, line_id, self.files[file_id])
                raise ValueError('NaN values detected !')

        return out     

class Era5(NetcdfDataset):
    '''
    era5 dataloader. for now it just links the default preprocessor with netcdf loader
    '''
    def __init__(self,
                 path,
                 domain,
                 engine='netcdf4',
                 variables = None,
                 pangulite_split=False,
                 pad_128=False):

        filename_filter = default_filename_filter[domain]
        
        prep = partial(default_preprocessor, pad_128=pad_128)
        super().__init__(path, filename_filter, engine, variables, prep=prep)
        
    def __getitem__(self, i):
        out = super().__getitem__(i)
        out['state_surface'] = out['state_surface'].unsqueeze(-3)
        return out

class Era5Forecast(NetcdfDataset):
    '''
    this dataset uses the previous one, but adds normalization
    '''
    def __init__(self, 
                 path='data/era5_240/full/',
                 domain='train',
                 engine='netcdf4',
                 filename_filter=None, 
                 variables = None,
                pad_128=False,
                lead_time_hours=24,
                multistep=1,
                load_prev=False,
                load_clim=False,
                input_norm_scheme='pangu', #'global', # or 'clim' # 'pangu' ?
                 output_norm_scheme='delta24',
                 z0012 = False,
                 data_augmentation = False,
                 train_split = 'all', # or 'even' or 'odd'
                 include_vertical_wind_component=False):
        self.__dict__.update({k:v for k, v in locals().items() if k!='self'})
        if variables is None:
            variables = dict(state_level=['geopotential', 'u_component_of_wind', 'v_component_of_wind', 'temperature', 'specific_humidity'], 
                             state_surface=['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'mean_sea_level_pressure'])
        
        if include_vertical_wind_component:
            variables['state_level'] = variables['state_level'] + ['vertical_velocity']
        
        self.timedelta = 6 if not z0012 else 12
        self.current_multistep = 1

        if filename_filter is None:
            filename_filter = default_filename_filter[domain]
            
        if z0012:
            filename_filter2 = lambda x: filename_filter(x) and ('0h' in x or '12h' in x)
        else:
            filename_filter2 = filename_filter

        if domain == 'train' and train_split == 'recent':
            filename_filter2 = lambda x: any([str(y) in x for y in range(2000, 2019)])
        elif domain == 'train' and train_split == 'recent2':
            filename_filter2 = lambda x: any([str(y) in x for y in range(2007, 2019)])


        prep = partial(default_preprocessor, pad_128=pad_128)

        super().__init__(path, filename_filter2, engine, variables, prep=prep)

        if domain in ('val', 'test'):
            # re-select timestamps
            year = 2019 if domain == 'val' else 2020
            start_time = np.datetime64(f'{year}-01-01T00:00:00')
            if self.load_prev:
                start_time = start_time - self.lead_time_hours*np.timedelta64(1, 'h')
            end_time = np.datetime64(f'{year+1}-01-01T00:00:00') + self.multistep*self.lead_time_hours*np.timedelta64(1, 'h')
            self.timestamps = [x for x in self.timestamps if start_time <= x[-1].astype('datetime64[s]')<end_time] # sort by timestamp
            self.id2pt = dict(enumerate(self.timestamps))

        # load the constant mask
        
        self.constant_masks = torch.load('stats/archesweather_constant_masks.pt')

        # np_masks = np.stack([np.load(f'stats/{m}.npy') 
        #                  for m in ('land_mask', 'soil_type', 'topography')])
        # th_masks = torch.from_numpy(np_masks)[:, :-1, :].float()
        # th_masks = torch.cat([th_masks[..., 720:], th_masks[..., :720]], dim=-1) # focus on Europe
        # th_masks -= th_masks.mean((-2, -1), keepdim=True)
        # th_masks /= th_masks.std((-2, -1), keepdim=True)
        # th_masks = th_masks.unsqueeze(-3)
        # self.constant_masks = TF.resize(th_masks, (120, 240), antialias=True)
    
        if include_vertical_wind_component:
            pangu_stats = torch.load('stats/pangu_norm_stats2_with_w.pt')
        else:
            pangu_stats = torch.load('stats/pangu_norm_stats2.pt')

        #re-order to match current stuff
        # pangu_stats = dict(surface_mean=pangu_stats['surface_mean'][[1, 2, 3, 0]],
        #                surface_std=pangu_stats['surface_std'][[1, 2, 3, 0]],
        #                level_mean=torch.flip(pangu_stats['level_mean'][[0, 3, 4, 2, 1]].squeeze(1), [-3]),
        #                level_std=torch.flip(pangu_stats['level_std'][[0, 3, 4, 2, 1]].squeeze(1), [-3]))
        
        self.output_level_stds = torch.tensor(1)
        self.output_surface_stds = torch.tensor(1)
        self.input_level_means = torch.tensor(0)
        self.input_level_stds = torch.tensor(1)
        self.input_surface_means = torch.tensor(0)
        self.input_surface_stds = torch.tensor(1)

        if self.input_norm_scheme == 'pangu':
            self.input_surface_means = pangu_stats['surface_mean']
            self.input_surface_stds = pangu_stats['surface_std']
            self.input_level_means = pangu_stats['level_mean']
            self.input_level_stds = pangu_stats['level_std']
         
        if output_norm_scheme == 'delta24':
            if include_vertical_wind_component:
                self.output_level_stds = torch.tensor([5.9786e+02, 7.4878e+00, 8.9492e+00, 2.7132e+00, 9.5222e-04, 0.3])[:, None, None, None]
            else:
                self.output_level_stds = torch.tensor([5.9786e+02, 7.4878e+00, 8.9492e+00, 2.7132e+00, 9.5222e-04])[:, None, None, None]

            self.output_surface_stds = torch.tensor([  3.8920,   4.5422,   2.0727, 584.0980])[:, None, None, None]

        elif output_norm_scheme == 'delta24full': # per-level normalization
            if include_vertical_wind_component:
                out_stats = torch.load('stats/delta24_stats_with_w.pt')                
            else:
                out_stats = torch.load('stats/delta24_stats.pt')
            self.output_level_stds = out_stats['level_std']
            self.output_surface_stds = out_stats['surface_std']
          
        # store coeffs
        lat_coeffs_equi = torch.tensor([torch.cos(x) for x in torch.arange(-torch.pi/2, torch.pi/2, torch.pi/120)])
        self.lat_coeffs_equi =  (lat_coeffs_equi/lat_coeffs_equi.mean())[None, None, None, :, None]

        # variable names
        self.pressure_levels = [  50,  100,  150,  200,  250,  300,  400,  500,  600,  700,  850,  925, 1000]
        self.surface_variables = ['U10', 'V10', 'T2m', 'SP']
        self.level_variables = [a+str(p) for a in ['Z', 'U', 'V', 'T', 'Q'] for p in self.pressure_levels]

        # load clim
        if self.load_clim:
            self.clim = xr.open_dataset(Path(path).parent.joinpath('era5_240_clim.nc'))


    def __len__(self):
        offset = self.multistep + self.load_prev
        return super().__len__() - offset * self.lead_time_hours//self.timedelta
            
        
    def __getitem__(self, i, normalize=True):
        out = {}
        # shift i if needed
        i = i + self.lead_time_hours//self.timedelta if self.load_prev else i
            
        #  load current sate
        out['state_constant'] = self.constant_masks
        obsi = super().__getitem__(i)
        out['time'] = obsi['time']
        out['state_surface'] = obsi['state_surface'].unsqueeze(-3)
        out['state_level'] = obsi['state_level']
        
        # next obsi. has function of 
        T = self.lead_time_hours # multistep

        next_obsi = super().__getitem__(i+T//self.timedelta)
        out['next_time'] = next_obsi['time']
        out['next_state_surface'] = next_obsi['state_surface'].unsqueeze(-3)
        out['next_state_level'] = next_obsi['state_level']

        # multistep
        if self.multistep > 1:
            out['lead_time_hours'] = torch.tensor([self.lead_time_hours*int(self.multistep)]).float()
            future_obsis = []
            for k in range(2, self.multistep+1):
                future = super().__getitem__(i+k*T//self.timedelta)
                future_obsis.append(future)
            
            out['future_time'] = future_obsis[-1]['time']
            out['future_state_surface'] = torch.stack([obsi['state_surface'].unsqueeze(-3)
                                                       for obsi in future_obsis], dim=0)
            out['future_state_level'] = torch.stack([obsi['state_level']
                                                     for obsi in future_obsis], dim=0)


        if self.load_prev:
            prev_obsi = super().__getitem__(i - self.lead_time_hours//self.timedelta)
            out['prev_time'] = prev_obsi['time']
            out['prev_state_surface'] = prev_obsi['state_surface'].unsqueeze(-3)
            out['prev_state_level'] = prev_obsi['state_level']

        if self.load_clim:
            _, _, timestamp = self.id2pt[i+T//self.timedelta]
            doy = np.datetime64(timestamp, 'D') - np.datetime64(timestamp, 'Y') + 1
            hour = (timestamp.astype('datetime64[h]') - timestamp.astype('datetime64[D]')).astype(int) % 24
            climi = self.clim.sel(dayofyear=doy.astype('int'), hour=hour)
            srf_np = climi[self.variables['state_surface']].to_array().to_numpy()
            out['clim_state_surface'] = self.prep(torch.from_numpy(srf_np))

            lvl_np = climi[self.variables['state_level']].to_array().to_numpy()
            out['clim_state_level'] = self.prep(torch.from_numpy(lvl_np))

        if normalize:
            out = self.normalize(out)
        
        if self.domain == 'train' and self.data_augmentation:
            from dataloaders.preprocessors import lonshift_augmentation
            out = lonshift_augmentation(out, rg=25, prob=0.8)
        return out
 
    def normalize(self, batch):
        device = batch['state_level'].device
        out = {k:v for k, v in batch.items()}
        
        if self.input_norm_scheme in ('global', 'pangu'):
            out['state_surface'] = (batch['state_surface'] - self.input_surface_means.to(device)) / self.input_surface_stds.to(device)
            out['state_level'] = (batch['state_level'] - self.input_level_means.to(device)) / self.input_level_stds.to(device)
        else:
            out['state_surface'] = batch['state_surface']
            out['state_level'] = batch['state_level']

        if self.output_norm_scheme in ('delta', 'climdeltafull', 'deltafull', 'delta24', 'delta24full'):
            if 'next_state_level' in batch:
                out['next_state_surface'] = (batch['next_state_surface'] - batch['state_surface']) / self.output_surface_stds.to(device)
                out['next_state_level'] = (batch['next_state_level'] - batch['state_level']) / self.output_level_stds.to(device)
                
            if 'future_state_level' in batch and self.multistep > 1:
                out['future_state_surface'] = (batch['future_state_surface'] - batch['state_surface'][None]) / self.output_surface_stds[None].to(device)# / multistep_scaler
                out['future_state_level'] = (batch['future_state_level'] - batch['state_level'][None]) / self.output_level_stds[None].to(device)# / multistep_scaler
            
            # WARNING: these are normalized with the correct order
            if 'prev_state_level' in batch and self.load_prev:
                out['prev_state_surface'] = (batch['state_surface'] - batch['prev_state_surface']) / self.output_surface_stds.to(device)
                out['prev_state_level'] = (batch['state_level'] - batch['prev_state_level']) / self.output_level_stds.to(device)
    
        elif self.output_norm_scheme == 'pangu':
            out['next_state_surface'] = (batch['next_state_surface'] - self.input_surface_means.to(device)) / self.input_surface_stds.to(device)
            out['next_state_level'] = (batch['next_state_level'] - self.input_level_means.to(device)) / self.input_level_stds.to(device)
        else:
            out['next_state_surface'] = batch['next_state_surface']
            out['next_state_level'] = batch['next_state_level']         

        return out

    def denormalize(self, input, batch=None):
        # uses batch['state_level'], batch['state_surface']
        if batch is None:
            batch = input
        device = batch['state_level'].device
        denorm_output = {k:v for k, v in input.items()}

        # need state level to denormalize
        denorm_state_level = batch['state_level']*self.input_level_stds.to(device) + self.input_level_means.to(device)
        denorm_state_surface = batch['state_surface']*self.input_surface_stds.to(device) + self.input_surface_means.to(device)

        if 'state_level' in input.keys():
            denorm_output['state_level'] = denorm_state_level
            denorm_output['state_surface'] = denorm_state_surface
            
        if self.output_norm_scheme == 'pangu':
            denorm_output['next_state_level'] = input['next_state_level']*self.input_level_stds.to(device) + self.input_level_means.to(device)
            denorm_output['next_state_surface'] = input['next_state_surface']*self.input_surface_stds.to(device) + self.input_surface_means.to(device)

        elif self.output_norm_scheme in ('delta24', 'delta'):
            if 'next_state_level' in input:
                denorm_output['next_state_level'] = input['next_state_level']*self.output_level_stds.to(device) + denorm_state_level
                denorm_output['next_state_surface'] = input['next_state_surface']*self.output_surface_stds.to(device) + denorm_state_surface

            if 'prev_state_level' in input: # not really needed but anyway
                denorm_output['prev_state_surface'] = denorm_state_surface - input['prev_state_surface']*self.output_surface_stds.to(device)
                denorm_output['prev_state_level'] = denorm_state_level - input['prev_state_level']*self.output_level_stds.to(device)

            if 'future_state_level' in input:
                denorm_output['future_state_level'] = input['future_state_level']*self.output_level_stds[None].to(device) + denorm_state_level[:, None]
                denorm_output['future_state_surface'] = input['future_state_surface']*self.output_surface_stds[None].to(device) + denorm_state_surface[:, None]


        else:
            raise ValueError('unknown output_norm_scheme')
        return denorm_output
    

    def normalize_next_batch(self, pred, batch):
        # takes a prediction as input and renormalizes it for next batch
        date = [pd.to_datetime(x, utc=True, format='%Y-%m-%d-%H-%M').tz_localize(None) for x in batch['time']]
        next_time = [(dt + pd.Timedelta(hours=24)).strftime('%Y-%m-%d-%H-%M') for dt in date]
        denorm_pred = self.denormalize(pred, batch)
        denorm_batch = self.denormalize(batch)

        denorm_next_batch = dict(prev_state_level=denorm_batch['state_level'],
                                prev_state_surface=denorm_batch['state_surface'],
                                state_level=denorm_pred['next_state_level'],
                                state_surface=denorm_pred['next_state_surface'],
                                state_constant=denorm_batch['state_constant'],
                                lead_time_hours=denorm_batch['lead_time_hours'],
                                time=next_time)   
        next_batch = self.normalize(denorm_next_batch)
        return next_batch
        
        
    
        