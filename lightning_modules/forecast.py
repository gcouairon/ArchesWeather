import pytorch_lightning as pl
import torch.nn as nn
import torch
import diffusers
from pathlib import Path
import numpy as np
import torch.utils.checkpoint as gradient_checkpoint
from evaluation.deterministic_metrics import headline_wrmse


lat_coeffs_equi = torch.tensor([torch.cos(x) for x in torch.arange(-torch.pi/2, torch.pi/2, torch.pi/120)])
lat_coeffs_equi =  (lat_coeffs_equi/lat_coeffs_equi.mean())[None, None, None, :, None]

pressure_levels = torch.tensor([  50,  100,  150,  200,  250,  300,  400,  500,  600,  700,  850,  925,
       1000]).float()
level_coeffs = (pressure_levels/pressure_levels.mean())[None, None, :, None, None]
graphcast_surface_coeffs = torch.tensor([0.1, 0.1, 1.0, 0.1])[None, :, None, None, None] # graphcast
pangu_surface_coeffs = torch.tensor([0.25, 0.25, 0.25, 0.25])[None, :, None, None, None] # pangu coeffs

class ForecastModule(pl.LightningModule):
    def __init__(self, 
                 backbone,
                 dataset=None,
                 pow=2, # 2 is standard mse
                 lr=1e-4, 
                 betas=(0.9, 0.98),
                 weight_decay=1e-5,
                 num_warmup_steps=1000, 
                 num_training_steps=300000,
                 num_cycles=0.5,
                 use_graphcast_coeffs=False,
                 decreasing_pressure=False,
                 increase_multistep_period=2,
                 **kwargs
                ):
        ''' should create self.encoder and self.decoder in subclasses
        '''
        super().__init__()
        #self.save_hyperparameters()
        self.__dict__.update(locals())
        self.backbone = backbone # necessary to put it on device
        #self.area_weights = dataset.area_weights[None, None, None]
        self.area_weights = lat_coeffs_equi
        # if hasattr(dataset, 'area_weights'):
        #     self.area_weights = dataset.area_weights[None, None, None]
        # else:
        #     self.area_weights = dict(equi=lat_coeffs_equi, 
        #                          cubed=lat_coeffs_cubed)[mode]
            
        
        
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
        
    def forward(self, batch):
        input_surface = torch.cat([batch['state_surface'],
                                     batch['state_constant']], dim=1)
        input_surface = input_surface.squeeze(-3)
        if self.decreasing_pressure:
            input_level = batch['state_level'].flip(-3)
        else:
            input_level = batch['state_level']
                                     
        out = self.backbone(input_surface, input_level)
        return out

    def forward_multistep(self, batch):
        # multistep forward with gradient checkpointing to save GPU memory
        lead_iter = int((batch['lead_time_hours'][0]//24).item())
        preds_future = []
        input_batch = batch
        denorm_input_batch = self.dataset.denormalize(input_batch)

        pred = self.forward(batch)
        next = pred # save first prediction
        
        for i in range(lead_iter-1):
            batch = self.dataset.normalize_next_batch(pred, batch)
            pred = gradient_checkpoint.checkpoint(self.forward, batch, use_reentrant=False)

            denorm_pred = self.dataset.denormalize(pred, batch)

            # renormalize with state from input_batch and save to preds_future
            denorm_pred['state_level'] = denorm_input_batch['state_level']
            denorm_pred['state_surface'] = denorm_input_batch['state_surface']
            renorm_pred = self.dataset.normalize(denorm_pred)

            preds_future.append(renorm_pred)

        future = dict(future_state_level=torch.stack([state_future['next_state_level']
                                                      for state_future in preds_future], dim=1),
                    future_state_surface=torch.stack([state_future['next_state_surface']
                                                      for state_future in preds_future], dim=1))
        return next, future
            
    def mylog(self, dct={}, **kwargs):
        #print(mode, kwargs)
        mode = 'train_' if self.training else 'val_'
        dct.update(kwargs)
        for k, v in dct.items():
            self.log(mode+k, v, prog_bar=True, sync_dist=True, add_dataloader_idx=True)
            
    def loss(self, pred, batch, prefix='next_state', multistep=False, **kwargs):
        device = batch['next_state_level'].device
        mse_surface = (pred[prefix+'_surface'] - batch[prefix+'_surface']).abs().pow(self.pow)
        surface_coeffs = pangu_surface_coeffs if not self.use_graphcast_coeffs else graphcast_surface_coeffs
        mse_surface = mse_surface.mul(self.area_weights.to(device)) # latitude coeffs
        mse_surface_w = mse_surface.mul(surface_coeffs.to(device))
    
        mse_level = (pred[prefix+'_level'] - batch[prefix+'_level']).pow(self.pow)
        mse_level = mse_level.mul(self.area_weights.to(device))
        mse_level_w = mse_level.mul(level_coeffs.to(device))
    
        nvar_level = mse_level_w.shape[-4]
        nvar_surface = surface_coeffs.sum().item()

        if multistep:
            lead_iter = int((batch['lead_time_hours'][0]//24).item())
            future_coeffs = torch.tensor([1/i**2 for i in range(2, lead_iter + 1)]).to(device)[None, :, None, None, None, None]
            mse_surface_w = mse_surface_w.mul(future_coeffs)
            mse_level_w = mse_level_w.mul(future_coeffs)

        
        # coeffs are for number of variables
        loss = (4*mse_surface_w.mean() + nvar_level*mse_level_w.mean())/(nvar_level + nvar_surface)
        
        return mse_surface, mse_level, loss
        

    def training_step(self, batch, batch_nb):

        if not 'future_state_level' in batch:
            # standard prediction 
            pred = self.forward(batch)
            _, _, loss = self.loss(pred, batch, prefix='next_state')
            self.mylog(loss=loss)

            denorm_pred = self.dataset.denormalize(pred, batch)
            denorm_batch = self.dataset.denormalize(batch)

            metrics = headline_wrmse(denorm_pred, denorm_batch, prefix='next_state')
            metrics_mean = {k:v.mean(0) for k, v in metrics.items()} # average on time and pred delta

            self.mylog(**metrics_mean)

        else:
            # multistep prediction
            next, future = self.forward_multistep(batch)
            _, _, next_loss = self.loss(next, batch, prefix='next_state')
            _, _, future_loss = self.loss(future, batch, prefix='future_state', multistep=True)


            lead_iter = int((batch['lead_time_hours'][0]//24).item())
            self.mylog(lead_iter=lead_iter)

            loss = (next_loss + (lead_iter - 1)*future_loss)/lead_iter
            self.mylog(future_loss=future_loss)
            self.mylog(next_loss=next_loss)
            self.mylog(loss=loss)

            # log metrics for next
            denorm_pred = self.dataset.denormalize(next, batch)
            denorm_batch = self.dataset.denormalize(batch)

            metrics = headline_wrmse(denorm_pred, denorm_batch, prefix='next_state')
            metrics_mean = {k:v.mean(0) for k, v in metrics.items()}
            self.mylog(**metrics_mean)

            #log some metrics for second step only
            denorm_pred = self.dataset.denormalize(future, batch)
            denorm_batch = self.dataset.denormalize(batch)

            metrics = headline_wrmse(denorm_pred, denorm_batch, prefix='future_state')
            metrics_mean = {k:v.mean(0)[0] for k, v in metrics.items()} # select the 48h lead time prediction with [0]
            self.mylog(**metrics_mean)
            
        return loss
        
        
    def validation_step(self, batch, batch_nb):
        print('current epoch', self.current_epoch)
        pred = self.forward(batch)
        _, _, loss = self.loss(pred, batch)
        self.mylog(loss=loss)
        # denorm and compute metrics

        denorm_pred = self.dataset.denormalize(pred, batch)
        denorm_batch = self.dataset.denormalize(batch)

        metrics = headline_wrmse(denorm_pred, denorm_batch, prefix='next_state')
        metrics_mean = {k:v.mean(0) for k, v in metrics.items()} # average on time and pred delta

        self.mylog(**metrics_mean)
        return loss
    
    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)
    
    def on_train_epoch_start(self, outputs=None):
        if self.dataset.multistep > 1:
            # increase multistep every 2 epochs
            self.dataset.multistep = 2 + self.current_epoch // self.increase_multistep_period


    def configure_optimizers(self):
        print('configure optimizers')
        decay_params = {k:True for k, v in self.named_parameters() if 'weight' in k and not 'norm' in k}
        opt = torch.optim.AdamW([{'params': [v for k, v in self.named_parameters() if k in decay_params]},
                                 {'params': [v for k, v in self.named_parameters() if k not in decay_params], 'weight_decay': 0}],
                                lr=self.lr, 
                                betas=self.betas, 
                                weight_decay=self.weight_decay)
        sched = diffusers.optimization.get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
            num_cycles=self.num_cycles)
        sched = { 'scheduler': sched,
                      'interval': 'step', # or 'epoch'
                      'frequency': 1}
        return [opt], [sched]

    
class ForecastModuleWithCond(ForecastModule):
    '''
    module that can take additional information:
    - month and hour
    - previous state
    - pred state (e.g. prediction of other weather model)
    '''
    def __init__(self, *args, cond_dim=32, use_pred=False, use_prev=False, **kwargs):
        from backbones.dit import TimestepEmbedder
        super().__init__(*args, **kwargs)
        # cond_dim should be given as arg to the backbone
        self.month_embedder = TimestepEmbedder(cond_dim)
        self.hour_embedder = TimestepEmbedder(cond_dim)
        self.use_pred = use_pred
        self.use_prev = use_prev

    def forward(self, batch):
        device = batch['state_surface'].device
        input_surface = torch.cat([batch['state_surface'], 
                                   batch['state_constant']], dim=1)
        input_level = batch['state_level']
        if self.use_pred and 'pred_state_surface' in batch:
            input_surface = torch.cat([input_surface, batch['pred_state_surface']], dim=1)
            input_level = torch.cat([input_level, batch['pred_state_level']], dim=1)

        if self.use_prev and 'prev_state_surface' in batch:
            input_surface = torch.cat([input_surface, batch['prev_state_surface']], dim=1)
            input_level = torch.cat([input_level, batch['prev_state_level']], dim=1)
            
        month = torch.tensor([int(x[5:7]) for x in batch['time']]).to(device)
        month_emb = self.month_embedder(month)
        hour = torch.tensor([int(x[-5:-3]) for x in batch['time']]).to(device)
        hour_emb = self.hour_embedder(hour)

        t_emb = month_emb + hour_emb

        input_surface = input_surface.squeeze(-3)
        out = self.backbone(input_surface, input_level, t_emb)
        return out
        
        
