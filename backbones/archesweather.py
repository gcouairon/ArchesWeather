import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import GELU as GeLU
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.utils.checkpoint as gradient_checkpoint

from .archesweather_layers import BasicLayer, DownSample, UpSample, PatchRecovery2D, PatchRecovery3D, PatchEmbed2D, PatchEmbed3D


class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x

class PatchRecovery5(nn.Module):
    ''' true upsampling with 3D conv
    '''
    def __init__(self, 
                 input_dim=None,
                 dim=192,
                 downfactor=4,
                 hidden_dim=96,
                 output_dim=69,
                 n_level_variables=5):
        # input dim equals input_dim*z since we will be flattening stuff ?
        super().__init__()
        self.downfactor = downfactor
        if input_dim is None:
            input_dim = 8*dim
        
        self.input_conv = nn.Conv2d(input_dim, 14*hidden_dim, kernel_size=1, stride=1, padding=0)
        self.interp = Interpolate(scale_factor=2, mode="bilinear", align_corners=True)

        self.head = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1), # kernel size 3 for interactions and smoothing
            nn.GELU(),
        )
        if downfactor == 4:
            self.head2 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=hidden_dim, eps=1e-6, affine=True),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1), # kernel size 3 for interactions and smoothing
            nn.GELU())
        
        self.proj_surface = nn.Conv2d(hidden_dim, 4, kernel_size=1, stride=1, padding=0)
        self.proj_level = nn.Conv3d(hidden_dim, n_level_variables, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        # recover enough levels
        bs = x.shape[0]
        x = x.flatten(1, 2) # put levels in the channel dim
        x = self.input_conv(x)
        x = x.reshape((bs, 14, -1, *x.shape[-2:])).flatten(0, 1) # put levels back
        x = self.interp(x)
        x = x.reshape(bs, 14, -1, *x.shape[-2:]).movedim(1, 2)
        x = self.head(x)
        if self.downfactor == 4:
            x = x.reshape((bs, 14, -1, *x.shape[-2:])).flatten(0, 1) # put levels back
            x = self.interp(x)
            x = x.reshape(bs, 14, -1, *x.shape[-2:]).movedim(1, 2)
            x = self.head2(x)

        output_surface = self.proj_surface(x[:, :, 0])
        output_level = self.proj_level(x[:, :, 1:])

        return output_level, output_surface.unsqueeze(-3)


class LinVert(nn.Module):
    '''
    a modification of Mlp that takes the full column
    '''
    def __init__(self, in_features,  
                 drop=0., **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(8*in_features, 8*in_features)

    def forward(self, x: torch.Tensor):
        shortcut = x
        x2 = shortcut.reshape((shortcut.shape[0], 8, -1, shortcut.shape[-1])).movedim(1, -2).flatten(-2, -1) # B, lat*lon, 8*C
        x2 = self.fc1(x2)
        x2 = x2.reshape((x2.shape[0], -1, 8, shortcut.shape[-1])).movedim(-2, 1).flatten(1, 2) # B, 8*lat*lon, C

        return shortcut + x2
    
# conditional basic layer

class CondBasicLayer(BasicLayer):
    def __init__(self, *args, dim=192, cond_dim=32, **kwargs):
        super().__init__(*args, dim=dim, **kwargs)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * dim, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        # init the modulation

    def forward(self, x, cond_emb=None):
        c = self.adaLN_modulation(cond_emb)
        return super().forward(x, c)
    
class ArchesWeatherCond(nn.Module):
    def __init__(self, 
                 lon_resolution=240,
                 lat_resolution=120,
                 emb_dim=192, 
                 cond_dim=32, # dim of the conditioning
                 two_poles=False,
                 num_heads=(6, 12, 12, 6), 
                 droppath_coeff=0.2,
                 patch_size=(2, 4, 4),
                 window_size=(2, 6, 12), 
                 depth_multiplier=1,
                 position_embs_dim=0,
                 surface_ch=7,
                 level_ch=5,
                 n_level_variables=5,
                 use_prev=False, 
                 use_skip=False,
                 conv_head=False,
                 freeze_backbone=False,
                 dropout=0.0,
                 first_interaction_layer=False,
                 gradient_checkpointing=False,
                **kwargs):
        super().__init__()
        self.__dict__.update(locals())
        drop_path = np.linspace(0, droppath_coeff/depth_multiplier, 8*depth_multiplier).tolist()
        # In addition, three constant masks(the topography mask, land-sea mask and soil type mask)
        self.zdim = 8 if patch_size[0]==2 else 5 # 5 for patch size 4 in z dim
        self.layer1_shape = (self.lat_resolution//self.patch_size[1], self.lon_resolution//self.patch_size[2])
        
        self.layer2_shape = (self.layer1_shape[0]//2, self.layer1_shape[1]//2)
        
        self.positional_embeddings = nn.Parameter(torch.zeros((position_embs_dim, lat_resolution, lon_resolution)))
        torch.nn.init.trunc_normal_(self.positional_embeddings, 0.02)
        
        
        self.patchembed2d = PatchEmbed2D(
            img_size=(lat_resolution, lon_resolution),
            patch_size=patch_size[1:],
            in_chans=surface_ch + position_embs_dim,  # add
            embed_dim=emb_dim,
        )
        self.patchembed3d = PatchEmbed3D(
            img_size=(13, lat_resolution, lon_resolution),
            patch_size=patch_size,
            in_chans=level_ch,
            embed_dim=emb_dim
        )

        if first_interaction_layer == 'linear':
            self.interaction_layer = LinVert(in_features=emb_dim)

        act_layer1 = act_layer2 = act_layer4 = nn.GELU

        self.layer1 = CondBasicLayer(
            dim=emb_dim,
            cond_dim=cond_dim,
            input_resolution=(self.zdim, *self.layer1_shape),
            depth=2*depth_multiplier,
            num_heads=num_heads[0],
            window_size=window_size,
            drop_path=drop_path[:2*depth_multiplier],
            act_layer=act_layer1,
            drop=dropout,
            **kwargs
        )
        self.downsample = DownSample(in_dim=emb_dim, input_resolution=(self.zdim, *self.layer1_shape), output_resolution=(self.zdim, *self.layer2_shape))
        self.layer2 = CondBasicLayer(
            dim=emb_dim * 2,
            cond_dim=cond_dim,
            input_resolution=(self.zdim, *self.layer2_shape),
            depth=6*depth_multiplier,
            num_heads=num_heads[1],
            window_size=window_size,
            drop_path=drop_path[2*depth_multiplier:],
            act_layer=act_layer2,
            drop=dropout,
            **kwargs
        )
        self.layer3 = CondBasicLayer(
            dim=emb_dim * 2,
            cond_dim=cond_dim,
            input_resolution=(self.zdim, *self.layer2_shape),
            depth=6*depth_multiplier,
            num_heads=num_heads[2],
            window_size=window_size,
            drop_path=drop_path[2*depth_multiplier:],
            act_layer=act_layer2,
            drop=dropout,
            **kwargs
        )
        self.upsample = UpSample(emb_dim * 2, emb_dim, (self.zdim, *self.layer2_shape), (self.zdim, *self.layer1_shape))
        out_dim = emb_dim if not self.use_skip else 2*emb_dim
        self.layer4 = CondBasicLayer(
            dim=out_dim,
            cond_dim=cond_dim,
            input_resolution=(self.zdim, *self.layer1_shape),
            depth=2*depth_multiplier,
            num_heads=num_heads[3],
            window_size=window_size,
            drop_path=drop_path[:2*depth_multiplier],
            act_layer=act_layer4,
            drop=dropout,
            **kwargs
        )
        # The outputs of the 2nd encoder layer and the 7th decoder layer are concatenated along the channel dimension.
        if self.freeze_backbone:
            for p in self.parameters():
                p.requires_grad = False
                
        if not self.conv_head:
            self.patchrecovery2d = PatchRecovery2D((lat_resolution, lon_resolution), patch_size[1:], out_dim, 4)
            self.patchrecovery3d = PatchRecovery3D((13, lat_resolution, lon_resolution), patch_size, out_dim, n_level_variables)

            for p in self.patchembed2d.parameters():
                p.requires_grad = True
            for p in self.patchembed3d.parameters():
                p.requires_grad = True

        if conv_head:
            self.patchrecovery = PatchRecovery5(input_dim=self.zdim*out_dim, output_dim=69, downfactor=patch_size[-1],
                                                n_level_variables=n_level_variables)
            for p in self.patchrecovery.parameters():
                p.requires_grad = True

    def forward(self, input_surface, input_level, cond_emb, niter=1, **kwargs):
        """
        Args:
            surface (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=4.
            surface_mask (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=3.
            upper_air (torch.Tensor): 3D n_pl=13, n_lat=721, n_lon=1440, chans=5.
        """
        lat_res, lon_res = self.lat_resolution, self.lon_resolution
        init_shape = (input_surface.shape[-2], input_surface.shape[-1])
        
        surface = input_surface
        upper_air = input_level


        pos_embs = self.positional_embeddings[None].expand((surface.shape[0], *self.positional_embeddings.shape))
        
        surface = torch.concat([surface, pos_embs], dim=1)
        surface = self.patchembed2d(surface)
        upper_air = self.patchembed3d(upper_air)

        x = torch.concat([surface.unsqueeze(2), upper_air], dim=2)
        B, C, Pl, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)

        if self.first_interaction_layer:
            x = self.interaction_layer(x)


        x = self.layer1(x, cond_emb)

        skip = x
        x = self.downsample(x)
        
        x = self.layer2(x, cond_emb)

        if self.gradient_checkpointing:
            x = gradient_checkpoint.checkpoint(self.layer3, x, cond_emb, use_reentrant=False)
        else:
            x = self.layer3(x, cond_emb)

        x = self.upsample(x)
        if self.use_skip and skip is not None:
            x = torch.concat([x, skip], dim=-1)
        x = self.layer4(x, cond_emb)

        output = x
        output = output.transpose(1, 2).reshape(output.shape[0], -1, 8, *self.layer1_shape)
        
        if self.freeze_backbone:
            output = output.detach()
            
        if not self.conv_head:
            output_surface = output[:, :, 0, :, :]
            output_upper_air = output[:, :, 1:, :, :]

            output_surface = self.patchrecovery2d(output_surface)
            output_level = self.patchrecovery3d(output_upper_air)
            
            output_surface = output_surface.unsqueeze(-3)
            
        else:

            output_level, output_surface = self.patchrecovery(output)
            
            
        out = dict(next_state_level=output_level, 
                    next_state_surface=output_surface)
        return out

