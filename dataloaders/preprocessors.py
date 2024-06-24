import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np


def default_preprocessor(tensor, pad_128=False):
    #cast
    tensor = tensor.float()
    
    nlat, nlon = tensor.shape[-2:]
    
    # lat first
    if nlat > nlon: 
        tensor = tensor.movedim(-1, -2)
        nlat, nlon = nlon, nlat
    
    tensor = torch.flip(tensor, dims=[-2]) # decreasing latitudes

    # remove pole (south by default)
    tensor = tensor[..., :nlat - nlat%2, :nlon - nlon%2]

    # focus on Europe
    half_lon = nlon // 2
    
    tensor = torch.cat([tensor[..., :, half_lon:], 
                        tensor[..., :, :half_lon]], dim=-1)

    if pad_128:
        tensor = F.pad(tensor, (8, 8, 4, 4))
        
    return tensor

def reverse_default_preprocessor(tensor):
    # first 
    half_lon = tensor.shape[-1] // 2
    tensor = torch.cat([tensor[..., :, half_lon:], 
                    tensor[..., :, :half_lon]], dim=-1)
    # add back south pole
    tensor = torch.cat([tensor, tensor[..., -1:, :]], dim=-2)

    # flip back
    tensor = torch.flip(tensor, dims=[-2])

    return tensor

def lonshift_augmentation(batch, rg=25, prob=0.8):
    shift = np.random.randint(-rg, rg)
    if np.random.rand() < prob:
        batch = {k:(torch.roll(v, shifts=shift, dims=-1) if isinstance(v, torch.Tensor) and (len(v.shape)>1) else v) for k, v in batch.items()}
    return batch

def mask_tensor(t, block_size=20, threshold=0.5):
    if not isinstance(t, torch.Tensor) or len(t.shape) < 2:
        return t
    mask = torch.rand((*t.shape[:-2], t.shape[-2]//block_size, t.shape[-1]//block_size))
    mask = TF.resize(mask, t.shape[-2:], interpolation=0)
    mask = (mask > threshold)
    return t*mask

def random_mask_augmentation(batch):
    thres = np.random.rand() / 2
    batch = {k:(mask_tensor(v, threshold=thres) if not k.startswith('next') else v) for k, v in batch.items()}
