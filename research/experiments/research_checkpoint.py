"""Atomic epoch checkpoints with optimizer and deterministic CPU RNG state."""
from pathlib import Path
import numpy as np
import torch


def save_checkpoint(path,net,optimizer,rng,epoch,metadata):
    name,keys,pos,gauss,cached = rng.get_state()
    data = dict(metadata,state_dict=net.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
                torch_rng=torch.get_rng_state(),numpy_rng=[name,keys.tolist(),pos,gauss,cached])
    path = Path(path); temporary = path.with_name(path.name+'.tmp')
    torch.save(data,temporary); temporary.replace(path)


def resume_checkpoint(path,net,optimizer,rng):
    data = torch.load(path,map_location='cpu',weights_only=True)
    net.load_state_dict(data['state_dict']); optimizer.load_state_dict(data['optimizer'])
    torch.set_rng_state(data['torch_rng'])
    name,keys,pos,gauss,cached = data['numpy_rng']
    rng.set_state((name,np.asarray(keys,np.uint32),pos,gauss,cached))
    return int(data['epoch'])
