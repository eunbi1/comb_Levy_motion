import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7"

import random
import torch.backends.cudnn as cudnn
import torch
import numpy as np


from sampling import *
from training import *
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

import random
import torch.backends.cudnn as cudnn

torch.manual_seed(4)
torch.cuda.manual_seed(4)
torch.cuda.manual_seed_all(4)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

for epoch in torch.arange(49,50):
  samples =sample(alpha=1.8, path='/scratch/private/eunbiyoon/comb_Levy_motion/CIFAR10beta_batch64lr0.0001ch128ch_mult[1, 2, 2, 2]num_res2dropout0.1clamp51.8_0.1_20/ckpt/CIFAR10batch64ch128ch_mult[1, 2, 2, 2]num_res2dropout0.1clamp5_epoch27_1.8_0.1_20.pth',
                   beta_min=0.1, beta_max=20, sampler='pc_sampler2', batch_size=64, num_steps=1000, LM_steps=5,
               Predictor=True, Corrector=False, trajectory=True, clamp=5, initial_clamp=3,
               clamp_mode="constant", x_0=False, ch=128, ch_mult=[1,2,2,2], num_res_blocks=2,
               datasets="CIFAR10", name=str(str(f'2epoch{epoch}')), dir_path='/scratch/private/eunbiyoon/comb_Levy_motion')

