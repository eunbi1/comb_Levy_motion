import torch
import copy
import time
import numpy as np
import tqdm
from scipy.special import gamma
import torchlevy
from torchlevy import LevyStable
levy = LevyStable()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def gamma_func(x):
    return torch.tensor(gamma(x))






def loss_fn(score_model, sde, x, t1, t2, e_L1, e_L11, e_L2,t_0, batch_size):
    sigma1 = sde.marginal_std1(t1)
    sigma2 = sde.marginal_std2(t2- t_0)

    x_coeff1 = sde.diffusion_coeff1(t1)
    x_coeff2 = sde.diffusion_coeff2(t2 -t_0)

    score1 = -1 / 2 * (e_L1)*torch.pow(sigma1+1e-4 , -1)[:,None,None,None]*sde.beta(t1)[:,None,None,None] # 32
    score2 = levy.score(e_L2, sde.alpha, type='cft').to(device)*torch.pow(sigma2+1e-4, -1)[:,None,None,None] *sde.beta(t2-t_0)[:,None,None,None]


    x_t1 = x_coeff1[:, None, None, None] * x + e_L1 * sigma1[:, None, None, None]
    t_0 = torch.ones(x.shape[0])*t_0

    x_t0 =sde.diffusion_coeff1(t_0)[:,None,None,None]*x + e_L11*sde.marginal_std1(t_0)[:,None,None,None]

    x_t2 = x_coeff2[:,None,None,None]*x_t0 +e_L2*sigma2[:,None,None,None]



    output1 = score_model(x_t1, t1)*sde.beta(t1)[:,None,None,None]
    output2 = score_model(x_t2, t2)*sde.beta(t2-t_0)[:,None,None,None]
    weight1 = (output1 - score1)
    weight2 = (output2 - score2)

    loss = weight1.square().sum(dim=(1, 2, 3)).mean(dim=0)+(weight2).square().sum(dim=(1, 2, 3)).mean(dim=0)

    #print('x_t', torch.min(x_t), torch.max(x_t))
    #print('e_L', torch.min(e_L),torch.max(e_L))
    #print('score', torch.min(score), torch.max(score))
    #print('output', torch.min(model(x_t, t)), torch.max(model(x_t, t)))
    #print('output*beta',torch.min(output), torch.max(output))
    #print('weight', torch.min(weight), torch.max(weight))

    return loss


