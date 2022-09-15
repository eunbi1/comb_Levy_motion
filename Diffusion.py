import torch

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'




class VPSDE:
    def __init__(self, alpha, beta_min=0.1, beta_max=10, T=1.,t_0 = 0.5, device=device):
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.alpha = alpha
        self.t_0 = 0.3
        self.T = T

    def beta(self, t):
        return (self.beta_1 - self.beta_0) * t + self.beta_0

    def marginal_log_mean_coeff2(self, t):
        log_alpha_t = - 1 / (2 * 2) * (t ** 2) * (self.beta_1 - self.beta_0) - 1 / 2 * t * self.beta_0
        return log_alpha_t

    def marginal_log_mean_coeff1(self, t):
        log_alpha_t = - 1 / (2 * self.alpha) * (t ** 2) * (self.beta_1 - self.beta_0) - 1 / self.alpha * t * self.beta_0
        return log_alpha_t

    def diffusion_coeff1(self, t):
        return torch.exp(self.marginal_log_mean_coeff1(t))

    def diffusion_coeff2(self, t):
        return torch.exp(self.marginal_log_mean_coeff2(t))

    def marginal_std2(self, t):
        sigma = torch.pow(1. - torch.exp(2 * self.marginal_log_mean_coeff1(t)), 1 / 2)
        return sigma

    def marginal_std1(self, t):
        sigma = torch.pow(1. - torch.exp(self.alpha * self.marginal_log_mean_coeff2(t)), 1 / self.alpha)
        return sigma

