from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    # 根据随机生成的 时间步 t 的索引，提取计算好的数值，shape 为 [batch_size, ]
    out = torch.gather(v, index=t, dim=0).float().to(device)
    # shape 为 [batch_size, 1, 1, 1] ，对齐与图片的维度，为图片的每个数值乘上系数。
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T
        # 注册缓冲区张量
        # tensor([0.0001, 0.0001, 0.0001, ..., 0.0199, 0.0199, 0.0200])
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        # 1- tensor([0.0001, 0.0001, 0.0001, ..., 0.0199, 0.0199, 0.0200])
        alphas = 1. - self.betas
        # 计算 alphas_bar ：累乘的结果
        alphas_bar = torch.cumprod(alphas, dim=0)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        # 累乘的平方根 为 x_0 的系数
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        # （1 - 累乘）的平方根 为 噪声的系数
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0):
        """
        Algorithm 1.
        """
        # 生成 [0, 999] 的整数张量，shape为[batch_size, ]
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        # 生成标准正态分布的早生，形状跟图像大小一样
        noise = torch.randn_like(x_0)
        
        # 根据加噪过程生成x_t, t是随机的
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
    
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')

        # print("loss shape is : ",loss.shape)

        return loss


class GaussianDDPMSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T
        # 注册缓冲变量
        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        # 计算 1 - betas
        alphas = 1. - self.betas
        # 计算累乘
        alphas_bar = torch.cumprod(alphas, dim=0)
        # 计算累乘的前一个数值，在累乘第一个位置添加一个1， 变成对应索引位置的a_t-1 的系数，同时保证1000个元素
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer('coeff1', torch.sqrt(1. / alphas))

        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))

        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.coeff1, t, x_t.shape) * x_t -
            extract(self.coeff2, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations

        # var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        # var = extract(var, t, x_t.shape)

        var = extract(self.posterior_var, t, x_t.shape)

        eps = self.model(x_t, t)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

        return xt_prev_mean, var

    def forward(self, x_T):
        x_t = x_T
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var= self.p_mean_variance(x_t=x_t, t=t)
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)   
     

class GaussianDDIMSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()
        self.model = model
        self.T = T
        # generate T steps of beta
        beta_t = torch.linspace(beta_1, beta_T, T, dtype=torch.float32)
        # calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
        alpha_t = 1.0 - beta_t
        self.register_buffer("alpha_t_bar", torch.cumprod(alpha_t, dim=0))

    def forward(self, x_t, steps = 50, eta=0.0):

        a = self.T // steps
        time_steps = np.asarray(list(range(0, self.T, a)))

        #time_steps = time_steps + 1
        time_steps_prev = np.concatenate([[0], time_steps[:-1]])

        for i in reversed(range(0, steps)):

            t = torch.full((x_t.shape[0],), int(time_steps[i]), device=x_t.device, dtype=torch.long)
            prev_t = torch.full((x_t.shape[0],), int(time_steps_prev[i]), device=x_t.device, dtype=torch.long)

            alpha_t = extract(self.alpha_t_bar, t, x_t.shape)
            alpha_t_prev = extract(self.alpha_t_bar, prev_t, x_t.shape)

            epsilon_theta_t = self.model(x_t, t)

            sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
            epsilon_t = torch.randn_like(x_t)
            x_t = (
                    torch.sqrt(alpha_t_prev / alpha_t) * x_t +
                    (torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) - torch.sqrt(
                        (alpha_t_prev * (1 - alpha_t)) / alpha_t)) * epsilon_theta_t +
                    sigma_t * epsilon_t
            )
        return torch.clip(x_t, -1.0, 1.0) 