from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision.utils import make_grid
from torch.amp import autocast, GradScaler
import time
import numpy as np


from data import load_dataset_and_make_dataloaders
from data import DataInfo
if __name__ == '__main__':
    gpu = torch.cuda.is_available()
    device = torch.device('cuda:0' if gpu else 'cpu')
    batch_size = 256
    
    dl, info = load_dataset_and_make_dataloaders(
        dataset_name='FashionMNIST',
        root_dir='data', # choose the directory to store the data 
        batch_size=batch_size,
        num_workers=0,   # you can use more workers if you see the GPU is waiting for the batches
        pin_memory=gpu,  # use pin memory if you're planning to move the data to GPU
    )

def check_for_nans(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"!! NaN detected in Parameter: {name} !!")
            return True

    for name, buf in model.named_buffers():
        if torch.isnan(buf).any():
            print(f"!! NaN detected in Buffer: {name} !!")
            return True
    return False

class Model(nn.Module):
    def __init__(
        self,
        image_channels: int,
        nb_channels: int,
        num_blocks: int,
        cond_channels: int,
        num_classes: int
    ) -> None:
        super().__init__()
        self.noise_emb = NoiseEmbedding(cond_channels)
        self.class_embedding = nn.Embedding(num_classes + 1, nb_channels*2)
        self.conv_in = nn.Conv2d(image_channels, nb_channels, kernel_size=3, padding=1)
        #self.blocks = nn.ModuleList([ResidualBlock(nb_channels, cond_channels) for _ in range(num_blocks)])
        self.downres1 = DownResBlock(nb_channels, nb_channels* 2, cond_channels)
        self.downres2 = DownResBlock(nb_channels * 2, nb_channels * 4, cond_channels)
        self.up1 = UpResBlock(192, nb_channels * 2, (16 , 16), cond_channels)
        self.up2 = UpResBlock(96, nb_channels, (32, 32), cond_channels)
        self.conv_out = nn.Conv2d(nb_channels, image_channels, kernel_size=3, padding=1)
    
    def forward(self, noisy_input: torch.Tensor, c_noise: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        noise_cond = self.noise_emb(c_noise)
        class_cond = self.class_embedding(class_labels)
        x_in = self.conv_in(noisy_input)
        x_1, skip_1 = self.downres1(x_in, noise_cond, class_cond)
        x_2, skip_2 = self.downres2(x_1, noise_cond, class_cond)
        x = self.up1(x_2, skip_2, noise_cond, class_cond)
        x = self.up2(x, skip_1, noise_cond, class_cond)
        return self.conv_out(x)
        


class NoiseEmbedding(nn.Module):
    def __init__(self, cond_channels: int) -> None:
        super().__init__()
        assert cond_channels % 2 == 0
        self.register_buffer('weight', torch.randn(1, cond_channels // 2)) # Random projection matrix
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        with autocast(device_type='cuda', enabled=False):
            assert input.ndim == 1
            f = 2 * torch.pi * input.unsqueeze(1) @ self.weight
            return torch.cat([f.cos(), f.sin()], dim=-1) 


class ResidualBlock(nn.Module):
    def __init__(self, nb_channels_in: int, nb_channels_out: int, cond_channels: int) -> None:
        super().__init__()
        self.noise_mlp = nn.Sequential(nn.Linear(cond_channels, nb_channels_in*4), nn.ReLU(), nn.Linear(nb_channels_in*4, nb_channels_in*4))
        self.noise_proj = nn.Linear(nb_channels_in*4, 2*nb_channels_in)
        default_grps = 32
        if nb_channels_in < default_grps:
            self.norm1 = nn.GroupNorm(4, nb_channels_in)
        else:
            self.norm1 = nn.GroupNorm(default_grps, nb_channels_in)
        self.conv1 = nn.Conv2d(nb_channels_in, nb_channels_out, kernel_size=3, stride=1, padding=1)
        if nb_channels_in < default_grps:
            self.norm2 = nn.GroupNorm(4, nb_channels_out)
        else:
            self.norm2 = nn.GroupNorm(default_grps, nb_channels_out)
        self.conv2 = nn.Conv2d(nb_channels_out, nb_channels_out, kernel_size=3, stride=1, padding=1)
        self.dim = nn.Identity()
        if nb_channels_in != nb_channels_out:
            self.dim = nn.Conv2d(nb_channels_in, nb_channels_out, 1)
    
    def forward(self, x: torch.Tensor, noise_cond: torch.Tensor, class_cond: torch.Tensor) -> torch.Tensor:
        noise_cond += class_cond
        noise_emb = self.noise_mlp(noise_cond)
        noise_proj = self.noise_proj(noise_emb)
        gamma, beta = torch.split(noise_proj, noise_proj.shape[1]//2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        y = gamma * self.norm1(x) + beta
        y = self.conv1(F.relu(y))
        y = self.conv2(F.relu(self.norm2(y)))
        return self.dim(x) + y
    
class DownResBlock(nn.Module):
    def __init__(self, nb_channels_in:int, nb_channels_out: int, cond_channels: int, num_blocks:int = 2, downsample_stride= 2) -> None:
        super().__init__()
        self.block1 = ResidualBlock(nb_channels_in, nb_channels_in, cond_channels)
        self.block2 = ResidualBlock(nb_channels_in, nb_channels_in, cond_channels)
        self.downsample = nn.Conv2d(nb_channels_in, nb_channels_out, kernel_size=3, stride=downsample_stride, padding=1)
    
    def forward(self, x: torch.Tensor, noise_cond: torch.Tensor, class_cond: torch.Tensor):
        x = self.block1(x, noise_cond, class_cond)
        x = self.block2(x, noise_cond, class_cond)
        return self.downsample(x), x

class UpResBlock(nn.Module):
    def __init__(self, nb_channels_in: int, nb_channels_out: int,  out_size: Tuple[int, int], cond_channels: int, num_blocks: int = 2):
        super().__init__()
        self.block1 = ResidualBlock(nb_channels_in, nb_channels_out, cond_channels)
        self.block2 = ResidualBlock(nb_channels_out, nb_channels_out, cond_channels)
        self.out_size = out_size
    def forward(self, x: torch.Tensor, skip: torch.Tensor, noise_cond: torch.Tensor, class_cond: torch.Tensor):
        x = F.interpolate(x, self.out_size)
        x = torch.cat((x, skip), dim=1)
        x = self.block1(x, noise_cond, class_cond)
        x = self.block2(x, noise_cond, class_cond)
        return x
        
        
def sigma_in(sigma, sigma_data=0.6627):
    return 1 / torch.sqrt(sigma_data ** 2 + sigma ** 2)

def sigma_out(sigma, sigma_data = 0.6627):
    return sigma * sigma_data / torch.sqrt(sigma**2 + sigma_data ** 2)

def sigma_skip(sigma, sigma_data = 0.6627): # Magic number from the data, fix later
    return sigma_data ** 2 / (sigma_data**2 + sigma ** 2)

def sigma_noise(sigma):
    return torch.log(sigma)/4

def sample_sigma(n, loc=-1.2, scale=1.2, sigma_min=2e-3, sigma_max=80): # p_noise
    return (torch.randn(n) * scale + loc).exp().clip(sigma_min, sigma_max)

def build_sigma_schedule(steps, rho=7, sigma_min=2e-3, sigma_max=80):
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + torch.linspace(0, 1, steps) * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas

def drop_class_labels(class_labels: torch.Tensor, rate:float):
    mask = torch.from_numpy(np.random.choice([True, False], class_labels.size(0), p=[1-rate, rate]))
    return class_labels.where(mask, 10)

def train_model(loader: Dataset, info: DataInfo, model : nn.Module, nb_epochs):
    opt = torch.optim.Adam(model.parameters(), 2e-4)
    scaler = GradScaler()
    losses = []
    nans = False
    for e in range(nb_epochs):
        if e == nb_epochs / 2: #Save one checkpoint for now
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            torch.save({
                'epoch' : e,
                'model_state_dict' : model.state_dict(),
                'optimizier_state_dict' : opt.state_dict(),
                'loss': loss
            }, f'./checkpoints/checkpoint_{timestamp}.tar')
        crit = nn.MSELoss()
        batch_count = 0
        loss = 0
        for load in loader:
            x = load[0].to(device)
            class_labels = load[1]
            class_labels = drop_class_labels(class_labels, 0.1)
            class_labels = class_labels.to(device)
            opt.zero_grad(set_to_none=True)

            sigma = sample_sigma(x.shape[0]).to(device)
            noise = torch.normal(mean=torch.zeros(x.shape), std=torch.ones(x.shape)).to(device)
            noisy = x + noise * sigma.view(x.shape[0], 1, 1, 1)

            c_in = sigma_in(sigma, info.sigma_data).to(device).view(x.shape[0], 1, 1, 1)
            c_out = sigma_out(sigma, info.sigma_data).to(device).view(x.shape[0], 1, 1, 1)
            c_skip = sigma_skip(sigma, info.sigma_data).to(device).view(x.shape[0], 1, 1, 1)
            c_noise = sigma_noise(sigma).to(device)
            with autocast(str(device)):
                res = model.forward(c_in * noisy, c_noise, class_labels)
                loss = crit(res, (x - c_skip * noisy) / c_out)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            if batch_count % 50 == 0:
                if check_for_nans(model):
                    nans = True
                    print('Stopped training due to nans')
                    break
            batch_count += 1
        if nans:
            break
        losses.append(loss)
    return losses

def denoise(x, sigma, model, device, class_labels):
    return sigma_skip(sigma) * x + sigma_out(sigma) * model.forward(sigma_in(sigma) * x, sigma_noise(sigma).expand(x.shape[0]).to(device), class_labels)

def sample_denoised(model, device, class_labels, batch_size, guidance_scale):
    sigmas = build_sigma_schedule(1000)
    return euler_sampling(sigmas, model, device, class_labels, batch_size, guidance_scale)


def euler_sampling(sigmas, model, device, class_labels, batch_size, guidance_scale):
    denoise_steps = []
    x = torch.randn(batch_size, 1, 32, 32) * sigmas[0]  # Initialize with pure gaussian noise ~ N(0, sigmas[0])

    class_labels = torch.cat((class_labels, torch.tensor([10] * batch_size).to(device)))
    x = x.to(device)

    sigmas = sigmas.to(device)
    for i, sigma in enumerate(sigmas):
        batch = x.repeat(2,1,1,1)
        with torch.no_grad():
            x_denoised = denoise(batch, sigma, model, device, class_labels)  
            # Where D(x, sigma) = cskip(sigma) * x + cout(sigma) * F(cin(sigma) * x, cnoise(sigma)) 
            # and F(.,.) is your neural network
        cond, uncond = x_denoised.split(batch_size, dim=0)
        pred = uncond + guidance_scale * (cond - uncond)
        sigma_next = sigmas[i + 1] if i < len(sigmas) - 1 else 0
        d = (x - pred) / sigma
        
        x = x + d * (sigma_next - sigma)  # Perform one step of Euler's method
        denoise_steps.append(x_denoised.cpu())
    return x, denoise_steps

if __name__ == '__main__': # To allow multiple worker threads in data-loading
    model = Model(info.image_channels, 32, 2, 64, 10)
    #saved = torch.load("./checkpoints/checkpoint_20251209_193930.tar")
    #model.load_state_dict(saved['model_state_dict'])
    model = model.to(device)
    input = dl[0]
    losses = train_model(input, info, model, 500)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), f'./diffusion_model{timestamp}.pth')
    np.save('Loss_history_epoch.npy', np.array(losses))
