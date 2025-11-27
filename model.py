import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision.utils import make_grid
import time


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
        num_workers=6,   # you can use more workers if you see the GPU is waiting for the batches
        pin_memory=gpu,  # use pin memory if you're planning to move the data to GPU
    )

class Model(nn.Module):
    def __init__(
        self,
        image_channels: int,
        nb_channels: int,
        num_blocks: int,
        cond_channels: int,
    ) -> None:
        super().__init__()
        self.noise_emb = NoiseEmbedding(cond_channels)
        self.conv_in = nn.Conv2d(image_channels, nb_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ResidualBlock(nb_channels) for _ in range(num_blocks)])
        self.conv_out = nn.Conv2d(nb_channels, image_channels, kernel_size=3, padding=1)
    
    def forward(self, noisy_input: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        cond = self.noise_emb(c_noise) # TODO: not used yet
        x = self.conv_in(noisy_input)
        for block in self.blocks:
            x = block(x, cond)
        return self.conv_out(x)
        


class NoiseEmbedding(nn.Module):
    def __init__(self, cond_channels: int) -> None:
        super().__init__()
        assert cond_channels % 2 == 0
        self.register_buffer('weight', torch.randn(1, cond_channels // 2))
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 1
        f = 2 * torch.pi * input.unsqueeze(1) @ self.weight
        return torch.cat([f.cos(), f.sin()], dim=-1) # Creating sinusoidal signature for each noise level


class ResidualBlock(nn.Module): # Skip
    def __init__(self, nb_channels: int) -> None:
        super().__init__()
        self.noise_level_projection = nn.Linear(2, nb_channels)
        self.norm1 = nn.BatchNorm2d(nb_channels)
        self.conv1 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(nb_channels)
        self.conv2 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x: torch.Tensor, noise_cond: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.relu(self.norm1(x)))
        noise_proj = self.noise_level_projection(noise_cond).unsqueeze(-1).unsqueeze(-1)
        y = y + noise_proj
        y = self.conv2(F.relu(self.norm2(y)))
        return x + y

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

def train_model(loader: Dataset, info: DataInfo, model : nn.Module, nb_epochs):
    opt = torch.optim.Adam(model.parameters())
    for e in range(nb_epochs):
        if e == nb_epochs / 2: #Save one checkpoint for now
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            torch.save({
                'epoch' : e,
                'model_state_dict' : model.state_dict(),
                'optimizier_state_dict' : opt.state_dict(),
                'loss': l
            }, f'./checkpoints/checkpoint_{timestamp}.tar')
        crit = nn.MSELoss()
        for load in loader:
            x = load[0]
            x = x.to(device)
            opt.zero_grad()

            sigma = sample_sigma(x.shape[0]).to(device)
            noise = torch.normal(mean=torch.zeros(x.shape), std=torch.ones(x.shape)).to(device)
            noisy = x + noise * sigma.view(x.shape[0], 1, 1, 1)

            c_in = sigma_in(sigma, info.sigma_data).to(device).view(x.shape[0], 1, 1, 1)
            c_out = sigma_out(sigma, info.sigma_data).to(device).view(x.shape[0], 1, 1, 1)
            c_skip = sigma_skip(sigma, info.sigma_data).to(device).view(x.shape[0], 1, 1, 1)
            c_noise = sigma_noise(sigma).to(device)
            res = model.forward(c_in * noisy, c_noise)
            l = crit(res, (x - c_skip * noisy) / c_out)
            l.backward()
            opt.step()

def denoise(x, sigma, model, device):
    return sigma_skip(sigma) * x + sigma_out(sigma) * model.forward(sigma_in(sigma) * x, sigma_noise(sigma).unsqueeze(0).to(device))

def sample_denoised(model, device):
    sigmas = build_sigma_schedule(1000)
    return euler_sampling(sigmas, model, device)


def euler_sampling(sigmas, model, device): # TODO: This overflows
    denoise_steps = []
    x = torch.randn(8, 1, 32, 32) * sigmas[0]  # Initialize with pure gaussian noise ~ N(0, sigmas[0])
    x = x.to(device)
    for i, sigma in enumerate(sigmas):
        
        with torch.no_grad():
            x_denoised = denoise(x, sigma, model, device)  
            # Where D(x, sigma) = cskip(sigma) * x + cout(sigma) * F(cin(sigma) * x, cnoise(sigma)) 
            # and F(.,.) is your neural network
        sigma_next = sigmas[i + 1] if i < len(sigmas) - 1 else 0
        d = (x - x_denoised) / sigma
        
        x = x + d * (sigma_next - sigma)  # Perform one step of Euler's method
        denoise_steps.append(x_denoised.cpu())
    return x, denoise_steps

if __name__ == '__main__': # To allow multiple worker threads in data-loading
    model = Model(info.image_channels, 32, 1, 2)
    model = model.to(device)
    input = dl[0]
    train_model(input, info, model, 1000)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), f'./diffusion_model{timestamp}.pth')
