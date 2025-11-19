import torch
import torch.nn as nn
import torch.nn.functional as F


from data import load_dataset_and_make_dataloaders
from data import DataInfo

gpu = torch.cuda.is_available()
device = torch.device('cuda:0' if gpu else 'cpu')
 
dl, info = load_dataset_and_make_dataloaders(
    dataset_name='FashionMNIST',
    root_dir='data', # choose the directory to store the data 
    batch_size=32,
    num_workers=0,   # you can use more workers if you see the GPU is waiting for the batches
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
            x = block(x)
        return self.conv_out(x)


class NoiseEmbedding(nn.Module):
    def __init__(self, cond_channels: int) -> None:
        super().__init__()
        assert cond_channels % 2 == 0
        self.register_buffer('weight', torch.randn(1, cond_channels // 2))
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 1
        f = 2 * torch.pi * input.unsqueeze(1) @ self.weight
        return torch.cat([f.cos(), f.sin()], dim=-1)


class ResidualBlock(nn.Module): # Skip
    def __init__(self, nb_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(nb_channels)
        self.conv1 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(nb_channels)
        self.conv2 = nn.Conv2d(nb_channels, nb_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.relu(self.norm1(x)))
        y = self.conv2(F.relu(self.norm2(y)))
        return x + y

def sigma_in(sigma, sigma_data):
    return 1 / torch.sqrt(sigma_data ** 2 + sigma ** 2)

def sigma_out(sigma, sigma_data):
    return sigma * sigma_data / torch.sqrt(sigma**2 + sigma_data ** 2)

def sigma_skip(sigma, sigma_data):
    return sigma_data ** 2 / sigma_data**2 + sigma ** 2

def sigma_noise(sigma):
    return torch.log(sigma)/4

def sample_sigma(n, loc=-1.2, scale=1.2, sigma_min=2e-3, sigma_max=80): # p_noise
    return (torch.randn(n) * scale + loc).exp().clip(sigma_min, sigma_max)

def build_sigma_schedule(steps, rho=7, sigma_min=2e-3, sigma_max=80):
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + torch.linspace(0, 1, steps) * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas

def train_model(loader, info: DataInfo, model : nn.Module, nb_epochs):
    for _ in range(nb_epochs):
        for x, y in loader:
            sigma = sample_sigma(32)
            noisy = x + torch.normal(0, info.sigma_data)
            c_in = sigma_in(sigma, info.sigma_data)
            c_out = sigma_out(sigma, info.sigma_data)
            c_skip = sigma_skip(sigma, info.sigma_data)
            c_noise = sigma_noise(sigma, info.sigma_data)
            res = model.forward(noisy, c_noise)
            l = nn.MSELoss()

           

def euler_sampling(sigmas, D):
    x = torch.randn(8, 1, 32, 32) * sigmas[0]  # Initialize with pure gaussian noise ~ N(0, sigmas[0])

    for i, sigma in enumerate(sigmas):
        
        with torch.no_grad():
            x_denoised = D(x, sigma)  
            # Where D(x, sigma) = cskip(sigma) * x + cout(sigma) * F(cin(sigma) * x, cnoise(sigma)) 
            # and F(.,.) is your neural network
        
        sigma_next = sigmas[i + 1] if i < len(sigmas) - 1 else 0
        d = (x - x_denoised) / sigma
        
        x = x + d * (sigma_next - sigma)  # Perform one step of Euler's method
    return x