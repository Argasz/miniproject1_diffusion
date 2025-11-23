import model
import torch
from PIL import Image
import matplotlib.pyplot as plt

gpu = torch.cuda.is_available()
device = torch.device('cuda:0' if gpu else 'cpu')

saved = torch.load('./diffusion_model.pth')
diffusion = model.Model(1, 32, 1, 2).to(device)
diffusion.load_state_dict(saved)
denoised_image, denoise_steps = model.sample_denoised(diffusion, device)

vis = []
for step in denoise_steps:
    img = step.clamp(-1, 1).add(1).div(2).mul(255).byte()  # [-1., 1.] -> [0., 1.] -> [0, 255]
    img = model.make_grid(img)
    vis.append(img)

for i in range(len(vis)):
    img = Image.fromarray(vis[i].permute(1, 2, 0).cpu().numpy())
    img.save(f'./test{i}.jpg')

# img = denoised_image.clamp(-1, 1).add(1).div(2).mul(255).byte()  # [-1., 1.] -> [0., 1.] -> [0, 255]
# img = model.make_grid(img)
# img = Image.fromarray(img.permute(1, 2, 0).cpu().numpy())
# img.save('./test.jpg')