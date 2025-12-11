import model
import torch
from PIL import Image


def save_image(image, fname):
    img = image.clamp(-1, 1).add(1).div(2).mul(255).byte()  # [-1., 1.] -> [0., 1.] -> [0, 255]
    img = model.make_grid(img)
    img = Image.fromarray(img.permute(1, 2, 0).cpu().numpy())
    img.save(fname)
LABEL_LOOKUP = {
    'T-shirt':0,
    'Trousers':1,
    'Pullover': 2,
    'Dress': 3,
    'Coat': 4,
    'Sandal':5,
    'Shirt':6,
    'Sneaker':7,
    'Bag':8,
    'Ankle boot':9
}
gpu = torch.cuda.is_available()
device = torch.device('cuda:0' if gpu else 'cpu')

saved = torch.load('./diffusion_model20251210_210536.pth')
diffusion = model.Model(1, 32, 2, 64, 10).to(device)
diffusion.load_state_dict(saved)
class_labels = torch.tensor([0,1,2,3,4,5,6,7]).to(device)
denoised_image, denoise_steps = model.sample_denoised(diffusion, device, class_labels, 8)

# dl, info = model.load_dataset_and_make_dataloaders(
#         dataset_name='FashionMNIST',
#         root_dir='data', # choose the directory to store the data 
#         batch_size=1,
#         num_workers=0,   # you can use more workers if you see the GPU is waiting for the batches
#         pin_memory=gpu,  # use pin memory if you're planning to move the data to GPU
# )

# img = next(iter(dl.train))[0].to(device)
# sigma = model.sample_sigma(img.shape[0]).to(device)
# noise = torch.normal(mean=torch.zeros(img.shape), std=torch.ones(img.shape)).to(device)
# noisy = img + noise * sigma.view(img.shape[0], 1, 1, 1)
# img_denoised = model.denoise(noisy, sigma, diffusion, device)
# save_image(img, './original.png')
# save_image(noisy, './noisy.png')
# save_image(img_denoised, './denoised.png')
vis = []
for step in denoise_steps:
    img = step.clamp(-1, 1).add(1).div(2).mul(255).byte()  # [-1., 1.] -> [0., 1.] -> [0, 255]
    img = model.make_grid(img)
    vis.append(img)

for i in range(len(vis)):
    if(i % 100 == 0):
        img = Image.fromarray(vis[i].permute(1, 2, 0).cpu().numpy())
        img.save(f'./test{i}.png')

img = denoised_image.clamp(-1, 1).add(1).div(2).mul(255).byte()  # [-1., 1.] -> [0., 1.] -> [0, 255]
img = model.make_grid(img)
img = Image.fromarray(img.permute(1, 2, 0).cpu().numpy())
img.save('./test.png')