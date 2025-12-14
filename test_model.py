import model
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg') # Not sure if necessary but got error about no display access in wsl
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision.models as models
from itertools import islice


def save_image(image, fname):
    img = image.clamp(-1, 1).add(1).div(2).mul(255).byte()  # [-1., 1.] -> [0., 1.] -> [0, 255]
    img = model.make_grid(img)
    img = Image.fromarray(img.permute(1, 2, 0).cpu().numpy())
    img.save(fname)

def test_model(device, diffusion, class_labels, scale):
    denoised_image, denoise_steps = model.sample_denoised(diffusion, device, class_labels, 8, scale)

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

def test_scale(min, max, diffusion, device, class_labels):
    images = []
    for s in range(min, max):
        denoised_image, _ = model.sample_denoised(diffusion, device, class_labels, 8, s)
        images.append(denoised_image)
        save_image(denoised_image, f'./scale_{s}.png')
    return images

def fid_test(device, diffusion, scale, batch_size, num_batches = None):
    dl, info = model.load_dataset_and_make_dataloaders(
        dataset_name='FashionMNIST',
        root_dir='data', # choose the directory to store the data 
        batch_size=batch_size,
        num_workers=0,   # you can use more workers if you see the GPU is waiting for the batches
        pin_memory=gpu,  # use pin memory if you're planning to move the data to GPU
    )

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    validation = dl.valid
    iter = validation if num_batches is None else islice(validation, num_batches)
    for i, batch in enumerate(iter):
        class_labels = batch[1]
        class_labels = class_labels.to(device)
        real = batch[0].to(device)
        real = (real + 1.0)/2.0
        real = real.repeat(1,3,1,1)
        fid.update(real, real=True)
        with torch.no_grad():
            gen_imgs, _ = model.sample_denoised(diffusion, device, class_labels[:batch[0].shape[0]], batch[0].shape[0], scale)
        class_labels = class_labels[batch[0].shape[0] * 2:] # *2 because the validation set doubles that batch size automatically in the dataloader

        gen_imgs = (gen_imgs + 1.0) /2
        gen_imgs = gen_imgs.repeat(1,3,1,1)
        fid.update(gen_imgs, real=False)
        print(f'Batch: {i}')
    return fid.compute()


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

#_= test_scale(1, 15, diffusion, device, class_labels)
#test_model(device, diffusion, class_labels, 3)
fid = fid_test(device, diffusion, 1.5, 32, None)
print(fid)