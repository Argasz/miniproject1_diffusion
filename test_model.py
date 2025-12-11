import model
import torch
from PIL import Image


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
test_model(device, diffusion, class_labels, 3)