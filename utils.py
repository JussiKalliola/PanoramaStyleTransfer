import torch
from PIL import Image
from scipy import ndimage
from torch.autograd import Variable
from torchvision import transforms
import numpy as np

# New perspectives for each image in cubemap
from perspective2perspectives.pers2pers import equir2pers, pers2equir

# Parameters for the perspective transforms
perspective_params = {
    '1': {
        '4': {'theta': 90, 'phi': -90, 'rotation': 0},
        '2': {'theta': 90, 'phi': 90, 'rotation': 0},
        '3': {'theta': 90, 'phi': 0, 'rotation': 0},
        '6': {'theta': -90, 'phi': 0, 'rotation': 0},
    },
    '2': {
        '6': {'theta': 180, 'phi': 90, 'rotation': 0},
        '3': {'theta': 0, 'phi': -90, 'rotation': 0},
        '5': {'theta': 90, 'phi': 0, 'rotation': -90},
        '1': {'theta': -90, 'phi': 0, 'rotation': 90},
    },
    '3': {
        '4': {'theta': 0, 'phi': -90, 'rotation': 0},
        '2': {'theta': 0, 'phi': 90, 'rotation': 0},
        '5': {'theta': 90, 'phi': 0, 'rotation': 0},
        '1': {'theta': -90, 'phi': 0, 'rotation': 0},
    },
    '4': {
        '6': {'theta': 180, 'phi': -90, 'rotation': 0},
        '3': {'theta': 0, 'phi': 90, 'rotation': 0},
        '5': {'theta': 90, 'phi': 0, 'rotation': 90},
        '1': {'theta': -90, 'phi': 0, 'rotation': -90},
    },
    '5': {
        '4': {'theta': -90, 'phi': -90, 'rotation': 0},
        '2': {'theta': -90, 'phi': 90, 'rotation': 0},
        '6': {'theta': 90, 'phi': 0, 'rotation': 0},
        '3': {'theta': -90, 'phi': 0, 'rotation': 0},
    },
    '6': {
        '4': {'theta': 0, 'phi': -90, 'rotation': 180},
        '2': {'theta': 0, 'phi': 90, 'rotation': 180},
        '1': {'theta': 90, 'phi': 0, 'rotation': 0},
        '5': {'theta': -90, 'phi': 0, 'rotation': 0},
    }
}

# Cubemap params: equi => cubemap or cubemap => equi
cube_params = {
    '3': { 'theta': 0, 'phi': 0 },
    '5': { 'theta': 90, 'phi': 0 },
    '6': { 'theta': 180, 'phi': 0 },
    '1': { 'theta': -90, 'phi': 0 },
    '2': { 'theta': 0, 'phi': 90 },
    '4': { 'theta': 0, 'phi': -90 }
}

# opens and returns image file as a PIL image (0-255)
def load_image(filename):
    img = Image.open(filename)
    return img

# assumes data comes in batch form (ch, h, w)
def save_image(filename, data):
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img = data.clone().numpy()
    img = ((img * std + mean).transpose(1, 2, 0)*255.0).clip(0, 255).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)

def transform_image_to_original(data):
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img = data.clone().numpy()
    img = ((img * std + mean).transpose(1, 2, 0)*255.0).clip(0, 255).astype("uint8")
    #img = Image.fromarray(img)
    return img

def get_perspective(images):
    im1, im1_id = images[0]
    im2, im2_id = images[1]

    im1_id=str(im1_id)
    im2_id=str(im2_id)

    height, width, _ = im1.shape

    eq, eq_mask = pers2equir(im1, height=480, width=960)
    new_perspective = equir2pers(eq, FOV=120, theta=perspective_params[im1_id][im2_id]['theta'],
                                 phi=perspective_params[im1_id][im2_id]['phi'], height=height, width=width)
    new_mask = equir2pers(eq_mask, FOV=120, theta=perspective_params[im1_id][im2_id]['theta'],
                                 phi=perspective_params[im1_id][im2_id]['phi'], height=height, width=width)

    if perspective_params[im1_id][im2_id]['rotation'] != 0:
        new_perspective = ndimage.rotate(new_perspective, perspective_params[im1_id][im2_id]['rotation'])
        new_mask = ndimage.rotate(new_mask, perspective_params[im1_id][im2_id]['rotation'])

    return new_perspective, new_mask

def cubemap_to_equirectangular(cubemap):
    width=1500
    height=750
    eq_im = np.zeros((height,width, 3))

    for key in cubemap.keys():
        eq, _ = pers2equir(im=cubemap[key], FOV=120, theta=cube_params[key]['theta'], phi=cube_params[key]['phi'], width=width, height=height)
        for y in range(height):
            for x in range(width):
                if np.sum(eq_im[y,x,:]) > 0 and np.sum(eq[y,x,:]) > 0:
                    eq_im[y, x, :]=np.mean(np.array([eq_im[y,x,:], eq[y,x,:]]), axis=0)
                else:
                    eq_im[y, x, :] += eq[y, x, :]
        #indices=np.nonzero(eq)
        #eq_im[indices]=eq[indices]
    eq_im=(255*(eq_im - np.min(eq_im))/np.ptp(eq_im)).astype(np.uint8)

    return eq_im

# Calculate Gram matrix (G = FF^T)
def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G

# using ImageNet values
def normalize_tensor_transform():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])