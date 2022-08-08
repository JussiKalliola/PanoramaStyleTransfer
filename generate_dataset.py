#import perspective_transform.pers2pers as pt
import os
from pathlib import Path
import cv2
from scipy import ndimage


from perspective2perspectives.pers2pers import *
from tqdm import tqdm


def read_image(img_path):
    im = cv2.imread(img_path, cv2.IMREAD_COLOR)
    im_name = Path(img_path).stem

    return im, im_name

def save_cubemaps(cubemap, save_path):
    Path(save_path).mkdir(parents=True, exist_ok=True)
    for key in cubemap.keys():
        if not isinstance(cubemap[key], dict):
            cv2.imwrite(os.path.join(save_path, f'{key}.png'), cubemap[key])
        else:
            for key2 in cubemap[key].keys():
                cv2.imwrite(os.path.join(save_path, f'{key}_{key2}.png'), cubemap[key][key2])

    pass


def convert_to_cubemap(im_path, width=600, height=600, FOV=120):
    I = cv2.imread(im_path, cv2.IMREAD_COLOR)
    equ = E2P.Equirectangular(I)

    cube_params = {
        '3': {
            'theta': 0,
            'phi': 0
        },
        '5': {
            'theta': 90,
            'phi': 0
        },
        '6': {
            'theta': 180,
            'phi': 0
        },
        '1': {
            'theta': -90,
            'phi': 0
        },
        '2': {
            'theta': 0,
            'phi': 90
        },
        '4': {
            'theta': 0,
            'phi': -90
        }
    }

    cubes={}

    # Compute projection and save cube images
    for key in cube_params.keys():
        theta = cube_params[key]['theta']
        phi = cube_params[key]['phi']

        cubes[key]= equ.GetPerspective(FOV=FOV, THETA=theta, PHI=phi, height=height, width=width)

    return cubes

def get_new_perspectives_and_masks(cubemap, height=600, width=600):
    new_perspectives={
        '1':{},
        '2': {},
        '3': {},
        '4': {},
        '5': {},
        '6': {},
    }
    masks={
        '1': {},
        '2': {},
        '3': {},
        '4': {},
        '5': {},
        '6': {},
    }

    # New perspectives for each image in cubemap
    perspective_params={
        '1': {
            '4': {'theta': 90, 'phi': -90, 'rotation': 0},
            '2': {'theta': 90, 'phi': 90, 'rotation': 0},
            '3': {'theta': 90, 'phi': 0, 'rotation': 0},
            '6': {'theta': -90, 'phi': 0, 'rotation': 0},
        },
        '2': {
            '6': {'theta':180, 'phi': 90, 'rotation':0},
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

    for key in cubemap.keys():
        eq, eq_mask = pers2equir(cubemap[key], height=480, width=960)
        for key2 in perspective_params[key]:
            new_perspective = equir2pers(eq, FOV=120, theta=perspective_params[key][key2]['theta'],
                                         phi=perspective_params[key][key2]['phi'], height=height, width=width)
            mask = equir2pers(eq_mask, FOV=120, theta=perspective_params[key][key2]['theta'],
                              phi=perspective_params[key][key2]['phi'], height=height, width=width)

            if perspective_params[key][key2]['rotation'] != 0:
                new_perspectives[key][key2] = ndimage.rotate(new_perspective, perspective_params[key][key2]['rotation'])
                masks[key][key2] = ndimage.rotate(mask, perspective_params[key][key2]['rotation'])
            else:
                new_perspectives[key][key2] = new_perspective
                masks[key][key2] = mask



    return new_perspectives, masks


# Generate cubemaps from equirectangular images. Cubemaps are saved in location ./360_dataset/cubemaps/
if __name__ == '__main__':
    # define paths
    dataset_base_path = os.path.join('360_dataset')
    equirec_path = os.path.join(dataset_base_path, 'images')
    cubemap_path = os.path.join(dataset_base_path, 'cubemaps')
    Path(cubemap_path).mkdir(parents=True, exist_ok=True)

    # Get all filenames in the dataset (equirectangular)
    equirec_filenames = [f for f in os.listdir(equirec_path) if os.path.isfile(os.path.join(equirec_path, f))]

    for im_filename in tqdm(equirec_filenames):
        # Read equirec image
        im_path=os.path.join(equirec_path, im_filename)
        im, im_name = read_image(im_path)

        # Create cubemap folder if it doesnt exists
        cubemap_save_path=os.path.join(cubemap_path, im_name)

        # Returns cubes in the following order: front, right, back, left, top, bottom
        cubemap=convert_to_cubemap(im_path=im_path, width=300, height=300, FOV=120)
        save_cubemaps(cubemap=cubemap, save_path=cubemap_save_path)
        new_perspectives, masks = get_new_perspectives_and_masks(cubemap)

        masks_save_path = os.path.join(cubemap_save_path, 'masks')
        save_cubemaps(cubemap=masks, save_path=masks_save_path)

        #new_perspective_save_path = os.path.join(cubemap_save_path, 'projections')
        #save_cubemaps(cubemap=new_perspectives, save_path=new_perspective_save_path)


    pass