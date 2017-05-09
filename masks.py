'''
Masks file for use in sampler
Phillip Kuznetsov 2017
'''
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.ndimage.filters import gaussian_laplace as laplace_filter

def get_mask(image, mask_type, inverse=True, args={} ):
    masks_map = {
        'random' : make_random_mask,
        'laplace' : make_laplace_mask,
        'square' : make_square_mask,
        '' : lambda x,y,z: None
    }
    mask_args = {
        'random' : ['percent_pix', 'sigma', 'use_laplace'],
        'square' : ['top_left', 'dims'],
        'laplace': ['max_density', 'min_density'],
        '': []
    }
    if mask_type not in masks_map:
        raise ValueError('mask type {} not found'.format(mask_type))
    args = {k : args[k] for k in mask_args[mask_type] if k in args}
    return masks_map[mask_type](image, inverse=inverse, **args)

def make_square_mask(image, top_left=None, dims=None, inverse=False):
    ''' constructs a square mask '''
    image_shape = image.shape
    assert len(image_shape)==4 and image_shape[1] == 3, 'image_shape must be a tuple of length 4, such as (1, 3, 277, 277). Got {}'.format(image_shape)
    assert top_left is None or type(top_left) is tuple and len(top_left)==2, 'top_left must be None or a tuple of length 2'
    assert dims is None or type(dims) is tuple and len(dims)==2, 'dims must be None or a tuple of length 2'
    if top_left is None:
        top_left_x = image_shape[2]/3
        top_left_y = image_shape[3] /3

    if dims is None:
        width = image_shape[2]/3
        height = image_shape[3] /3

    # make a square mask
    zeros = np.zeros(image_shape)
    ones = np.ones(image_shape)
    if inverse:
        mask = zeros
        not_mask = ones
    else:
        mask = ones
        not_mask = zeros
    mask[:,:, top_left_x : width + top_left_x, top_left_y: top_left_y + height]  = not_mask[:,:, top_left_x : width + top_left_x, top_left_y: top_left_y + height]
    return mask

def laplace_process(image, sigma):
    ''' take the laplace filter with binary erosion '''
    im = np.linalg.norm(image,axis=1)
    im = laplace_filter(im, sigma)
    im = im > np.mean(im)
    im = binary_erosion(im)
    return im

def blur(im):
    im =  np.linalg.norm(im[0,:,:,:], axis=0)
    im = scipy.ndimage.filters.gaussian_filter(im,4)
    return im

def sample(im, max_density=0.05,min_density=0.01):
    im = (max_density)*(np.abs(im)/np.max(np.abs(im)))
    return (np.random.rand(*im.shape) < im) + (np.random.rand(*im.shape) < min_density) > 0
def combine_masks(a, b):
    return np.minimum(a, b)
def make_laplace_mask(image, max_density=0.05, min_density=0.01, inverse=False):
    return sample(image, max_density, min_density) * (not inverse)

def make_random_mask(image, sigma=3, percent_pix = 0.03, use_laplace=False, inverse=False):
    ''' Returns a mask from randomly sampled points'''
    image_shape = image.shape
    assert len(image_shape)==4 and image_shape[1] == 3, 'image_shape must be a tuple of length 4, such as (1, 3, 277, 277)'
    im = laplace_process(image, sigma)
    # make the mask
    zeros = np.zeros(image_shape)
    ones = np.ones(image_shape)
    if not inverse:
        mask = zeros
        not_mask = ones
    else:
        mask = ones
        not_mask = zeros
    random_field = np.random.rand(*image_shape[2:]) > 1 - percent_pix
    mask[:,:, random_field]  = not_mask[:,:, random_field]
    if not use_laplace:
        return mask
    else:
        im = laplace_process(image, sigma)
        return (-im + mask) > 0
