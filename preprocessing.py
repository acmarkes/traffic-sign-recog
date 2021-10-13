from skimage import exposure
from skimage.morphology import disk
from skimage.morphology import ball
from skimage.filters import rank
from skimage.color import rgb2gray
import numpy as np

#PREPROCESSING

def grayscaling(img):
    #turns a rgb image to grayscale
    #input: np.array of an image
    #output: np.array of a processed image
    
    img = rgb2gray(img)
    return img

def local_histogram_equalizer(img):
    #does local histogram equalization to enhance image contrast
    #input: np.array of an image
    #output: np.array of a processed image
    
    footprint = disk(15.5)
    img = rank.equalize(img, selem=footprint)
    return img

def normalize(img):
    #normalizes pixel values
    #input: np.array of an image
    #output: np.array of a processed image
        
    img = np.divide(img, 255)    
    return img

#%%
def preprocessor(imgs, debug=False):
    #sequence of processing steps for the CNN
    #input: array of np.arrays of images
    #output: array of np.arrays of processed images

    if debug:
        print(f'orig shape: {np.array(imgs).shape}')

    gray_imgs = list(map(grayscaling, imgs))

    if debug:        
        print(f'gray shape: {np.array(gray_imgs).shape}')

    eq_imgs = list(map(local_histogram_equalizer, gray_imgs))
    
    if debug:
        print(f'eq shape: {np.array(eq_imgs).shape}')

    norm_imgs = np.array(list(map(normalize, eq_imgs)))
    
    if debug:
        print(f'final shape: {np.array(norm_imgs).shape}')
    norm_imgs = norm_imgs[..., None]

    return norm_imgs
