"""
Author: Xu Pan. 2021
"""
import numpy as np
from PIL import Image
from matplotlib.pyplot import imshow, figure

def makeGaussian(size, radius=100, sharpness=3, center=None, annular=0, shift=0):
    """ 
    Make a square gaussian kernel.
    size is the length of a side of the square.
    inside radius, mask values are 1.
    outside the radius, there is a gaussian kernal with FWHM=sharpness.
    inside annular, it is reversed gaussian, i.e. decrease to 0. For annular 
    stimuli, i.e. donuts.
    """
    
    # assert radius > (sharpness/2)
    if radius < (sharpness/2):
        return np.zeros((size,size))
        
    assert annular < radius
    
    radius = radius - sharpness/2
    
    if annular > sharpness/2:
        annular = annular - sharpness/2
    
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]+shift
        y0 = center[1]+shift
    
    if sharpness != 0:
        outer_gaussian = np.exp(-4*np.log(2) * (np.clip((x-x0)**2 + (y-y0)**2 - radius**2,0,None)) / sharpness**2)
    else:
        outer_gaussian = np.heaviside(-(x-x0)**2 - (y-y0)**2 + radius**2, 0.5)
    
    if annular>0:
        if sharpness != 0:
            inner_gaussian = np.exp(-4*np.log(2) * (np.clip(-(x-x0)**2 - (y-y0)**2 + annular**2,0,None)) / sharpness**2)
        else:
            inner_gaussian = np.heaviside((x-x0)**2 + (y-y0)**2 - annular**2, 0.5)
        return inner_gaussian * outer_gaussian
    return outer_gaussian

def makeGrating(size, spatialf, ori=0, phase=0, imsize=224, sharpness=3, contrast=1, annular=0, dtype='uint8', shift=0):
    """
    Make a square grating.
    size: the full-width-half-maximum of gaussian mask
    which can be thought of as an effective radius.
    spatialf: spatial frequency.
    ori: orientation, 0 is horizental. 90 is vertical.
    phase: 0-360
    imsize: the image size.
    annular: inside diameter of the donut.
    sharpness: pixels of HMFW of gaussian mask.
    contrast: 0-1.
    """
    ori = ori/180*np.pi
    im = np.ones((imsize,imsize))
    # the last term is to make center phase 0.
    phi = (phase/np.pi*180-2*np.pi/spatialf*imsize/2)
    for x in range(imsize):
        for y in range(imsize):
            im[x,y] = np.sin(2*np.pi/spatialf*((x*np.cos(ori)+y*np.sin(ori))+phi))           
    gaussianmask = makeGaussian(imsize, size, sharpness, annular=annular, shift=shift)
    im = im*gaussianmask*contrast
    im = (im+1) / 2 * 255
    im = np.repeat(im[:,:,np.newaxis],3,axis=2)
    im = im.astype(dtype)
    return im
