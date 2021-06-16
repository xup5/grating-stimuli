"""
Author: Xu Pan. 2021
"""
import numpy as np
from PIL import Image
from matplotlib.pyplot import imshow, figure

def makeGaussian(size, radius=100, sharpness=5, center=None):
    """ 
    Make a square gaussian kernel.
    size is the length of a side of the square.
    inside radius, mask values are 1.
    outside the radius, there is a gaussian kernal with FWHM=sharpness.
    """
    
    assert radius>(sharpness/2)
    radius = radius - sharpness/2
    
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]
    
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]
    if sharpness != 0:
        return np.exp(-4*np.log(2) * (np.clip((x-x0)**2 + (y-y0)**2 - radius**2,0,None)) / sharpness**2)
    else:
        return np.heaviside(-(x-x0)**2 - (y-y0)**2 + radius**2, 0.5)

def makeGrating(size, spatialf, ori=0, phase=0, imsize=224, sharpness=5, dtype='uint8'):
    """
    Make a square grating.
    size: the full-width-half-maximum of gaussian mask
    which can be thought of as an effective radius.
    spatialf: spatial frequency.
    ori: orientation, 0 is horizental. 90 is vertical.
    phase: 0-360
    imsize: the image size.
    """
    ori = ori/180*np.pi
    im = np.ones((imsize,imsize))
    # the last term is to make center phase 0.
    phi = (phase/np.pi*180-2*np.pi/spatialf*imsize/2)
    for x in range(imsize):
        for y in range(imsize):
            im[x,y] = np.sin(2*np.pi/spatialf*((x*np.cos(ori)+y*np.sin(ori))+phi))           
    gaussianmask = makeGaussian(imsize, size, sharpness)
    im = im*gaussianmask
    im = (im+1) / 2 * 255
    im = np.repeat(im[:,:,np.newaxis],3,axis=2)
    im = im.astype(dtype)
    return im