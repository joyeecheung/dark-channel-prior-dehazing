#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image

def get_dark_channel(data, w):
    """Get dark channels from image data (RGB images only).

    Parameters
    -----------
    data: a M * N * 3 array containing data in the image where
          M is the height, N is the width, 3 represents R/G/B channels
    w: window size

    Return
    -----------
    A 2-d array of data in the dark channel.
    """
    # from equation 5
    M, N, _ = data.shape
    padded = np.pad(data, ((w/2, w/2), (w/2, w/2), (0, 0)), 'edge')
    darkch = np.zeros((M, N))
    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(padded[i:i + w - 1, j:j + w - 1, :])
    return darkch


def get_atmosphere(data, darkch, p):
    # check 4.4
    m, n = darkch.shape
    flatdata = data.reshape(m * n, 3)
    flatdark = darkch.ravel()
    searchidx = (-flatdark).argsort()[:m * n * p]  # find top m*n*p indexes
    return np.mean(flatdata.take(searchidx, axis=0), axis=0)


def get_transmission(data, atmosphere, darkch, omega, w):
    # equation 12
    newdata = data / atmosphere
    return 1 - omega * get_dark_channel(newdata, w)


def get_radiance(I, tmin=0.1, Amax=220, w=15, p=0.001, omega=0.95):
    # equation 16
    I = I.astype(np.float64)
    m, n, _ = I.shape
    Idark = get_dark_channel(I, w)
    A = get_atmosphere(I, Idark, p)
    t = get_transmission(I, A, Idark, omega, w)
    t = np.repeat(t, 3).reshape(m, n, 3)
    return (I - A)/t + A


def dehaze_naive(im):
    radiance = get_radiance(np.asarray(im))
    radiance = np.maximum(np.minimum(radiance, 255), 0)
    return Image.fromarray(radiance.astype(np.uint8))


def boxfilter(data, r):
    """
    Parameters
    ----------
    data: a single channel/gray image with data normalized to [0,1]
    r: window radius
    """
    M, N = data.shape
    dest = np.zeros((M, N))

    sumY = np.cumsum(data, axis=0)
    dest[:r+1] = sumY[r: 2*r+1]
    dest[r+1:M-r] = sumY[2*r+1:] - sumY[:M-2*r-1]
    dest[-r:] = np.tile(sumY[-1], (r, 1)) - sumY[M-2*r-1:M-r-1]

    sumX = np.cumsum(dest, axis=1)
    dest[:, :r+1] = sumX[:, r:2*r+1]
    dest[:, r+1:N-r] = sumX[:, 2*r+1:] - sumX[:, :N-2*r-1]
    dest[:, -r:] = np.tile(sumX[:, -1][:, None], (1, r)) - sumX[:, N-2*r-1:N-r-1]
    return dest

