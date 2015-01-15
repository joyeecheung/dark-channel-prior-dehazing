#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implementation for Single Image Haze Removal Using Dark Channel Prior.

Reference:
http://research.microsoft.com/en-us/um/people/kahe/cvpr09/
http://research.microsoft.com/en-us/um/people/kahe/eccv10/
"""

import numpy as np
from PIL import Image

from guidedfilter import guided_filter

R, G, B = 0, 1, 2  # index for convenience
L = 256  # color depth


def get_dark_channel(I, w):
    """Get the dark channel prior in the (RGB) image data.

    Parameters
    -----------
    I:  an M * N * 3 numpy array containing data ([0, L-1]) in the image where
        M is the height, N is the width, 3 represents R/G/B channels.
    w:  window size

    Return
    -----------
    An M * N array for the dark channel prior ([0, L-1]).
    """
    M, N, _ = I.shape
    padded = np.pad(I, ((w / 2, w / 2), (w / 2, w / 2), (0, 0)), 'edge')
    darkch = np.zeros((M, N))
    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(padded[i:i + w, j:j + w, :])  # CVPR09, eq.5
    return darkch


def get_atmosphere(I, darkch, p):
    """Get the atmosphere light in the (RGB) image data.

    Parameters
    -----------
    I:      the M * N * 3 RGB image data ([0, L-1]) as numpy array
    darkch: the dark channel prior of the image as an M * N numpy array
    p:      percentage of pixels for estimating the atmosphere light

    Return
    -----------
    A 3-element array containing atmosphere light ([0, L-1]) for each channel
    """
    # reference CVPR09, 4.4
    M, N = darkch.shape
    flatI = I.reshape(M * N, 3)
    flatdark = darkch.ravel()
    searchidx = (-flatdark).argsort()[:M * N * p]  # find top M * N * p indexes
    print 'atmosphere light region:', [(i / N, i % N) for i in searchidx]

    # return the highest intensity for each channel
    return np.max(flatI.take(searchidx, axis=0), axis=0)


def get_transmission(I, A, darkch, omega, w):
    """Get the transmission esitmate in the (RGB) image data.

    Parameters
    -----------
    I:       the M * N * 3 RGB image data ([0, L-1]) as numpy array
    A:       a 3-element array containing atmosphere light
             ([0, L-1]) for each channel
    darkch:  the dark channel prior of the image as an M * N numpy array
    omega:   bias for the estimate
    w:       window size for the estimate

    Return
    -----------
    An M * N array containing the transmission rate ([0.0, 1.0])
    """
    return 1 - omega * get_dark_channel(I / A, w)  # CVPR09, eq.12


def dehaze_raw(I, tmin=0.2, Amax=220, w=15, p=0.0001,
               omega=0.95, guided=True, r=40, eps=1e-3):
    """Get the dark channel prior, atmosphere light, transmission rate
       and refined transmission rate for raw RGB image data.

    Parameters
    -----------
    I:      M * N * 3 data as numpy array for the hazy image
    tmin:   threshold of transmission rate
    Amax:   threshold of atmosphere light
    w:      window size of the dark channel prior
    p:      percentage of pixels for estimating the atmosphere light
    omega:  bias for the transmission estimate

    guided: whether to use the guided filter to fine the image
    r:      the radius of the guidance
    eps:    epsilon for the guided filter

    Return
    -----------
    (Idark, A, rawt, refinedt) if guided=False, then rawt == refinedt
    """
    m, n, _ = I.shape
    Idark = get_dark_channel(I, w)

    A = get_atmosphere(I, Idark, p)
    A = np.minimum(A, Amax)  # threshold A
    print 'atmosphere', A

    rawt = get_transmission(I, A, Idark, omega, w)
    print 'raw transmission rate',
    print 'between [%.4f, %.4f]' % (rawt.min(), rawt.max())

    rawt = refinedt = np.maximum(rawt, tmin)  # threshold t
    if guided:
        normI = (I - I.min()) / (I.max() - I.min())  # normalize I
        refinedt = guided_filter(normI, refinedt, r, eps)

    print 'refined transmission rate',
    print 'between [%.4f, %.4f]' % (refinedt.min(), refinedt.max())

    return Idark, A, rawt, refinedt


def get_radiance(I, A, t):
    """Recover the radiance from raw image data with atmosphere light
       and transmission rate estimate.

    Parameters
    ----------
    I:      M * N * 3 data as numpy array for the hazy image
    A:      a 3-element array containing atmosphere light
            ([0, L-1]) for each channel
    t:      estimate fothe transmission rate

    Return
    ----------
    M * N * 3 numpy array for the recovered radiance
    """
    tiledt = np.zeros_like(I)  # tiled to M * N * 3
    tiledt[:, :, R] = tiledt[:, :, G] = tiledt[:, :, B] = t
    return (I - A) / tiledt + A  # CVPR09, eq.16


def dehaze(im, tmin=0.2, Amax=220, w=15, p=0.0001,
           omega=0.95, guided=True, r=40, eps=1e-3):
    """Dehaze the given RGB image.

    Parameters
    ----------
    im:     the Image object of the RGB image
    guided: refine the dehazing with guided filter or not
    other parameters are the same as `dehaze_raw`

    Return
    ----------
    (dark, rawt, refinedt, rawrad, rerad)
    Images for dark channel prior, raw transmission estimate,
    refiend transmission estimate, recovered radiance with raw t,
    recovered radiance with refined t.
    """
    I = np.asarray(im, dtype=np.float64)
    Idark, A, rawt, refinedt = dehaze_raw(I, tmin, Amax, w, p,
                                          omega, guided, r, eps)
    white = np.full_like(Idark, L - 1)

    def to_img(raw):
        # threshold to [0, L-1]
        cut = np.maximum(np.minimum(raw, L - 1), 0).astype(np.uint8)

        if len(raw.shape) == 3:
            print 'Range for each channel:'
            for ch in xrange(3):
                print '[%.2f, %.2f]' % (raw[:, :, ch].max(), raw[:, :, ch].min())
            return Image.fromarray(cut)
        else:
            return Image.fromarray(cut)

    return [to_img(raw) for raw in (Idark, white * rawt, white * refinedt,
                                    get_radiance(I, A, rawt),
                                    get_radiance(I, A, refinedt))]
