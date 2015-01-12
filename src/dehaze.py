#!/usr/bin/env python
# -*- coding: utf-8 -*-


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
    return np.max(flatdata.take(searchidx, axis=0), axis=0)
