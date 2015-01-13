#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
from itertools import combinations_with_replacement
from collections import defaultdict
from numpy.linalg import inv

from util import get_filenames

R, G, B = 0, 1, 2


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
    padded = np.pad(data, ((w / 2, w / 2), (w / 2, w / 2), (0, 0)), 'edge')
    darkch = np.zeros((M, N))
    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(padded[i:i + w, j:j + w, :])
    return darkch


def get_atmosphere(data, darkch, p):
    # check 4.4
    M, N = darkch.shape
    flatdata = data.reshape(M * N, 3)
    flatdark = darkch.ravel()
    searchidx = (-flatdark).argsort()[:M * N * p]  # find top m*n*p indexes
    print 'atmosphere light region:', [(i/N, i%N) for i in searchidx]
    return np.max(flatdata.take(searchidx, axis=0), axis=0)


def get_transmission(data, atmosphere, darkch, omega, w):
    # equation 12
    newdata = data / atmosphere
    return 1 - omega * get_dark_channel(newdata, w)


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
    dest[:r + 1] = sumY[r: 2 * r + 1]
    dest[r + 1:M - r] = sumY[2 * r + 1:] - sumY[:M - 2 * r - 1]
    dest[-r:] = np.tile(sumY[-1], (r, 1)) - sumY[M - 2 * r - 1:M - r - 1]

    sumX = np.cumsum(dest, axis=1)
    dest[:, :r + 1] = sumX[:, r:2 * r + 1]
    dest[:, r + 1:N - r] = sumX[:, 2 * r + 1:] - sumX[:, :N - 2 * r - 1]
    dest[:, -r:] = np.tile(sumX[:, -1][:, None], (1, r)) - \
        sumX[:, N - 2 * r - 1:N - r - 1]
    return dest


def guided_filter(I, p, r=40, eps=1e-3):
    M, N = p.shape
    base = boxfilter(np.ones((M, N)), r)

    means = [boxfilter(I[:, :, i], r) / base for i in xrange(3)]
    mean_p = boxfilter(p, r) / base
    means_IP = [boxfilter(I[:, :, i] * p, r) / base for i in xrange(3)]
    covIP = [means_IP[i] - means[i] * mean_p for i in xrange(3)]

    var = defaultdict(dict)
    for i, j in combinations_with_replacement(range(3), 2):
        var[i][j] = boxfilter(
            I[:, :, i] * I[:, :, j], r) / base - means[i] * means[j]

    a = np.zeros((M, N, 3))
    for y, x in np.ndindex(M, N):
        Sigma = np.array([[var[R][R][y, x], var[R][G][y, x], var[R][B][y, x]],
                          [var[R][G][y, x], var[G][G][y, x], var[G][B][y, x]],
                          [var[R][B][y, x], var[G][B][y, x], var[B][B][y, x]]])
        cov = np.array([c[y, x] for c in covIP])
        a[y, x] = np.dot(cov, inv(Sigma + eps * np.eye(3)))  # eq 14

    b = mean_p - a[:, :, R] * means[R] - \
        a[:, :, G] * means[G] - a[:, :, B] * means[B]

    # eq 16
    q = (boxfilter(a[:, :, R], r) * I[:, :, R] + boxfilter(a[:, :, G], r) *
         I[:, :, G] + boxfilter(a[:, :, B], r) * I[:, :, B] + boxfilter(b, r)) / base

    return q


def get_radiance(I, tmin=0.1, Amax=220, w=15, p=0.0001,
                 omega=0.95, guided=False):
    # equation 16
    m, n, _ = I.shape
    Idark = get_dark_channel(I, w)
    A = get_atmosphere(I, Idark, p)
    oldt = t = get_transmission(I, A, Idark, omega, w)
    print 'transmission between', t.min(), t.max()
    print 'atmosphere', A
    if guided:
        t = guided_filter(I, t)
    newt = np.maximum(t, tmin)
    tiledt = np.zeros_like(I)
    tiledt[:, :, R] = tiledt[:, :, G] = tiledt[:, :, B] = newt
    return Idark, oldt, newt, (I - A) / tiledt + A


def dehaze(im, guided=True):
    data = np.asarray(im, dtype=np.float64)
    darkch, oldt, t, radiance = get_radiance(data, guided=guided)
    white = np.full_like(darkch, 255)

    def to_img(raw):
        # cut = np.maximum(np.minimum(raw, 255), 0)
        print raw.max(), raw.min()
        if len(raw.shape) == 3:
            r, g, b = [raw[:, :, ch] for ch in xrange(3)]
            rm, gm, bm = [Image.fromarray(ch).convert('L') for ch in (r, g, b)]
            return Image.merge('RGB', (rm, gm, bm))
        else:
            return Image.fromarray(raw).convert('L')

    return [to_img(raw) for raw in (darkch, white * oldt, white * t, radiance)]


if __name__ == '__main__':
    for src, dest in get_filenames():
        print 'processing', src
        im = Image.open(src)
        dark, oldt, t, radiance = dehaze(im)
        dark.save(dest % 'dark')
        oldt.save(dest % 'oldt')
        t.save(dest % 't')
        radiance.save(dest % 'radiance')
        print 'saved', dest
