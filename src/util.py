#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from PIL import Image

SRC_DIR = 'img'
IMG_NAMES = ('canon7.jpg', 'cones.jpg', 'IMG_8763.jpg', 'IMG_8766.jpg',
             'ny12_photo.jpg', 'ny17_photo.jpg', 'pumpkins.jpg',
             'stadium1.jpg', 'toys.jpg', 'yellowmountain.jpg')


def get_filenames():
    """Return a named tuple of filenames(absolute filepath)."""
    file_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir, _ = os.path.split(file_dir)
    src_path = os.path.join(parent_dir, SRC_DIR)
    filenames = map(lambda name: os.path.join(src_path, name),
                    IMG_NAMES)
    return filenames


def img_to_array(img, dtype=None):
    """Convert an image to an 2-d array.

    Parameters
    ----------
    img: the input image
    dtype: the data type of the output array, if not specified,
           it will use the smallest data type that can hold all the data
           without overflow or underflow.

    Return
    ----------
    If the mode of the image is RGB, then three 2-d arrays will be returned
    """
    if img.mode == 'RGB':
        # convert each channel to a numpy array
        return [img_to_array(ch) for ch in img.split()]
    else:
        return np.array(img.getdata(), dtype=dtype).reshape(img.size[::-1])


def array_to_img(data, mode=None):
    """Convert a 2-d numpy array to an image.

    Parameters
    ----------
    img: the input image
    mode:
    """
    if not mode:
        return Image.fromarray(data)
    elif mode == 'RGB':
        channels = [array_to_img(ch, 'L') for ch in data]
        return Image.merge('RGB', channels)
    else:
        return Image.fromarray(data).convert(mode)
