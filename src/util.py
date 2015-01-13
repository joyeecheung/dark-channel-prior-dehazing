#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from PIL import Image

SRC_DIR = 'img'
DEST_DIR = 'result'
IMG_NAMES = ('canon7.jpg', 'cones.jpg', 'IMG_8763.jpg', 'IMG_8766.jpg',
             'ny12_photo.jpg', 'ny17_photo.jpg', 'pumpkins.jpg',
             'stadium1.jpg', 'toys.jpg', 'yellowmountain.jpg')


def get_filenames():
    """Return a named tuple of filenames(absolute filepath)."""
    file_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir, _ = os.path.split(file_dir)
    src_path = os.path.join(parent_dir, SRC_DIR)
    dest_path = os.path.join(parent_dir, DEST_DIR)
    filenames = zip((os.path.join(src_path, name) for name in IMG_NAMES),
                    (os.path.join(dest_path, name) for name in IMG_NAMES))
    return filenames
