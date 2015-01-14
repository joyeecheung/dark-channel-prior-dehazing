#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image

from util import get_filenames
from dehaze import dehaze


def main():
    for src, dest in get_filenames():
        print 'processing', src + '...'
        im = Image.open(src)
        dark, rawt, refinedt, rawrad, rerad = dehaze(im)
        dark.save(dest % 'dark')
        rawt.save(dest % 'rawt')
        refinedt.save(dest % 'refinedt')
        rawrad.save(dest % 'radiance-rawt')
        rerad.save(dest % 'radiance-refinedt')
        print 'saved', dest


if __name__ == '__main__':
    main()
