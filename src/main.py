#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from functools import partial

from PIL import Image

from util import get_filenames
from dehaze import dehaze

SP_IDX = (5, 6, 8, 12)  # for testing parameters
SP_PARAMS = ({'tmin': 0.2, 'Amax': 170, 'w': 15, 'r': 40},
             {'tmin': 0.5, 'Amax': 190, 'w': 15, 'r': 40},
             {'tmin': 0.5, 'Amax': 220, 'w': 15, 'r': 40})


def generate_results(src, dest, generator):
    print 'processing', src + '...'
    im = Image.open(src)
    dark, rawt, refinedt, rawrad, rerad = generator(im)
    dark.save(dest % 'dark')
    rawt.save(dest % 'rawt')
    refinedt.save(dest % 'refinedt')
    rawrad.save(dest % 'radiance-rawt')
    rerad.save(dest % 'radiance-refinedt')
    print 'saved', dest


def main():
    filenames = get_filenames()
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=int,
                        choices=range(len(filenames)),
                        help="index for single input image")
    parser.add_argument("-t", "--tmin", type=float, default=0.2,
                        help="minimum transmission rate")
    parser.add_argument("-A", "--Amax", type=int, default=220,
                        help="maximum atmosphere light")
    parser.add_argument("-w", "--window", type=int, default=15,
                        help="window size of dark channel")
    parser.add_argument("-r", "--radius", type=int, default=40,
                        help="radius of guided filter")

    args = parser.parse_args()

    if args.input is not None:
        src, dest = filenames[args.input]
        dest = dest.replace("%s",
                     "%s-%d-%d-%d-%d" % ("%s", args.tmin * 100, args.Amax,
                                           args.window, args.radius))
        generate_results(src, dest, partial(dehaze, tmin=args.tmin, Amax=args.Amax,
                                           w=args.window, r=args.radius))
    else:
        for idx in SP_IDX:
            src, dest = filenames[idx]
            for param in SP_PARAMS:
                newdest = dest.replace("%s",
                     "%s-%d-%d-%d-%d" % ("%s", param['tmin'] * 100,
                                         param['Amax'], param['w'],
                                         param['r']))
                generate_results(src, newdest, partial(dehaze, **param))

        for src, dest in filenames:
            generate_results(src, dest, dehaze)

if __name__ == '__main__':
    main()
