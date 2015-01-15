##Dependencies

1. pillow(~2.6.0)
2. Numpy(~1.9.0)

If the scripts throw `AttributeError: __float__`, make sure your pillow has jpeg support e.g. try:

    $ sudo apt-get install libjpeg-dev
    $ sudo pip uninstall pillow
    $ sudo pip install pillow

##How to generate the results

Enter the `src` directory, run `python main.py`. It will use images under `img` directory as default to produce the results. The results will show up in `result` directory.

To test special configurations for a given image, for example, to test the image with index `0` (check `IMG_NAMES` in `util.py` for indexes) and `tmin = 0.2`, `Amax = 170`, `w = 15`, `r = 40`, run

    $ python main.py -i 0 -t 0.2 -A 170 -w 15 -r 40

## Naming convetion of the results

For input image `name.jpg` using the default parameters, the naming convention is:

1. dark channel: `name-dark.jpg`
2. raw transmission map: `name-rawt.jpg`
3. refined tranmission map: `name-refinedt.jpg`
4. image dehazed with the raw transmission map: `name-radiance-rawt.jpg`
5. image dehazed with the refined transmission map: `name-radiance-refinedt.jpg`

If there are special configurations for the parameters, for example, , then the base name will be appended with `-20-170-50-40` e.g. the dark channel is `name-dark-20-170-50-40.jpg`

##Directory structure

    .
	├─ README.md
	├─ requirements.txt
	├─ doc
	│   └── report.pdf
	├─ img (source images)
	│   └── ... (input images from CVPR 09 supplementary materials)
	├─ result (the results)
    │   └── ...
	└─ src (the python source code)
        ├── dehaze.py (dehazing using the dark channel prior)
        ├── main.py (generate the results for the report)
        ├── guidedfilter.py (guided filter)
        └── util.py (utilities)

##About

* [Github repository](https://github.com/joyeecheung/dark-channel-prior-dehazing)
* Author: Qiuyi Zhang
* Time: Jan. 2015
