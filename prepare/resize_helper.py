import os
import sys
import random
import tqdm

from parastash import transforms as T
from multiprocessing import Pool

import PIL.Image

PIL.Image.MAX_IMAGE_PIXELS = None

HOME = os.environ['HOME']
SRC_DIR = f'{HOME}/.stash/caches'
NO_OVERRIDE = True

W = 640

AUG = T.resize_augmentations_auto(W)


def is_jpeg(file):
    _, file_extension = os.path.splitext(file)
    return file_extension.endswith('jpeg')


def _is_no_resize(d):
    return os.path.exists(d) and NO_OVERRIDE


def _resize(src, dest):
    try:
        img = AUG(T.load_image(src))
        T.save_image(img, dest)
    except OSError as e:
        print(f"{src} processing failed - {e}", file=sys.stderr)


def resize(tuple):
    address, file = tuple

    if file != 'index.jpeg':
        return

    src = f"{address}/{file}"
    dest = f"{address}/index-w-640.jpeg"

    _resize(src, dest)


def resize_linked():
    data = [
        (f"{address}", f"{file}")
        for address, dirs, files in os.walk(SRC_DIR) if files
        for file in files if str(file) if is_jpeg(file)
    ]
    # print(data)
    print(f"len={len(data)}")

    def _launch(do_work, data, pool):
        for _ in tqdm.tqdm(pool.imap_unordered(do_work, data), total=len(data)):
            pass

    with Pool(4) as p:
        _launch(resize, data, p)

    print('done')


resize_linked()
