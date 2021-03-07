import os
import sys
import random
import tqdm

import parastash.model.transforms as T
from multiprocessing import Pool

import PIL.Image

PIL.Image.MAX_IMAGE_PIXELS = None


def is_jpeg(file):
    _, file_extension = os.path.splitext(file)
    return file_extension.endswith('jpeg')


AUG = T.resize_augmentations()
SRC_DIR = '/Users/bohdans/.stash/caches'
DEST_DIR = '/Users/bohdans/.stash/dataset_002'

TRAIN_DIR = f"{DEST_DIR}/train/class_001"
TEST_DIR = f"{DEST_DIR}/test/class_001"

NO_OVERRIDE = True


def _is_no_resize(d):
    return os.path.exists(d) and NO_OVERRIDE


def _resize(src, dest):
    try:
        img = AUG(T.load_image(src))
        T.save_image(img, dest)
    except OSError as e:
        print(f"{src} processing failed - {e}", file=sys.stderr)


def process_train(tuple):
    dest, src = tuple
    dest_f = f"{TRAIN_DIR}/{dest}"
    if _is_no_resize(dest_f):
        return

    _resize(src, dest_f)


def process_test(tuple):
    dest, src = tuple
    dest_f = f"{TEST_DIR}/{dest}"
    if _is_no_resize(dest_f):
        return

    _resize(src, dest_f)


def mk_dataset_linked(storage_dir, test_pct=0.3, is_shuffle=False):
    data = [
        (f"{os.path.basename(address)}-index.jpeg", f"{address}/{file}")
        for address, dirs, files in os.walk(storage_dir) if files
        for file in files if str(file) if is_jpeg(file)
    ]
    if is_shuffle:
        data = random.shuffle(data)

    train = int(len(data) * (1.0 - test_pct))
    test = len(data) - train
    print(f"Processing data: total={len(data)}, train={train}, test={test}")

    train_data = data[:train]
    test_data = data[train:]

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    def _launch(do_work, data, pool):
        for _ in tqdm.tqdm(pool.imap_unordered(do_work, data), total=len(data)):
            pass

    with Pool(4) as p:
        _launch(process_train, train_data, p)

    with Pool(4) as p:
        _launch(process_test, test_data, p)

    print("Finish.")


mk_dataset_linked(
    storage_dir=SRC_DIR,
)
