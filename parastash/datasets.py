from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from parastash.transforms import pil_can_open, load_image

import tempfile
import os
import tarfile
import shutil
import logging
import torch

from concurrent.futures.thread import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class Dataset(ImageFolder):
    def __init__(self, root, transformations):
        super().__init__(
            root,
            transform=transformations,
            is_valid_file=pil_can_open,
            loader=load_image,
        )

    def create_loader(self, batch_size, shuffle,
                      drop_last=False, num_workers=0, pin_memory=True) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )


class FolderDataset(Dataset):
    def __init__(self, storage_dir, transformations=None):
        super().__init__(storage_dir, transformations)


def _extract_tarball(tar_path: str, extract_root: str, remove=False):
    logger.info("[tar] extract %s into %s...",tar_path,tar_path)
    with tarfile.open(tar_path) as my_tar:
        my_tar.extractall(extract_root)  # specify which folder to extract to
    if remove:
        os.remove(tar_path)


class TarDataset(Dataset):

    def __init__(self, tar_file, transformations=None, remove_tarballs=False, dest_dir=None):
        if dest_dir:
            if os.path.exists(dest_dir):
                raise ValueError("%s directory exist" % dest_dir)
            os.makedirs(dest_dir)
            self.extracted_directory = dest_dir
        else:
            self.extracted_directory = tempfile.mkdtemp()

        _extract_tarball(tar_file, self.extracted_directory, remove_tarballs)
        super().__init__(self.extracted_directory, transformations)

    def __del__(self):
        shutil.rmtree(self.extracted_directory)
