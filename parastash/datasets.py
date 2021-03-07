from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from parastash.model.transforms import pil_can_open, load_image


class Dataset(ImageFolder):
    def __init__(self, root, transformations):
        super().__init__(
            root,
            transform=transformations,
            is_valid_file=pil_can_open,
            loader=load_image,
        )

    def create_loader(self, batch_size, shuffle, drop_last=False, num_workers=0, pin_memory=True) -> DataLoader:
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

