from PIL import Image

from torchvision import transforms
from torchvision.transforms.functional import pad

import io
import PIL
import abc

from typing import Union

MEAN = [0.5347586, 0.49634728, 0.48577875]
STD = [0.34097806, 0.3337877, 0.33717933]
INSTAGRAM_MEAN = [0.485, 0.456, 0.406]
INSTAGRAM_STD = [0.229, 0.224, 0.225]


class SquarePad:
    def __init__(self, fill=0, padding_mode="constant"):
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return pad(image, padding, self.fill, self.padding_mode)


class BiTransform(object):
    def __init__(self, left_transform, right_transform=None):
        self.left_transform = left_transform
        self.right_transform = right_transform or left_transform

    def __call__(self, sample):
        left = self.left_transform(sample)
        right = self.right_transform(sample)
        return left, right


def resize_augmentations(input_shape=224):
    """
    Return image transformations for inference.
    Images are padded to a square with zeros, and then resized to
    (input_shape, input_shape).


    :param input_shape: The expected output size, interpreted as square
    of (input_shape, input_shape)
    :param mean: The mean to use in normalization
    :param std: The std to use in normalization
    :return: The composed torchvision transform
    """

    transformations = transforms.Compose(
        [
            transforms.Resize((input_shape, input_shape)),
        ]
    )
    return transformations


def basic_augmentations(input_shape=224, mean=None, std=None):
    """
    Return image transformations for inference.
    Images are padded to a square with zeros, and then resized to
    (input_shape, input_shape).


    :param input_shape: The expected output size, interpreted as square
    of (input_shape, input_shape)
    :param mean: The mean to use in normalization
    :param std: The std to use in normalization
    :return: The composed torchvision transform
    """
    mean = mean or MEAN
    std = std or STD

    transformations = transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((input_shape, input_shape)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return transformations


def byol_augmentations(input_shape=224, mean=None, std=None):
    """
    Return image augmentations as described in https://arxiv.org/abs/2006.07733

    :param input_shape: The expected output size, interpreted as square
    of (input_shape, input_shape)
    :param mean: The mean to use in normalization
    :param std: The std to use in normalization
    :return: The composed torchvision transform
    """
    mean = mean or MEAN
    std = std or STD

    kernel_size = input_shape // 10
    if kernel_size % 2 == 0:
        kernel_size += 1

    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
    gaussian_blur = transforms.GaussianBlur(kernel_size=kernel_size)
    common_prefix = (
        transforms.RandomResizedCrop(size=input_shape, interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
    )
    common_suffix = (
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    )
    left_transforms = transforms.Compose(
        (*common_prefix, gaussian_blur, *common_suffix)
    )
    right_transforms = transforms.Compose(
        (*common_prefix, transforms.RandomApply([gaussian_blur], p=0.1), *common_suffix)
    )
    return BiTransform(left_transforms, right_transforms)


def pil_can_open(filename):
    """
    Return True if PIL can successfully open the provided file.
    Checks for corrupted files.
    """
    try:
        load_image(filename)
        return True
    except (PIL.UnidentifiedImageError, ValueError, OSError):
        return False


def load_image(f: Union[str, bytes]):
    """
    Load image from file and convert it to RGB
    """
    if isinstance(f, str):
        with open(f, "rb") as f:
            image = Image.open(f)
            image = _convert_image(image)
        return image
    elif isinstance(f, bytes):
        image = Image.open(io.BytesIO(f))
        return _convert_image(image)
    else:
        raise ValueError("type")


def save_image(f: PIL.Image, dest: str):
    f.save(dest)


def get_image_rgb(bytes_):
    """
    Load image from bytes and convert it to RGB
    """
    image = Image.open(io.BytesIO(bytes_))
    return _convert_image(image)


def _convert_image(image):
    """
    Convert a PIL Image to RGB.
    Standard PIL.Image.convert does not play nice with images with
    transparency as it puts them on a black background.
    Use pure_pil_alpha_to_color_v2 to put on a white background instead.
    """
    special_modes = (
        "LA",
        "RGBa",
        "PA",
    )
    has_transparency = image.info.get("transparency") is not None
    if (
            has_transparency and image.mode in ("L", "RGB", "P")
    ) or image.mode in special_modes:
        image = image.convert("RGBA")

    if image.mode == "RGBA":
        image = _pure_pil_alpha_to_color_v2(image)
    image = image.convert("RGB")
    return image


def _pure_pil_alpha_to_color_v2(image, color=(255, 255, 255)):
    """
    Alpha composite an RGBA Image with a specified color.
    Simpler, faster version than the solutions above.
    Source: http://stackoverflow.com/a/9459208/284318
    Keyword Arguments:
    image -- PIL RGBA Image object
    color -- Tuple r, g, b (default 255, 255, 255)
    """
    image.load()  # needed for split()
    background = Image.new("RGB", image.size, color)
    background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return background
