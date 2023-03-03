from pathlib import Path
from typing import Union

import albumentations as albu
import cv2
import matplotlib
import numpy as np

from oml.const import MEAN, STD, TNormParam
from oml.transforms.images.albumentations.transforms import TTransformsList
from oml.utils.images.images import imread_cv2
from oml.utils.misc import git_root

# set root of GIT repository
root = Path(git_root())
dataset = "GRETI"
seed = 42
PAD_COLOR = (0, 0, 0)

# set paths
data_dir = root / "examples" / "datasets" / dataset
train_folder = Path(data_dir / "train")

train_images = list(train_folder.glob("*/*.jpg"))


imread_cv2(train_images[0])
# pl;ot image usiing cv2:
import cv2
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2


def get_blurs() -> TTransformsList:
    blur_augs = [
        albu.MotionBlur(),
        albu.MedianBlur(),
        albu.Blur(),
        albu.GaussianBlur(sigma_limit=(0.7, 2.0)),
    ]
    return blur_augs


def get_noises() -> TTransformsList:
    noise_augs = [
        albu.CoarseDropout(
            max_holes=3, max_height=20, max_width=20, fill_value=PAD_COLOR, p=0.3
        ),
        albu.GaussNoise(p=0.7),
        albu.ISONoise(p=0.7),
        albu.MultiplicativeNoise(p=0.7),
    ]
    return noise_augs


def get_augs_albu(
    im_size: int, mean: TNormParam = MEAN, std: TNormParam = STD
) -> albu.Compose:
    augs = albu.Compose(
        [
            albu.RandomCrop(im_size, im_size),
            albu.OneOf(get_blurs(), p=0.5),
            albu.CLAHE(p=0.2),
            albu.Sharpen(p=0.2),
            albu.Emboss(p=0.2),
            albu.RandomBrightnessContrast(p=0.3),
            albu.OneOf(get_noises(), p=0.5),
            albu.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=2,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                value=PAD_COLOR,
                p=0.3,
            ),
            ToTensorV2(),
        ],
    )
    return augs


def plot_augmentation_grid(
    image: str, n: int = 10, n_cols: int = 3, figsize: tuple = (8, 8), augs: list = None
) -> plt.Figure:
    """
    Plot a grid of image variations with augmentations applied.

    The function reads the image from the specified file path and displays a
    grid of `n` images. The first image is the original image, and the rest are
    variations with optional augmentations applied. The number of rows in the
    grid is automatically calculated based on the `n` and `n_cols` arguments.
    The augmentations are applied using the `augs` argument, which should be a
    list of image augmentation functions that accept an image as input and
    return a modified image. If `augs` is None or an empty list, no
    augmentations will be applied.

    Args:
        image (str or Path): Path to the image file to be displayed.
        n (int, optional): Total number of images to be displayed, including the original image. Defaults to 10.
        n_cols (int, optional): Number of columns in the grid. Defaults to 3.
        figsize (tuple, optional): Figure size. Defaults to (8, 8).
        augs (list, optional): List of augmentations to be applied. Defaults to None.

    Returns:
        plt.Figure: A Matplotlib Figure object containing the image grid.

    """
    if isinstance(image, (str, Path)):
        image = imread_cv2(image)  # type: ignore
    n_rows = int(n / n_cols) + 1
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        if i == 0:
            ax.imshow(image)
            ax.axis("off")
            ax.set_title("Original")
        else:
            if i - 1 < n:
                aug_image = augs(image=image)["image"]
                ax.imshow(aug_image.permute(1, 2, 0).numpy())
                ax.axis("off")
            else:
                ax.axis("off")

    fig.tight_layout()
    return fig


fig = plot_augmentation_grid(train_images[0], n=40, n_cols=7, augs=get_augs_albu(224))
plt.show()


# Quarantine until necessary

# albu.OneOf(get_spatials(), p=0.5),
# albu.OneOf(get_blurs(), p=0.5),
# albu.OneOf(get_colors_level(), p=0.8),
# albu.OneOf(get_noise_channels(), p=0.2),
# albu.OneOf(get_noises(), p=0.25),

# %%
