from pathlib import Path

import matplotlib.pyplot as plt

from oml.transforms.images.albumentations import get_greti_augs_train
from oml.utils.images.images import imread_cv2
from oml.utils.misc import git_root

# set root of GIT repository
root = Path(git_root()) / "great-tit-train"
dataset = "GRETI"
seed = 42
PAD_COLOR = (0, 0, 0)

# set paths
data_dir = root / "datasets" / dataset
train_folder = data_dir / "train"
train_images = list(train_folder.glob("*/*.jpg"))


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


fig = plot_augmentation_grid(
    train_images[0], n=60, n_cols=9, augs=get_greti_augs_train(224)
)
plt.show()
