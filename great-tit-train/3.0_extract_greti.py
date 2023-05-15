from pathlib import Path
from time import strftime

import albumentations as albu
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from oml.const import MEAN, STD, TNormParam
from oml.models.vit.vit import ViTExtractor
from oml.utils.images.images import imread_cv2
from oml.utils.misc import git_root

# set root of GIT repository
root = Path(git_root()) / "great-tit-train"
dataset = "GRETI"
seed = 42

# set paths
data_dir = root / "datasets" / dataset
train_folder = data_dir / "train"
df_path = data_dir / "df.csv"

# get all image paths and labels
image_paths = list(train_folder.glob("*/*.jpg"))
labels = [path.parent.name for path in image_paths]

best_model = "2023-05-15_11-38-59_metric_learning"
best_model_path = root / "logs" / dataset / best_model / "checkpoints" / "best.ckpt"


extractor = ViTExtractor(
    str(best_model_path),
    "vits16",
    normalise_features=False,
)


# import image from image_paths as ndarray:
def get_greti_augs_val(
    im_size: int, mean: TNormParam = MEAN, std: TNormParam = STD
) -> albu.Compose:
    return albu.Compose(
        [
            albu.RandomCrop(im_size, im_size),
        ]
    )


# imagepaths = np.random.choice(image_paths, 20)


# num_images = len(imagepaths)
# num_rows = int(np.ceil(num_images / 5))
# fig, axs = plt.subplots(num_rows, 5, figsize=(15, num_rows * 3))
# if num_rows == 1:
#     axs = np.reshape(axs, (1, -1))
# for i, image_path in enumerate(imagepaths):
#     image = np.array(imread_cv2(image_path))
#     image = get_greti_augs_val(224)(image=image)["image"]
#     # extract features from image:
#     kk = extractor.draw_attention(image)
#     ax = axs[i // 5, i % 5]
#     ax.imshow(kk)
#     ax.set_axis_off()
# plt.tight_layout()
# plt.show()


from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.transform import resize


def plot_images_with_attention(
    image_paths: List[str],
    num_columns: int,
    image_size: int,
    fig_size: int,
    aspect_ratio: float,
):
    num_images = len(image_paths)
    num_rows = int(np.ceil(num_images / num_columns))
    num_columns = min(num_columns, num_images)

    # Calculate figure size to maintain constant spacing between rows and columns
    fig_width = int(num_columns * fig_size / aspect_ratio)
    fig_height = num_rows * fig_size
    fig_size = (fig_width, fig_height)

    fig, axs = plt.subplots(num_rows, num_columns * 2, figsize=fig_size)
    for ax in axs.ravel():
        ax.imshow(np.zeros((image_size, image_size, 3)))
        ax.set_axis_off()
    for i, image_path in enumerate(image_paths):
        image = np.array(imread_cv2(image_path))
        image = get_greti_augs_val(224)(image=image)["image"]
        attention_map = extractor.draw_attention(image)
        image_resized = resize(image, (image_size, image_size))
        attention_map_resized = resize(attention_map, (image_size, image_size))
        ax = axs[i // num_columns, 2 * (i % num_columns)]
        ax.imshow(image_resized)
        ax.set_axis_off()
        ax = axs[i // num_columns, 2 * (i % num_columns) + 1]
        ax.imshow(attention_map_resized)
        ax.set_axis_off()
    plt.tight_layout()
    for ax in axs.ravel()[num_images * 2 :]:
        ax.set_visible(False)
    return fig


# define image paths, ViTExtractor object, and other parameters
imagepaths = np.random.choice(image_paths, 20)
num_cols = 5
num_rows = int(np.ceil(len(imagepaths) / num_cols))
image_size = 224
fig_size = 10
aspect_ratio = 0.75  # e.g., 1.0 for a square figure, 1.5 for a landscape figure, 0.75 for a portrait figure

# create the plot
fig = plot_images_with_attention(
    imagepaths, num_cols, image_size, fig_size, aspect_ratio
)


# generate a new name for the figure each time
time = strftime("%Y%m%d_%H%M%S")
fig_dir = root / "outputs" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
pdf_filename = f"attention_plot_{time}.pdf"
png_filename = f"attention_plot_{time}.png"

with PdfPages(fig_dir / pdf_filename) as pdf:
    pdf.savefig(fig)

# also save the figure as a png
fig.savefig(fig_dir / png_filename)
