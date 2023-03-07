from pathlib import Path
from matplotlib import pyplot as plt
import albumentations as albu

import numpy as np
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

best_model = "GRETI2023-03-05_22-42-42_metric_learning"
best_model_path = root / "logs" / best_model / "checkpoints" / "best.ckpt"


extractor = ViTExtractor(
    str(best_model_path),
    "vits16",
    normalise_features=False,
)

print(extractor.model.patch_embed.proj.kernel_size[0])

# import image from image_paths as ndarray:


def get_greti_augs_val(
    im_size: int, mean: TNormParam = MEAN, std: TNormParam = STD
) -> albu.Compose:
    return albu.Compose(
        [
            albu.RandomCrop(im_size, im_size),
        ]
    )


imagepaths = np.random.choice(image_paths, 20)


num_images = len(imagepaths)
num_rows = int(np.ceil(num_images / 5))
fig, axs = plt.subplots(num_rows, 5, figsize=(15, num_rows * 3))
if num_rows == 1:
    axs = np.reshape(axs, (1, -1))
for i, image_path in enumerate(imagepaths):
    image = np.array(imread_cv2(image_path))
    image = get_greti_augs_val(224)(image=image)["image"]
    # extract features from image:
    kk = extractor.draw_attention(image)
    ax = axs[i // 5, i % 5]
    ax.imshow(kk)
    ax.set_axis_off()
plt.tight_layout()
plt.show()
