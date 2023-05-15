from pathlib import Path
from time import strftime

import albumentations as albu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as t
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torchvision.transforms import Compose, InterpolationMode, Normalize, ToTensor

from oml.const import MEAN, MOCK_DATASET_PATH, STD, TNormParam
from oml.inference.flat import inference_on_images
from oml.models import ViTExtractor
from oml.models.vit.vit import ViTExtractor
from oml.registry.transforms import get_transforms_for_pretrained
from oml.utils.download_mock_dataset import download_mock_dataset
from oml.utils.images.images import imread_cv2
from oml.utils.misc import git_root
from oml.utils.misc_torch import pairwise_dist

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


df = pd.read_csv(df_path)
df = df[df["split"] == "validation"]
queries = df[df["is_query"]]["path"].tolist()
# galleries = df[df["is_gallery"]]["path"].tolist()


extractor = ViTExtractor(
    str(best_model_path),
    "vits16",
    normalise_features=True,
)


def get_normalisation_resize_greti(
    im_size: int = 224,
    crop_size: int = 224,
    mean: TNormParam = MEAN,
    std: TNormParam = STD,
) -> t.Compose:
    transforms = t.Compose(
        [
            t.CenterCrop(crop_size),
            t.ToTensor(),
            t.Normalize(mean=mean, std=std),
        ]
    )
    return transforms


args = {"num_workers": 10, "batch_size": 400, "verbose": True, "use_fp16": True}
features_queries = inference_on_images(
    extractor, paths=queries, transform=get_normalisation_resize_greti(), **args
)

print(len(features_queries[0]))
print(features_queries[0].shape)
print(len(features_queries))
print(features_queries[0])

# features_galleries = inference_on_images(
#     extractor, paths=galleries, transform=transform, **args
# )

# # Now we can explicitly build pairwise matrix of distances or save you RAM via using kNN
# use_knn = True
# top_k = 3

# if use_knn:
#     from sklearn.neighbors import NearestNeighbors

#     knn = NearestNeighbors(algorithm="auto", p=2)
#     knn.fit(features_galleries)
#     dists, ii_closest = knn.kneighbors(
#         features_queries, n_neighbors=top_k, return_distance=True
#     )

# else:
#     dist_mat = pairwise_dist(x1=features_queries, x2=features_galleries)
#     dists, ii_closest = torch.topk(dist_mat, dim=1, k=top_k, largest=False)

# print(f"Top {top_k} items closest to queries are:\n {ii_closest}")
