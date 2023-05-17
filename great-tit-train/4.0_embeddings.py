from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as t

from oml.const import MEAN, STD, TNormParam
from oml.inference.flat import inference_on_images
from oml.models import ViTExtractor
from oml.models.vit.vit import ViTExtractor
from oml.utils.misc import git_root
from oml.utils.misc_torch import pairwise_dist
import json
import pickle

# ──── FUNCTION DEFINITIONS ─────────────────────────────────────────────────────


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


def build_comparison_dataframe(dist_mat, filenames):
    num_files = len(filenames)
    comparisons = []

    for i in range(num_files):
        for j in range(i + 1, num_files):
            filename1 = filenames[i]
            filename2 = filenames[j]
            distance = dist_mat[i, j]

            comparison = {
                "f1": filename1,
                "f2": filename2,
                "distance": distance,
            }
            comparisons.append(comparison)

    df = pd.DataFrame(comparisons)
    return df


# ──── SETTINGS ─────────────────────────────────────────────────────────────────


# set root of GIT repository
root = Path(git_root()) / "great-tit-train"
dataset = "GRETI"
seed = 42

# set paths
data_dir = root / "datasets" / dataset
output_dir = data_dir / "output"
train_folder = data_dir / "train"
df_path = data_dir / "df.csv"
img_feat_file = output_dir / "img_feat.npy"
img_feat_file.parent.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)
lab_dict_path = data_dir / "label_map.json"


# ──── IMPORT DATA ──────────────────────────────────────────────────────────────

# get all image paths and labels
image_paths = list(train_folder.glob("*/*.jpg"))
labels = [path.parent.name for path in image_paths]

best_model = "2023-05-15_11-38-59_metric_learning"
best_model_path = root / "logs" / dataset / best_model / "checkpoints" / "best.ckpt"

df = pd.read_csv(df_path)
df = df[df["split"] == "validation"]
queries = df[df["is_query"]][
    "path"
].tolist()  # TODO: this and any consequence needs to change
# galleries = df[df["is_gallery"]]["path"].tolist()

# ──── EXTRACT OR READ FEATURE VECTORS: ────────────────────────────────────────

extractor = ViTExtractor(
    str(best_model_path),
    "vits16",
    normalise_features=True,
)

if img_feat_file.exists():
    features_queries = torch.from_numpy(np.load(img_feat_file))
    print(features_queries.shape)

else:
    nworkers = 10
    bsize = len(queries) // nworkers
    args = {
        "num_workers": nworkers,
        "batch_size": bsize,
        "verbose": True,
        "use_fp16": True,
    }
    features_queries = inference_on_images(
        extractor, paths=queries, transform=get_normalisation_resize_greti(), **args
    )
    np.save(img_feat_file, features_queries)


# Calculate median of vectors per class using classes from queries.label as indexes
# and the tensor of vectors from 'features_queries', which has the same order:

# Read label dict:
label_dict = json.load(open(lab_dict_path))
# inverse label dict - NOTE: remove when updating
label_dict = {v: k for k, v in label_dict.items()}

# Get labels of queries:
labels = df[df["is_query"]]["label"].to_list()
labels = [label_dict[label] for label in labels]

query_ids = [
    Path(path).parent.name for path in queries
]  # NOTE will not work with new dataset format

assert labels == query_ids, "labels and query_ids are not in the same order"

labels = np.array(labels)
n_feats = np.array(features_queries)
unique_labels = np.unique(labels)
label_indices = [np.where(labels == label)[0] for label in unique_labels]
median_vectors = [np.median(n_feats[indices], axis=0) for indices in label_indices]
median_vectors = torch.from_numpy(np.array(median_vectors))
median_dist_mat = pairwise_dist(x1=median_vectors, x2=median_vectors)
median_dist_mat = median_dist_mat.cpu().numpy()


# Calculate distance matrix for all queries
dist_mat = pairwise_dist(x1=features_queries, x2=features_queries)
dist_mat = dist_mat.cpu().numpy()
# remove everything below the diagonal
dist_mat[np.tril_indices(dist_mat.shape[0])] = np.nan

# Convert to long format, with one row per comparison
filenames = [Path(path).stem for path in queries]
dist_df_long = build_comparison_dataframe(dist_mat, filenames)
median_dist_df_long = build_comparison_dataframe(median_dist_mat, unique_labels)

# order by distance:
median_dist_df_long = median_dist_df_long.sort_values(by="distance")

# save all ditance matrices and the long format dataframes to pickle files (not numpy files)
# (under output_dir)
median_dist_df_long.to_pickle(output_dir / "median_dist_df_long.pkl")
dist_df_long.to_pickle(output_dir / "dist_df_long.pkl")
# dump into pickles using context manager
with open(output_dir / "median_dist_mat.pkl", "wb") as f:
    pickle.dump(median_dist_mat, f)
with open(output_dir / "dist_mat.pkl", "wb") as f:
    pickle.dump(dist_mat, f)
