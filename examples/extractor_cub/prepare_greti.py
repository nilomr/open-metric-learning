import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from oml.utils.misc import git_root

# set root of GIT repository
root = Path(git_root())
dataset = "GRETI"
seed = 42

# set paths
data_dir = root / "examples" / "datasets" / dataset
train_folder = Path(data_dir / "train")
df_path = Path(data_dir / "df.csv")

# get all image paths and labels
image_paths = list(train_folder.glob("*/*.jpg"))
labels = [path.parent.name for path in image_paths]


# create dataframe
df = pd.DataFrame(
    {
        "label": labels,
        "path": image_paths,
    }
)

# shuffle within each group (based on labelcolumn):
df = (
    df.groupby("label")
    .apply(lambda x: x.sample(frac=1, random_state=seed))
    .reset_index(drop=True)
)

# split into train and validation set wihin each group:
train, validate = train_test_split(
    df, test_size=0.5, stratify=df["label"], random_state=seed
)

# concatenate train and validate, adding a column to indicate split:
train["split"], validate["split"] = "train", "validation"
split_df = pd.concat([train, validate]).sort_values(by=["label"])


val_indices = split_df[split_df["split"] == "validation"].index
split_df.loc[val_indices, "split"] = "validation"
split_df.loc[val_indices, ["is_query", "is_gallery"]] = True

# change false in ["is_query", "is_gallery"] to na
split_df.loc[:, ["is_query", "is_gallery"]] = split_df.loc[
    :, ["is_query", "is_gallery"]
].replace({False: pd.NA})

# check that each split by label has at least two images:
if split_df.groupby(["label", "split"]).count().path.min() < 1:
    warnings.warn("Some classes have only one sample in one of the splits")

# conver the label column to int, saving a dictionaruy to map back:
label_map = {label: i for i, label in enumerate(split_df.label.unique())}
split_df.label = split_df.label.map(label_map)

split_df.to_csv(df_path, index=False)
