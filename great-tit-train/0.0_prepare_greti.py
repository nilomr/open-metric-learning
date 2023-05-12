import json
import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from oml.utils.misc import git_root


# set root of GIT repository
def git_root():
    import subprocess

    return (
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .decode("utf-8")
        .strip()
    )


root = Path(git_root()) / "great-tit-train"
dataset = "GRETI"
seed = 42

# set paths
data_dir = root / "datasets" / dataset
train_folder = data_dir / "train"  # TODO: just export all images together
test_folder = data_dir / "test"
df_path = data_dir / "df.csv"

# get all image paths and labels


# TODO: just export all images together
def sort_paths(paths):
    """Sorts a list of pathlib.Path objects based on the name of the parent folder."""
    parent_names = [path.parent.name for path in paths]
    sorted_indices = sorted(range(len(parent_names)), key=lambda i: parent_names[i])
    return [paths[i] for i in sorted_indices]


image_paths = list(train_folder.glob("*/*.jpg"))  # + list(test_folder.glob("*/*.jpg"))
# order_paths = sort_paths(image_paths)
labels = [path.parent.name for path in image_paths]


# create dataframe and shuffle within each group (based on 'label' column)
df = (
    pd.DataFrame(
        {
            "label": labels,
            "path": image_paths,
        }
    )
    .groupby("label")
    .apply(lambda x: x.sample(frac=1, random_state=seed))
    .reset_index(drop=True)
)

# split into train and validation set wihin each group:
train, validate = train_test_split(
    df, test_size=0.4, stratify=df["label"], random_state=seed
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

# conver the label column to int, saving a dictionary to map back:
label_map = {label: i for i, label in enumerate(split_df.label.unique())}
split_df.label = split_df.label.map(label_map)

split_df.groupby(["label", "split"]).count()

# save label map as json in data_dir:
with open(data_dir / "label_map.json", "w") as f:
    json.dump(label_map, f)


split_df.to_csv(df_path, index=False)
