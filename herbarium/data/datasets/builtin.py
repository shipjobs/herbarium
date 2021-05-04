# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from herbarium.data import DatasetCatalog, MetadataCatalog

from .builtin_meta import _get_builtin_metadata
from .herb import register_herb_instances
from pathlib import Path

# ==== Predefined datasets and splits for Herbarium ==========

_PREDEFINED_SPLITS_HERB = {}
_PREDEFINED_SPLITS_HERB["herb"] = {
    "herb_2021_train": ("herb/2021/train", "metadata.json", "train_annotations.json"),
    "herb_2021_val": ("herb/2021/train", "metadata.json", "val_annotations.json"),
    "herb_2021_test": ("herb/2021/test", "metadata.json", "annotations.json"),
}

def register_all_herbarium(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_HERB.items():
        for key, (dataset_root, metadata_file, annotation_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            dataset_root = os.path.join(root, dataset_root)
            images_root = dataset_root
            metadata_file =  os.path.join(Path(dataset_root).parent, metadata_file)
            annotation_file = os.path.join(dataset_root, annotation_file)
            register_herb_instances(
                key,
                _get_builtin_metadata(dataset_name, metadata_file),
                annotation_file if "://" not in annotation_file else annotation_file,
                images_root,
            )

# True for open source;
# Internally at fb, we register them elsewhere
if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("HERBARIUM_DATASETS", "datasets")
    register_all_herbarium(_root)
