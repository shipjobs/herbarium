# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import datetime
import io
import itertools
import json
import logging
import pickle
import numpy as np
import os
import shutil
import pycocotools.mask as mask_util
import multiprocessing as mp
from itertools import product

from fvcore.common.timer import Timer
from iopath.common.file_io import file_lock
from PIL import Image
from tqdm import tqdm
import torch

#from herbarium.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from herbarium.utils.file_io import PathManager

from .. import DatasetCatalog, MetadataCatalog

"""
This file contains functions to parse Herb-format annotations into dicts in "Herbarium format".
"""


logger = logging.getLogger(__name__)

__all__ = ["load_herb_json", "convert_to_herb_json", "register_herb_instances"]

def process_per_record(img_ann, image_root, ann_keys, meta):
    #print("Processor {} start".format(worker_id))
    img_dict, anno_dict_list = img_ann
    record = {}
    record["file_name"] = os.path.join(image_root, img_dict["file_name"])
    record["height"] = img_dict["height"]
    record["width"] = img_dict["width"]
    image_id = record["image_id"] = img_dict["id"]

    objs = []
    for anno in anno_dict_list:
        assert anno["image_id"] == image_id

        obj = {key: anno[key] for key in ann_keys if key in anno}
        # TODO: change class_id into hierarchy id here
        if meta is not None:
            curr_category = meta.cats[anno['category_id']]
            obj["order_id"] = meta.order_map[curr_category["order"]]
            obj["family_id"] = meta.family_map[curr_category["family"]]
            obj["species_id"] = meta.species_map[curr_category["name"]]

        objs.append(obj)
    record["annotations"] = objs

    return record

class anns_generator:
    def __init__(self, anns, w_id, num_workers):
        self.anns = anns
        self.w_id = w_id
        self.num_workers = num_workers

    def __iter__(self):
        for ann in itertools.islice(self.anns, self.w_id, None, self.num_workers):
            yield ann 

def update_meta(json_file, dataset_name=None):

    from pyherbtools.herb import HERB

    if dataset_name is not None and "test" not in dataset_name:

        logger.info("Update Metadat of {} dataset".format(dataset_name))
        timer = Timer()
        json_file = PathManager.get_local_path(json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            herb_api = HERB(json_file)
        if timer.seconds() > 1:
            logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(herb_api.getCatIds())
        cats = herb_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        logger.info("Creating hierarchy target from given annotation")

        order_family_hierarchy = torch.zeros(len(meta.family_map),len(meta.order_map))
        family_species_hierarchy = torch.zeros(len(meta.species_map),len(meta.family_map))
        
        for cat in cats:
            order_id = meta.order_map[cat["order"]]
            family_id = meta.family_map[cat["family"]]
            species_id = meta.species_map[cat["name"]]

            order_family_hierarchy[family_id][order_id] = 1e2
            family_species_hierarchy[species_id][family_id] = 1e2
        
        from torch import nn
        meta.order_family_hierarchy = nn.Softmax(dim=1)(order_family_hierarchy)
        meta.family_species_hierarchy = nn.Softmax(dim=1)(family_species_hierarchy)
        meta.cats = cats

        meta.num_classes = {
            "family": len(meta.family_map),
            "order": len(meta.order_map),
            "species": len(meta.species_map),
        }


def load_herb_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file with Herbarium's instances annotation format.
    Currently supports Family, Order, class annotations.

    Args:
        json_file (str): full path to the json file in Herb instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str or None): the name of the dataset (e.g., herb_2021_train).
            When provided, this function will also do the following:

            * Put "family", "order", "name" into the metadata associated with this dataset.
            * Build Class hierarchy in metadataset 
            * Map the category ids into a hierarchy id and continuous id (needed by standard dataset format),
              and add "hierarchy_id_to_contiguous_id" to the metadata associated
              with this dataset.

            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict. The values for these keys will be returned as-is.
            For example, the region_id annotations are loaded in this way.
            
            * Currently region_id is not provided in dataset

    Returns:
        list[dict]: a list of dicts in Herbarium standard dataset dicts format 
        when `dataset_name` is not None.
        If `dataset_name` is None, the returned `category_ids` may be
        incontiguous and may not conform to the Herbarium standard format.

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pyherbtools.herb import HERB

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        herb_api = HERB(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    # sort indices for reproducible results
    img_ids = sorted(herb_api.imgs.keys())
    imgs = herb_api.loadImgs(img_ids)
    anns = [herb_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(herb_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in HERB format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []

    ann_keys = ["category_id", "hierarchy_id"] + (extra_annotation_keys or [])

    logger.info("Convert HERB format into herbarium format")

    timer = Timer()



    if "test" not in dataset_name:
        meta = MetadataCatalog.get(dataset_name)
        dataset_dicts = [process_per_record(anns, image_root, ann_keys, meta) for anns in imgs_anns]

    logger.info("Processing Record takes {:.2f} seconds.".format(timer.seconds()))

    return dataset_dicts


# TODO: Change here to fit on herbarium dataset
# Need at evaluation stage

def convert_to_herb_dict(dataset_name):
    """
    Convert an instance detection/segmentation or keypoint detection dataset
    in herbarium's standard format into COCO json format.

    Generic dataset description can be found here:
    https://herbarium.readthedocs.io/tutorials/datasets.html#register-a-dataset

    COCO data format description can be found here:
    http://cocodataset.org/#format-data

    Args:
        dataset_name (str):
            name of the source dataset
            Must be registered in DatastCatalog and in herbarium's standard format.
            Must have corresponding metadata "thing_classes"
    Returns:
        coco_dict: serializable dict in COCO json format
    """

    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    # unmap the category mapping ids for COCO
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
        reverse_id_mapping = {v: k for k, v in metadata.dataset_id_to_hierarchy_id.items()}
        reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[contiguous_id]  # noqa
    else:
        reverse_id_mapper = lambda contiguous_id: contiguous_id  # noqa

    categories = [
        {"id": reverse_id_mapper(id), "name": name}
        for id, name in enumerate(metadata.thing_classes)
    ]

    logger.info("Converting dataset dicts into COCO format")
    coco_images = []
    coco_annotations = []

    for image_id, image_dict in enumerate(dataset_dicts):
        coco_image = {
            "id": image_dict.get("image_id", image_id),
            "width": int(image_dict["width"]),
            "height": int(image_dict["height"]),
            "file_name": str(image_dict["file_name"]),
        }
        coco_images.append(coco_image)

        anns_per_image = image_dict.get("annotations", [])
        for annotation in anns_per_image:
            # create a new dict with only COCO fields
            coco_annotation = {}

            # COCO requirement: XYWH box format for axis-align and XYWHA for rotated
            bbox = annotation["bbox"]
            if isinstance(bbox, np.ndarray):
                if bbox.ndim != 1:
                    raise ValueError(f"bbox has to be 1-dimensional. Got shape={bbox.shape}.")
                bbox = bbox.tolist()
            if len(bbox) not in [4, 5]:
                raise ValueError(f"bbox has to has length 4 or 5. Got {bbox}.")
            from_bbox_mode = annotation["bbox_mode"]
            to_bbox_mode = BoxMode.XYWH_ABS if len(bbox) == 4 else BoxMode.XYWHA_ABS
            bbox = BoxMode.convert(bbox, from_bbox_mode, to_bbox_mode)

            # COCO requirement: instance area
            if "segmentation" in annotation:
                # Computing areas for instances by counting the pixels
                segmentation = annotation["segmentation"]
                # TODO: check segmentation type: RLE, BinaryMask or Polygon
                if isinstance(segmentation, list):
                    polygons = PolygonMasks([segmentation])
                    area = polygons.area()[0].item()
                elif isinstance(segmentation, dict):  # RLE
                    area = mask_util.area(segmentation).item()
                else:
                    raise TypeError(f"Unknown segmentation type {type(segmentation)}!")
            else:
                # Computing areas using bounding boxes
                if to_bbox_mode == BoxMode.XYWH_ABS:
                    bbox_xy = BoxMode.convert(bbox, to_bbox_mode, BoxMode.XYXY_ABS)
                    area = Boxes([bbox_xy]).area()[0].item()
                else:
                    area = RotatedBoxes([bbox]).area()[0].item()

            if "keypoints" in annotation:
                keypoints = annotation["keypoints"]  # list[int]
                for idx, v in enumerate(keypoints):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # For COCO format consistency we substract 0.5
                        # https://github.com/facebookresearch/herbarium/pull/175#issuecomment-551202163
                        keypoints[idx] = v - 0.5
                if "num_keypoints" in annotation:
                    num_keypoints = annotation["num_keypoints"]
                else:
                    num_keypoints = sum(kp > 0 for kp in keypoints[2::3])

            # COCO requirement:
            #   linking annotations to images
            #   "id" field must start with 1
            coco_annotation["id"] = len(coco_annotations) + 1
            coco_annotation["image_id"] = coco_image["id"]
            coco_annotation["bbox"] = [round(float(x), 3) for x in bbox]
            coco_annotation["area"] = float(area)
            coco_annotation["iscrowd"] = int(annotation.get("iscrowd", 0))
            coco_annotation["category_id"] = int(reverse_id_mapper(annotation["category_id"]))

            # Add optional fields
            if "keypoints" in annotation:
                coco_annotation["keypoints"] = keypoints
                coco_annotation["num_keypoints"] = num_keypoints

            if "segmentation" in annotation:
                seg = coco_annotation["segmentation"] = annotation["segmentation"]
                if isinstance(seg, dict):  # RLE
                    counts = seg["counts"]
                    if not isinstance(counts, str):
                        # make it json-serializable
                        seg["counts"] = counts.decode("ascii")

            coco_annotations.append(coco_annotation)

    logger.info(
        "Conversion finished, "
        f"#images: {len(coco_images)}, #annotations: {len(coco_annotations)}"
    )

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated COCO json file for Detectron2.",
    }
    coco_dict = {"info": info, "images": coco_images, "categories": categories, "licenses": None}
    if len(coco_annotations) > 0:
        coco_dict["annotations"] = coco_annotations
    return coco_dict


def convert_to_herb_json(dataset_name, output_file, allow_cached=True):
    """
    Converts dataset into COCO format and saves it to a json file.
    dataset_name must be registered in DatasetCatalog and in herbarium's standard format.

    Args:
        dataset_name:
            reference from the config file to the catalogs
            must be registered in DatasetCatalog and in herbarium's standard format
        output_file: path of json file that will be saved to
        allow_cached: if json file is already present then skip conversion
    """

    # TODO: The dataset or the conversion script *may* change,
    # a checksum would be useful for validating the cached data

    PathManager.mkdirs(os.path.dirname(output_file))
    with file_lock(output_file):
        if PathManager.exists(output_file) and allow_cached:
            logger.warning(
                f"Using previously cached COCO format annotations at '{output_file}'. "
                "You need to clear the cache file if your dataset has been modified."
            )
        else:
            logger.info(f"Converting annotations of dataset '{dataset_name}' to HERB format ...)")
            coco_dict = convert_to_herb_dict(dataset_name)

            logger.info(f"Caching COCO format annotations at '{output_file}' ...")
            tmp_file = output_file + ".tmp"
            with PathManager.open(tmp_file, "w") as f:
                json.dump(coco_dict, f)
            shutil.move(tmp_file, output_file)


def register_herb_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in Herbarium's json annotation format for classification.

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "herb_2021_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_herb_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging


    if metadata is not None:
        MetadataCatalog.get(name).set(
            json_file=json_file, image_root=image_root, evaluator_type="herb", **metadata
        )

        update_meta(json_file, name)


if __name__ == "__main__":
    """
    Test the Herbarium json dataset loader.

    Usage:
        python -m herbarium.data.datasets.herb \
            path/to/json path/to/image_root dataset_name

        "dataset_name" can be "herb_2021_val", or other
        pre-registered ones
    """
    from herbarium.utils.logger import setup_logger
    import herbarium.data.datasets  # noqa # add pre-defined metadata
    import sys

    logger = setup_logger(name=__name__)
    assert sys.argv[3] in DatasetCatalog.list()
    meta = MetadataCatalog.get(sys.argv[3])

    dicts = load_herb_json(sys.argv[1], sys.argv[2], sys.argv[3])
    logger.info("Done loading {} samples.".format(len(dicts)))

    dirname = "herb-data-vis"
    os.makedirs(dirname, exist_ok=True)
    for d in dicts:
        img = np.array(Image.open(d["file_name"]))
        # TODO: do something for visualize in this "herb-data-vis" and implement in util.visualizer