from annotation_utils.coco.structs import COCO_Dataset
from annotation_utils.dataset.config.dataset_config import \
    DatasetConfigCollectionHandler, DatasetConfigCollection, DatasetConfig
from pasonatron.detectron2.util.dataset_parser import Detectron2_Annotation_Dict
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
import cv2
import os
from tqdm import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer

from logger import logger
from annotation_utils.coco.structs import COCO_Dataset
from common_utils.file_utils import file_exists

from pasonatron.detectron2.lib.roi_heads import CustomROIHeads, ROI_HEADS_REGISTRY
from pasonatron.detectron2.lib.trainer import COCO_Keypoint_Trainer

import copy
import torch
from imgaug import augmenters as iaa
import imgaug as ia
from imgaug.augmentables.polys import PolygonsOnImage
from imgaug.augmentables.kps import KeypointsOnImage
from imgaug.augmentables.bbs import BoundingBoxesOnImage
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data import build_detection_train_loader
import numpy as np
from logger import logger
from common_utils.cv_drawing_utils import cv_simple_image_viewer, draw_bbox, draw_keypoints, draw_segmentation
from common_utils.common_types.keypoint import Keypoint2D_List, Keypoint2D
from common_utils.common_types.bbox import BBox
from common_utils.common_types.segmentation import Segmentation, Polygon
from common_utils.file_utils import file_exists
from imageaug import AugHandler, Augmenter as aug

def load_augmentation_settings(handler_save_path: str):

    if not file_exists(handler_save_path):
        handler = AugHandler(
            [
                aug.Crop(percent=[0.2, 0.5]),
                aug.Affine(scale = {"x": tuple([0.8, 1.2]), "y":tuple([0.8, 1.2])}, translate_percent= {"x": tuple([0.1, 0.11]), "y":tuple([0.1, 0.11])}, rotate= [-180, 180], order= [0, 0], cval= [0, 0], shear= [0,0])
            ]
        )
        handler.save_to_path(save_path=handler_save_path, overwrite=True)
    else:
        handler = AugHandler.load_from_path(handler_save_path)

    return handler

def mapper(dataset_dict):
    # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    handler = load_augmentation_settings(handler_save_path = 'augmentation_settings.json')    
    image, dataset_dict = handler(image=image, dataset_dict_detectron=dataset_dict)
    
    annots = [obj for obj in dataset_dict.pop("annotations")
		if obj.get("iscrowd", 0) == 0]

    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    instances = utils.annotations_to_instances(annots, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)

    return dataset_dict