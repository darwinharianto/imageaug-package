from annotation_utils.coco.refactored.structs import COCO_Dataset
from annotation_utils.dataset.refactored.config.dataset_config import \
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
from annotation_utils.coco.refactored.structs import COCO_Dataset
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



class Test_Keypoint_Trainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.predictor = DefaultPredictor(cfg)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg=cfg, mapper=mapper)

def load_augmentation_settings(handler_save_path: str):

    if not file_exists(handler_save_path):
        handler = AugHandler(
            [
                aug.Crop(percent=[0.2, 0.5]),
            ]
        )
        handler.save_to_path(save_path=handler_save_path, overwrite=True)
    else:
        handler = AugHandler.load_from_path(handler_save_path)

    return handler


def register_dataset_to_detectron(instance_name: str,img_dir_path: str, ann_path: str):
    register_coco_instances(
        name=instance_name,
        metadata={},
        json_file=ann_path,
        image_root=img_dir_path
    )
    MetadataCatalog.get(instance_name).thing_classes = ['Measure', 'numbers']


def mapper(dataset_dict):
    # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    handler = load_augmentation_settings(handler_save_path = 'test_handler.json')    
    image, dataset_dict = handler(image=image, dataset_dict_detectron=dataset_dict)
    
    annots = [obj for obj in dataset_dict.pop("annotations")
		if obj.get("iscrowd", 0) == 0]

    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    instances = utils.annotations_to_instances(annots, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)

    return dataset_dict

def setup_config_file(instance_name:str, cfg, detectron_model:str, max_iter: int = 300, base_lr:float = 0.015, num_workers: int = 2, ims_per_batch: int = 2, BATCH_SIZE_PER_IMAGE: int = 128):


    cfg.merge_from_file(model_zoo.get_config_file(detectron_model))
    cfg.DATASETS.TRAIN = (instance_name,)
    cfg.DATASETS.TEST = (instance_name,)  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = num_workers
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(detectron_model)  # initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.MAX_ITER = (
        max_iter
    )  # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        BATCH_SIZE_PER_IMAGE
    )  # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MetadataCatalog.get(instance_name).thing_classes)  
    return cfg

if __name__ == "__main__":

    instance_name = "measure"
    dest_folder_img_combined = "../../../coco_data_measure_random_color"
    dest_json_file_combined = "../../../coco_data_measure_random_color/HSR-coco.json"

    register_dataset_to_detectron(instance_name=instance_name,img_dir_path= dest_folder_img_combined, ann_path = dest_json_file_combined)

    cfg = setup_config_file(instance_name=instance_name, cfg=get_cfg(), detectron_model="COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = Test_Keypoint_Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

