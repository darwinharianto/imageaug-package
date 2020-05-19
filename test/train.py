# save dataset configuration to config file

from annotation_utils.coco.structs import COCO_Dataset
from annotation_utils.dataset.config.dataset_config import \
    DatasetConfigCollectionHandler, DatasetConfigCollection, DatasetConfig
from pasonatron.detectron2.util.dataset_parser import Detectron2_Annotation_Dict
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data.dataset_mapper import DatasetMapper
import cv2
import os
from tqdm import tqdm
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2 import model_zoo

from logger import logger
from annotation_utils.coco.structs import COCO_Dataset
from common_utils.file_utils import file_exists

from pasonatron.detectron2.lib.roi_heads import CustomROIHeads, ROI_HEADS_REGISTRY

import imageaug
from imageaug import AugHandler, Augmenter as aug
from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg


##### test

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
from shapely.geometry import Polygon
import shapely

def load_augmentation_settings(handler_save_path: str):

    if not file_exists(handler_save_path):
        handler = AugHandler(
            [
                aug.Crop(percent=[0.2, 0.5]),
                aug.Affine(scale = {"x": tuple([0.8, 1.2]), "y":tuple([0.8, 1.2])}, translate_percent= {"x": tuple([0, 0]), "y":tuple([0, 0])}, rotate= [0, 0], order= [0, 0], cval= [0, 0], shear= [0,0], fit_output=True)
            ]
        )
        handler.save_to_path(save_path=handler_save_path, overwrite=True)
    else:
        handler = AugHandler.load_from_path(handler_save_path)

    return handler

class MyMapper(DatasetMapper):
    def __init__(self, cfg, json_file_path, is_train=True):
        super().__init__(cfg, is_train=is_train)
        self.aug_handler = AugHandler.load_from_path(json_file_path)
    
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        ### my code ##
        image, dataset_dict = self.aug_handler(image=image, dataset_dict_detectron=dataset_dict)
        ### my code ##

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # USER: Remove if you don't use pre-computed proposals.
        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, self.min_box_side_len, self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            with PathManager.open(dataset_dict.pop("sem_seg_file_name"), "rb") as f:
                sem_seg_gt = Image.open(f)
                sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            dataset_dict["sem_seg"] = sem_seg_gt
        return dataset_dict

def make_config_file_from_datasets(datasets_dict: {}, config_path: str):
    handler = DatasetConfigCollectionHandler()

    dataset_list = []

    for key in datasets_dict:
        dataset_list.append(
             DatasetConfig(
                        img_dir=key,
                        ann_path=datasets_dict[key],
                        ann_format='coco'
            )
        )
    handler.append(
        DatasetConfigCollection(
            dataset_list
        )
    )

    handler.save_to_path(config_path, overwrite=True)


def combine_dataset_from_config_file(config_path: str, dest_folder_img: str, dest_json_file: str ):
    # combine dataset
    combined_dataset = COCO_Dataset.combine_from_config(
        config_path=config_path,
        img_sort_attr_name='file_name',
        show_pbar=True
    )
    combined_dataset.move_images(
        dst_img_dir=dest_folder_img,
        preserve_filenames=False,
        update_img_paths=True,
        overwrite=True,
        show_pbar=True
    )
    combined_dataset.save_to_path(save_path=dest_json_file, overwrite=True)

def register_dataset_to_detectron(instance_name: str,img_dir_path: str, ann_path: str):
    register_coco_instances(
        name=instance_name,
        metadata={},
        json_file=ann_path,
        image_root=img_dir_path
    )
    MetadataCatalog.get(instance_name).thing_classes = ['measure']

def setup_config_file(instance_name:str, cfg):

    # change model zoo  and weightsto use other NN type
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_1x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_1x.yaml")  # Let training initialize from model zoo
    cfg.MODEL.ROI_HEADS.NAME = 'CustomROIHeads'
    cfg.DATASETS.TRAIN = (instance_name,)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.003
    cfg.SOLVER.MAX_ITER = (10000)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (512)   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.INPUT.MIN_SIZE_TRAIN = 512

    return cfg



class Trainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
	mapper = MyMapper()
        return build_detection_train_loader(cfg, mapper=mapper)

if __name__ == "__main__":

    instance_name = "measure"
    dest_folder_img_combined = "/home/doors/workspace/darwin/from_gosar/my_real_measure_selected/img/mp_900_26_04_2020_22_18_37_coco-data"
    dest_json_file_combined = "/home/doors/workspace/darwin/from_gosar/my_real_measure_selected/img/mp_900_26_04_2020_22_18_37_coco-data/measure-only.json"

    register_dataset_to_detectron(instance_name=instance_name,img_dir_path= dest_folder_img_combined, ann_path = dest_json_file_combined)

    cfg = setup_config_file(instance_name=instance_name, cfg=get_cfg())


    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

