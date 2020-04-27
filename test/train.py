# save dataset configuration to config file

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
from detectron2.structures import BoxMode
import cv2
import numpy as np
from logger import logger
from common_utils.cv_drawing_utils import cv_simple_image_viewer, draw_bbox, draw_keypoints, draw_segmentation
from common_utils.common_types.keypoint import Keypoint2D_List, Keypoint2D
from common_utils.common_types.bbox import BBox
from common_utils.common_types.segmentation import Segmentation, Polygon
from common_utils.file_utils import file_exists
from imageaug import AugHandler, Augmenter as aug
from pasonatron.detectron2.util.augmentation.aug_visualizer import aug_visualizer


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
    # MetadataCatalog.get(instance_name).thing_classes = ['Measure', 'numbers']

    MetadataCatalog.get(instance_name).thing_classes = ['hsr']
    MetadataCatalog.get(instance_name).keypoint_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
    MetadataCatalog.get(instance_name).keypoint_flip_map = []
    MetadataCatalog.get(instance_name).keypoint_connection_rules = [
        ('A', 'B', (0, 0, 255)),
        ('B', 'C', (0, 0, 255)),
        ('C', 'D', (0, 0, 255)),
        ('D', 'A', (0, 0, 255)),
        ('A', 'E', (0, 0, 255)),
        ('B', 'F', (0, 0, 255)),
        ('C', 'G', (0, 0, 255)),
        ('D', 'H', (0, 0, 255)),
        ('E', 'F', (0, 0, 255)),
        ('F', 'G', (0, 0, 255)),
        ('G', 'H', (0, 0, 255)),
        ('H', 'E', (0, 0, 255)),
        ('E', 'I', (0, 0, 255)),
        ('F', 'J', (0, 0, 255)),
        ('G', 'K', (0, 0, 255)),
        ('H', 'L', (0, 0, 255)),
        ('I', 'J', (0, 0, 255)),
        ('J', 'K', (0, 0, 255)),
        ('K', 'L', (0, 0, 255)),
        ('L', 'I', (0, 0, 255))
    ]

def setup_config_file(instance_name:str, cfg):

    # # change model zoo  and weightsto use other NN type
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    # cfg.MODEL.ROI_HEADS.NAME = 'CustomROIHeads'
    # cfg.DATASETS.TRAIN = (instance_name,)
    # cfg.DATALOADER.NUM_WORKERS = 2
    # cfg.SOLVER.IMS_PER_BATCH = 2
    # cfg.SOLVER.BASE_LR = 0.003
    # cfg.SOLVER.MAX_ITER = (5050)
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (512)   # faster, and good enough for this toy dataset (default: 512)
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class
    # cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 12
    # cfg.INPUT.MIN_SIZE_TRAIN = 512

    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"))
    cfg.DATASETS.TRAIN = (instance_name,)
    cfg.DATASETS.TEST = (instance_name,)  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml")  # initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.015
    cfg.SOLVER.MAX_ITER = (
        100000
    )  # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        128
    )  # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # 2 classes
    return cfg




def load_settings(handler_save_path: str):

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

def perform_augmentation(dataset_dict, handler: AugHandler):

    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    keypoints = []
    bbox = []
    segmentation = []
    ann_instance = 0

    for item in dataset_dict["annotations"]:
        ann_instance += 1
        if "keypoints" in item:
            item["keypoints"] = Keypoint2D_List.from_numpy(np.array(item["keypoints"]))
            keypoints_num = len(item["keypoints"])
            keypoints.append(item["keypoints"])

        if "segmentation" in item:
            item["segmentation"] = Segmentation.from_list(item["segmentation"])
            segmentation.append(item["segmentation"])
        if "bbox" in item:
            item["bbox"] = BBox(xmin=item["bbox"][0], xmax=item["bbox"][0]+item["bbox"][2], ymin=item["bbox"][1], ymax=item["bbox"][1]+item["bbox"][3])
            bbox.append(item["bbox"])
            item["bbox_mode"] = BoxMode.XYXY_ABS

    if len(keypoints) != 0 and len(bbox) != 0 and len(segmentation) != 0:
        image, keypoints, bbox, poly = handler(image=image, keypoints= keypoints, bounding_boxes=bbox, polygons=segmentation)
        kpts_aug_list = keypoints[0].to_numpy(demarcation=True)[:, :2].reshape(ann_instance, keypoints_num, 2)
        kpts_aug_list = [[[x, y, 2] for x, y in kpts_aug] for kpts_aug in kpts_aug_list]
        keypoints = [Keypoint2D_List.from_list(kpts_aug, demarcation=True) for kpts_aug in kpts_aug_list]
    elif len(keypoints) == 0 and len(bbox) != 0 and len(segmentation) != 0:
        image,  bbox, poly = handler(image=image, bounding_boxes=bbox, polygons=segmentation)
    elif len(keypoints) != 0 and len(bbox) != 0 and len(segmentation) == 0:
        image, keypoints, bbox = handler(image=image, keypoints= keypoints, bounding_boxes=bbox)
        kpts_aug_list = keypoints[0].to_numpy(demarcation=True)[:, :2].reshape(ann_instance, keypoints_num, 2)
        kpts_aug_list = [[[x, y, 2] for x, y in kpts_aug] for kpts_aug in kpts_aug_list]
        keypoints = [Keypoint2D_List.from_list(kpts_aug, demarcation=True) for kpts_aug in kpts_aug_list]
    elif len(keypoints) != 0 and len(bbox) == 0 and len(segmentation) != 0:
        image, keypoints, poly = handler(image=image, keypoints= keypoints, polygons=segmentation)
        kpts_aug_list = keypoints[0].to_numpy(demarcation=True)[:, :2].reshape(ann_instance, keypoints_num, 2)
        kpts_aug_list = [[[x, y, 2] for x, y in kpts_aug] for kpts_aug in kpts_aug_list]
        keypoints = [Keypoint2D_List.from_list(kpts_aug, demarcation=True) for kpts_aug in kpts_aug_list]
    elif len(keypoints) != 0 and len(bbox) == 0 and len(segmentation) == 0:
        image, keypoints = handler(image=image, keypoints= keypoints)
        kpts_aug_list = keypoints[0].to_numpy(demarcation=True)[:, :2].reshape(ann_instance, keypoints_num, 2)
        kpts_aug_list = [[[x, y, 2] for x, y in kpts_aug] for kpts_aug in kpts_aug_list]
        keypoints = [Keypoint2D_List.from_list(kpts_aug, demarcation=True) for kpts_aug in kpts_aug_list]
    elif len(keypoints) == 0 and len(bbox) != 0 and len(segmentation) == 0:
        image, bbox = handler(image=image, bounding_boxes=bbox)
    elif len(keypoints) == 0 and len(bbox) == 0 and len(segmentation) != 0:
        image, poly = handler(image=image, polygons=segmentation)

    annots = []

    for i in range(len(dataset_dict["annotations"])):
        item = dataset_dict["annotations"][i]
        if "keypoints" in item:
            item["keypoints"] = np.asarray(keypoints[i].to_list(), dtype="float64").reshape(-1,3)
        if "bbox" in item:
            item["bbox"] = bbox[i].to_list()
        if "segmentation" in item:
            item["segmentation"] = [poly[i].to_list()]
        annots.append(item)
        

    return image, annots


from common_utils.common_types.bbox import BBox
from common_utils.common_types.segmentation import Segmentation

def mapper(dataset_dict):
    # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    handler = load_settings(handler_save_path = 'test_handler.json')
    image, annots = perform_augmentation(dataset_dict=dataset_dict, handler=handler)
    vis = image.copy()
    for annot in annots:
        bbox = BBox.from_list(annot['bbox'], input_format='pminpmax')
        vis = draw_bbox(img=vis, bbox=bbox)
        vis_kpts = annot['keypoints'][:, :2]
        vis = draw_keypoints(img=vis, keypoints=vis_kpts)
        if 'segmentation' in annot:
            seg = Segmentation.from_list(annot['segmentation'], demarcation=False)
            vis = draw_segmentation(img=vis, segmentation=seg, transparent=True)
    aug_visualizer.step(vis)

    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    instances = utils.annotations_to_instances(annots, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)

    return dataset_dict


class Test_Keypoint_Trainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.predictor = DefaultPredictor(cfg)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg=cfg, mapper=mapper)

if __name__ == "__main__":

    instance_name = "measure"
    # config_path = "scratch_config.yaml"
    dest_folder_img_combined = "/home/clayton/workspace/prj/data_keep/data/toyota/dataset/real/phone_videos/new/sampled_data/VID_20200217_161043/img"
    dest_json_file_combined = "/home/clayton/workspace/prj/data_keep/data/toyota/dataset/real/phone_videos/new/sampled_data/VID_20200217_161043/coco/output.json"

    # instance_name = "hsr"
    # dest_folder_img_combined = "./dummy_data/18_03_2020_18_03_10_coco-data"
    # dest_json_file_combined = "./dummy_data/18_03_2020_18_03_10_coco-data/HSR-coco.json"
    # datasets_dict_img_annot = {
    #     './dummy_data/18_03_2020_18_03_10_coco-data' : './dummy_data/18_03_2020_18_03_10_coco-data/HSR-coco.json',
    #     './dummy_data/31_03_2020_17_37_04_coco-data': './dummy_data/31_03_2020_17_37_04_coco-data/HSR-coco.json'
    # }

    # make_config_file_from_datasets(datasets_dict= datasets_dict_img_annot, config_path= config_path)
    # combine_dataset_from_config_file(config_path= config_path, dest_folder_img= "./combined_img", dest_json_file= "./combined.json" )
    register_dataset_to_detectron(instance_name=instance_name,img_dir_path= dest_folder_img_combined, ann_path = dest_json_file_combined)

    cfg = setup_config_file(instance_name=instance_name, cfg=get_cfg())

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = Test_Keypoint_Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

