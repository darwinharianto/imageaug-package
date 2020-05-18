# save dataset configuration to config file

from annotation_utils.coco.structs import COCO_Dataset
from annotation_utils.dataset.config.dataset_config import \
    DatasetConfigCollectionHandler, DatasetConfigCollection, DatasetConfig
from pasonatron.detectron2.util.dataset_parser import Detectron2_Annotation_Dict
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
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
# from pasonatron.detectron2.lib.trainer import COCO_Keypoint_Trainer

# from augmented_loader import mapper # Change this import path
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

def mapper(dataset_dict):
    # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    handler = load_augmentation_settings(handler_save_path = 'augmentation_settings.json')  

    ############################################################################################
    ################################ this process need more review #############################
    ############################################################################################
    for i in range(len(dataset_dict["annotations"])):
        item = dataset_dict["annotations"][i]
        if "segmentation" in item:
            if len(item["segmentation"]) != 0:
                for j in range(len(item["segmentation"])):
                    segmentation_list = item["segmentation"][j]
                    if len(segmentation_list) < 5:
                        logger.red(f"segmentation has abnormal point count {segmentation_list}" )
                        continue 
                    segmentation_list = np.array(segmentation_list).reshape(-1,2)
                    poly_shapely = Polygon(segmentation_list)
                    poly_list = list(zip(*poly_shapely.exterior.coords.xy))
                    line_non_simple = shapely.geometry.LineString(poly_list)
                    mls = shapely.ops.unary_union(line_non_simple)
                    polygons = list(shapely.ops.polygonize(mls))
                    if len(polygons) == 0:
                        item["segmentation"][j] = segmentation_list
                    else:
                        shapely_polygon = Polygon(polygons[0])
                        vals_tuple = shapely_polygon.exterior.coords.xy
                        numpy_array = np.array(vals_tuple).T[:-1]
                        flattened_list = numpy_array.reshape(-1).tolist()
                        item["segmentation"][j] = flattened_list
                dataset_dict["annotations"][i]["segmentation"] = item["segmentation"]
    ##############################################################################################

    seg_count = 0
    bbox_count = 0
    for item in dataset_dict["annotations"]:
        if "segmentation" in item:
            seg_count +=1
        if "bbox" in item:
            bbox_count +=1
        if bbox_count != seg_count:
            logger.red(f"bbox and seg = {seg_count}, {bbox_count}")
            print(dataset_dict["file_name"])

            


    image, dataset_dict = handler(image=image, dataset_dict_detectron=dataset_dict)
    
    annots = [obj for obj in dataset_dict.pop("annotations")
		if obj.get("iscrowd", 0) == 0]

    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    instances = utils.annotations_to_instances(annots, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)

    if True:
        vis_img = image.copy()
        bbox_list = [BBox.from_list(vals) for vals in dataset_dict["instances"].gt_boxes.tensor.numpy().tolist()]
        seg_list = [Segmentation([Polygon.from_list(poly.tolist(), demarcation=False) for poly in seg_polys]) for seg_polys in dataset_dict["instances"].gt_masks.polygons]
        for bbox, seg in zip(bbox_list, seg_list):
            if len(seg) > 0 and False:
                vis_img = draw_segmentation(img=vis_img, segmentation=seg, transparent=True)
            vis_img = draw_bbox(img=vis_img, bbox=bbox)
        aug_visualizer.step(vis_img)

    return dataset_dict


###### test end

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
    # combined_dataset.display_preview(kpt_idx_offset=-1, start_idx=0)

def register_dataset_to_detectron(instance_name: str,img_dir_path: str, ann_path: str):
    register_coco_instances(
        name=instance_name,
        metadata={},
        json_file=ann_path,
        image_root=img_dir_path
    )
    MetadataCatalog.get(instance_name).thing_classes = ['measure']
    # MetadataCatalog.get(instance_name).keypoint_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
    # MetadataCatalog.get(instance_name).keypoint_flip_map = []
    # MetadataCatalog.get(instance_name).keypoint_connection_rules = [
    #     ('A', 'B', (0, 0, 255)),
    #     ('B', 'C', (0, 0, 255)),
    #     ('C', 'D', (0, 0, 255)),
    #     ('D', 'A', (0, 0, 255)),
    #     ('A', 'E', (0, 0, 255)),
    #     ('B', 'F', (0, 0, 255)),
    #     ('C', 'G', (0, 0, 255)),
    #     ('D', 'H', (0, 0, 255)),
    #     ('E', 'F', (0, 0, 255)),
    #     ('F', 'G', (0, 0, 255)),
    #     ('G', 'H', (0, 0, 255)),
    #     ('H', 'E', (0, 0, 255)),
    #     ('E', 'I', (0, 0, 255)),
    #     ('F', 'J', (0, 0, 255)),
    #     ('G', 'K', (0, 0, 255)),
    #     ('H', 'L', (0, 0, 255)),
    #     ('I', 'J', (0, 0, 255)),
    #     ('J', 'K', (0, 0, 255)),
    #     ('K', 'L', (0, 0, 255)),
    #     ('L', 'I', (0, 0, 255))
    # ]

def setup_config_file(instance_name:str, cfg):

    # # change model zoo  and weightsto use other NN type
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml"))
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml")  # Let training initialize from model zoo
    # cfg.MODEL.ROI_HEADS.NAME = 'CustomROIHeads'
    # cfg.DATASETS.TRAIN = (instance_name,)
    # cfg.DATALOADER.NUM_WORKERS = 2
    # cfg.SOLVER.IMS_PER_BATCH = 2
    # cfg.SOLVER.BASE_LR = 0.003
    # cfg.SOLVER.MAX_ITER = (10000)
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (512)   # faster, and good enough for this toy dataset (default: 512)
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class
    # cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 12
    # cfg.INPUT.MIN_SIZE_TRAIN = 512


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
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    # cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 12
    cfg.INPUT.MIN_SIZE_TRAIN = 512

    return cfg



class Trainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=mapper)

if __name__ == "__main__":

    instance_name = "measure"
    # config_path = "scratch_config.yaml"
    dest_folder_img_combined = "/home/doors/workspace/darwin/from_gosar/my_real_measure_selected/img/mp_900_26_04_2020_22_18_37_coco-data"
    dest_json_file_combined = "/home/doors/workspace/darwin/from_gosar/my_real_measure_selected/img/mp_900_26_04_2020_22_18_37_coco-data/measure-only.json"

    # datasets_dict_img_annot = {
    #     './dummy_data/18_03_2020_18_03_10_coco-data' : './dummy_data/18_03_2020_18_03_10_coco-data/HSR-coco.json',
    #     './dummy_data/31_03_2020_17_37_04_coco-data': './dummy_data/31_03_2020_17_37_04_coco-data/HSR-coco.json'
    # }

    # make_config_file_from_datasets(datasets_dict= datasets_dict_img_annot, config_path= config_path)
    # combine_dataset_from_config_file(config_path= config_path, dest_folder_img= "./combined_img", dest_json_file= "./combined.json" )
    register_dataset_to_detectron(instance_name=instance_name,img_dir_path= dest_folder_img_combined, ann_path = dest_json_file_combined)

    cfg = setup_config_file(instance_name=instance_name, cfg=get_cfg())


    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    # trainer.resume_or_load(resume=True)

