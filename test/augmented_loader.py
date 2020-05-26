import copy
import torch
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
from imgaug.augmentables.polys import PolygonsOnImage
from imgaug.augmentables.kps import KeypointsOnImage
from imgaug.augmentables.bbs import BoundingBoxesOnImage
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.structures import BoxMode
import cv2

from logger import logger
from common_utils.cv_drawing_utils import cv_simple_image_viewer, draw_bbox, draw_keypoints, draw_segmentation
from common_utils.common_types.keypoint import Keypoint2D_List, Keypoint2D
from common_utils.common_types.bbox import BBox
from common_utils.common_types.segmentation import Segmentation, Polygon
from common_utils.file_utils import file_exists
from imageaug import AugHandler, Augmenter as aug

from aug_visualizer import aug_visualizer
# from ..dataset_parser import Detectron2_Annotation_Dict


def load_settings(handler_save_path: str):

    handler_save_path = 'test_handler.json'
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
    elif len(keypoints) != 0 and len(bbox) != 0 and len(segmentation) == 0:
        image, keypoints, bbox = handler(image=image, keypoints= keypoints, bounding_boxes=bbox)
    elif len(keypoints) != 0 and len(bbox) == 0 and len(segmentation) != 0:
        image, keypoints, poly = handler(image=image, keypoints= keypoints, polygons=segmentation)
    elif len(keypoints) != 0 and len(bbox) == 0 and len(segmentation) == 0:
        image, keypoints = handler(image=image, keypoints= keypoints)
    elif len(keypoints) == 0 and len(bbox) != 0 and len(segmentation) != 0:
        image, bbox, poly = handler(image=image, bounding_boxes=bbox, polygons=segmentation)
    elif len(keypoints) == 0 and len(bbox) != 0 and len(segmentation) == 0:
        image, bbox = handler(image=image, bounding_boxes=bbox)
    elif len(keypoints) == 0 and len(bbox) == 0 and len(segmentation) != 0:
        image, poly = handler(image=image, polygons=segmentation)

    if "keypoints" in locals() and len(keypoints) != 0:
        kpts_aug_list = keypoints[0].to_numpy(demarcation=True)[:, :2].reshape(ann_instance, keypoints_num, 2)
        kpts_aug_list = [[[x, y, 2] for x, y in kpts_aug] for kpts_aug in kpts_aug_list]
        keypoints = [Keypoint2D_List.from_list(kpts_aug, demarcation=True) for kpts_aug in kpts_aug_list]

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



def mapper(dataset_dict):
    # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    handler = load_settings(handler_save_path = 'test_handler.json')

    
    # image, annots = perform_augmentation(dataset_dict=dataset_dict, handler=handler)
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
        # kpts_list = [Keypoint2D_List.from_numpy(arr, demarcation=True) for arr in dataset_dict["instances"].gt_keypoints.tensor.numpy()]
        # print(bbox_list, seg_list, kpts_list)
        # for bbox, seg, kpts in zip(bbox_list, seg_list, kpts_list):
        # for bbox, kpts in zip(bbox_list, kpts_list):
        for bbox, seg in zip(bbox_list, seg_list):
            if len(seg) > 0 and False:
                vis_img = draw_segmentation(img=vis_img, segmentation=seg, transparent=True)
            vis_img = draw_bbox(img=vis_img, bbox=bbox)
            # vis_img = draw_keypoints(img=vis_img, keypoints=kpts.to_numpy(demarcation=True)[:, :2].tolist(), radius=1)
        cv2.imshow("a", vis_img)
        cv2.waitKey(100)
        # aug_visualizer.step(vis_img)

    return dataset_dict



if __name__ == "__main__":
    import json
    with open('/Users/darwinharianto/Desktop/hayashida/Unreal/18_03_2020_18_03_10_coco-data/HSR-detectron.json') as f:
        dataset_dicts = json.loads(f.read())
    for dataset_dict in dataset_dicts:
        for item in dataset_dict["annotations"]:
            del item["keypoints"]
            # del item["segmentation"]
        dataset_dict = mapper(dataset_dict)
    # dataset_dict = mapper(dataset_dict)
    # data_loader = build_detection_train_loader(cfg, mapper=mapper)
