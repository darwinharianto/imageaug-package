import cv2
from annotation_utils.coco.structs import COCO_Dataset
from imageaug.aug_mode import AugHandler, Augmenter
from common_utils.cv_drawing_utils import draw_bbox, draw_keypoints, draw_segmentation, draw_skeleton, cv_simple_image_viewer
from common_utils.common_types.keypoint import Keypoint2D_List
from common_utils.constants.color_constants import Color
from common_utils.file_utils import make_dir_if_not_exists, delete_all_files_in_dir
from logger import logger

output_dir = 'comparison'
make_dir_if_not_exists(output_dir)
delete_all_files_in_dir(output_dir)

handler = AugHandler(
    aug_modes=[
        Augmenter.AverageBlur(k=[4, 6], frequency=1.0),
        Augmenter.Crop(percent=[0.1, 0.4]),
        Augmenter.Affine(),
        Augmenter.DirectedEdgeDetect()
    ]
)

dataset_dir = '/home/clayton/workspace/prj/data_keep/data/toyota/dataset/sim/20200214/1/coco-data'
dataset = COCO_Dataset.load_from_path(
    json_path=f'{dataset_dir}/fixed_HSR-coco.json',
    img_dir=dataset_dir
)

color_list = [Color.RED1, Color.GREEN, Color.BLUE, Color.PINK, Color.ORANGE, Color.YELLOW1, Color.PURPLE]
color_list = [color.bgr for color in color_list]

img_buffer = []
for frame_i, coco_image in enumerate(dataset.images):
    logger.info(f'coco_image.file_name: {coco_image.file_name}')
    if frame_i < 0:
        continue
    img = cv2.imread(coco_image.coco_url)
    img_h, img_w = img.shape[:2]
    orig_vis = img.copy()
    img_buffer.append(img)

    bbox_list = []
    seg_list = []
    kpts_list = []

    anns = dataset.annotations.get_annotations_from_imgIds([coco_image.id])
    for i, ann in enumerate(anns):
        bbox_list.append(ann.bbox)
        seg_list.append(ann.segmentation)
        kpts_list.append(ann.keypoints)
        orig_vis = dataset.draw_annotation(
            img=orig_vis, ann_id=ann.id, kpt_idx_offset=-1,
            bbox_color=color_list[i], bbox_label_color=color_list[i],
            seg_color=color_list[i], seg_transparent=True,
            kpt_color=color_list[i], kpt_label_color=color_list[i],
            skeleton_color=color_list[i]
        )

    logger.purple(f'len(bbox_list): {len(bbox_list)}')
    logger.purple(f'len(kpts_list): {len(kpts_list)}')
    logger.purple(f'len(seg_list): {len(seg_list)}')
    assert len(bbox_list) == len(kpts_list) and len(kpts_list) == len(seg_list)

    aug_img, aug_kpts_list, aug_bbox_list, aug_seg_list = handler(
        image=img, bounding_boxes=bbox_list, segmentations=seg_list, keypoints=kpts_list
    )
    logger.purple(f'len(aug_bbox_list): {len(aug_bbox_list)}')
    logger.purple(f'len(aug_kpts_list): {len(aug_kpts_list)}')
    logger.purple(f'len(aug_seg_list): {len(aug_seg_list)}')
    # assert len(aug_bbox_list) == len(aug_kpts_list) and len(aug_kpts_list) == len(aug_seg_list)
    count = 0

    for aug_bbox, aug_kpts, aug_seg in zip(aug_bbox_list, aug_kpts_list, aug_seg_list):
    # for aug_bbox, aug_seg in zip(aug_bbox_list, aug_seg_list):
        aug_img = draw_bbox(
            img=aug_img, bbox=aug_bbox, color=color_list[count], text=f'{count}', label_color=color_list[count]
        )
        aug_img = draw_segmentation(
            img=aug_img, segmentation=aug_seg, transparent=True, color=color_list[count]
        )
        aug_img = draw_skeleton(
            img=aug_img, keypoints=aug_kpts.to_numpy(demarcation=True)[:, :2].tolist(),
            keypoint_skeleton=dataset.categories.get_unique_category_from_name(name='hsr').skeleton,
            index_offset=-1, color=color_list[count]
        )
        aug_kpts = Keypoint2D_List.buffer(aug_kpts)
        aug_img = draw_keypoints(
            img=aug_img, keypoints=aug_kpts.to_numpy(demarcation=True)[:, :2].tolist(), color=color_list[count]
        )
        count += 1
    
    aug_img_h, aug_img_w = aug_img.shape[:2]
    target_width = int((img_h / aug_img_h) * aug_img_w)
    aug_img = cv2.resize(aug_img, dsize=(target_width, img_h))
    comparison_img = cv2.hconcat([orig_vis, aug_img])

    cv2.imwrite(f'{output_dir}/{coco_image.file_name}', comparison_img)

    # quit_flag = cv_simple_image_viewer(img=comparison_img, preview_width=1800)
    # if quit_flag:
    #     break