from annotation_utils.coco.refactored.structs import COCO_Dataset
from logger import logger
from common_utils.image_utils import concat_n_images
from common_utils.cv_drawing_utils import cv_simple_image_viewer
from common_utils.file_utils import file_exists
import cv2
from common_utils.common_types.keypoint import Keypoint2D_List, Keypoint2D


from imageaug import AugHandler, Augmenter as aug

dataset = COCO_Dataset.load_from_path(
    json_path='/Users/darwinharianto/Desktop/hayashida/Unreal/18_03_2020_18_03_10_coco-data/HSR-coco.json',
    img_dir='/Users/darwinharianto/Desktop/hayashida/Unreal/18_03_2020_18_03_10_coco-data'
)

resize_save_path = 'test_resize.json'
handler_save_path = 'test_handler.json'
if not file_exists(resize_save_path):
    resize = aug.Resize(width=500, height=500)
    resize.save_to_path(save_path=resize_save_path, overwrite=True)
    logger.info(f'Created new Resize save.')
else:
    resize = aug.Resize.load_from_path(resize_save_path)
    logger.info(f'Loaded Resize from save.')
if not file_exists(handler_save_path):
    handler = AugHandler(
        [
            aug.Crop(percent=[0.2, 0.5]),
            aug.Flipud(p=0.5),
            aug.Superpixels()
            # aug.Sharpen(alpha=[-1,0.1], lightness=[0,3])
        ]
    )
    handler.save_to_path(save_path=handler_save_path, overwrite=True)
    logger.info(f'Created new AugHandler save.')
else:
    handler = AugHandler.load_from_path(handler_save_path)
    handler = AugHandler(aug_modes=
        [
            # aug.Affine(scale = {"x": tuple([0.8, 1.2]), "y":tuple([0.8, 1.2])}, translate_percent= {"x": tuple([0.1, 0.11]), "y":tuple([0.1, 0.11])}, rotate= [-180, 180], order= [0, 0], cval= [0, 0], shear= [0,0]),
            # aug.Crop(percent=[0.2, 0.5]),
            # aug.Flipud(p=0.5),
            # aug.Superpixels(),
            # aug.Sharpen(alpha=[0,0.1], lightness=[0.8,1]),
            # aug.Emboss(alpha=[0,0.1], strength=[0.8,1]),
            # aug.AdditiveGaussianNoise(),
            # aug.Invert(p=1, per_channel=False),
            # aug.Add(value=[-20,20], per_channel=True),
            # aug.LinearContrast(alpha=[0.6,1.4], per_channel=True),
            # aug.Grayscale(alpha=0.8),
            # aug.Multiply(mul=[0.8,1.2], per_channel=False),
            # aug.ElasticTransformation(alpha=[0,40], sigma=[4,6]),
            aug.PiecewiseAffine(scale=[0.0,0.05]),
            # aug.ContrastNormalization(alpha=[0.7,1], per_channel=True),
            # aug.AverageBlur(k=[1,7]),
            # aug.MotionBlur(k=[3,7], angle=[0,360]),
            # aug.BilateralBlur(d=[1,9]),
            # aug.EdgeDetect(alpha=[0,0.5]),
            # aug.DirectedEdgeDetect(alpha=[0,0.5], direction=[0,1]),
            # aug.Dropout(p=[0,0.05], per_channel=False),
            # aug.CoarseDropout(p=[0,0.5]),
            aug.Resize(),
            aug.Grayscale(alpha=0.9, frequency=0.1),
            aug.BilateralBlur(d=[1,2])
        ]
    )
    handler.save_to_path(save_path=handler_save_path, overwrite=True)
    logger.info(f'Loaded AugHandler from save.')

img_buffer = []
for coco_image in dataset.images:
    img = cv2.imread(coco_image.coco_url)
    img_buffer.append(img)

    keypoints = []
    bbox = []
    segmentation = []
    ann_instance = 0

    for item in dataset.annotations:
        if item.image_id == coco_image.id:
            keypoints_num = len(item.keypoints)
            ann_instance += 1

            keypoints.append(item.keypoints)
            bbox.append(item.bbox)
            segmentation.append(item.segmentation)


    image, keypoints, bbox, poly = handler(image=img, keypoints= keypoints, bounding_boxes=bbox, polygons=segmentation)
    kpts_aug_list = keypoints[0].to_numpy(demarcation=True)[:, :2].reshape(ann_instance, keypoints_num, 2)
    kpts_aug_list = [[[x, y, 2] for x, y in kpts_aug] for kpts_aug in kpts_aug_list]
    keypoints = [Keypoint2D_List.from_list(kpts_aug, demarcation=True) for kpts_aug in kpts_aug_list]

    # print(image, keypoints, bbox, poly)
    cv2.imshow("a", image)
    cv2.waitKey(5000)
    break

        # print(results)

    #     preview0 = concat_n_images(img_list=resize(images=img_buffer), orientation=0)
    #     preview1 = concat_n_images(img_list=resize(images=results), orientation=0)
    #     preview = concat_n_images(img_list=[preview0, preview1], orientation=1)
    #     img_buffer = []
    #     quit_flag = cv_simple_image_viewer(img=preview, preview_width=1000)
    #     if quit_flag:
    #         break
