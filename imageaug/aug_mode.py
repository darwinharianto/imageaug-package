from __future__ import annotations
from typing import TypeVar, Generic, List
import json
import operator
import random
import imgaug as ia
import numpy as np
import imgaug.augmenters as iaa
from logger import logger
from common_utils.check_utils import check_required_keys, \
    check_file_exists, check_list_length, check_type, \
    check_type_from_list

from common_utils.cv_drawing_utils import cv_simple_image_viewer, draw_bbox, draw_keypoints, draw_segmentation
from common_utils.common_types.keypoint import Keypoint2D_List, Keypoint2D
from common_utils.common_types.bbox import BBox
from common_utils.common_types.segmentation import Segmentation, Polygon
from common_utils.file_utils import file_exists

from imgaug.augmentables.polys import PolygonsOnImage
from imgaug.augmentables.kps import KeypointsOnImage
from imgaug.augmentables.bbs import BoundingBoxesOnImage

from .base import BaseMode, BaseModeHandler
import inspect

def check_param(class_name: str, param_name: str, value, lower_limit=None, upper_limit=None):
    if (lower_limit is not None and value < lower_limit) or (upper_limit is not None and value > upper_limit):
        logger.error(f'{class_name} parameter {param_name} must be in the range [{lower_limit}, {upper_limit}].')
        logger.error(f'Encountered {param_name}={value}')
        raise Exception

def check_param_range(class_name: str, param_name: str, value: list, lower_limit=None, upper_limit=None):
    check_list_length(value, correct_length=2, ineq_type='eq')
    if value[0] > value[1]:
        logger.error(f'Encountered {class_name}.{param_name} value[0] > value[1]')
        logger.error(f'value: {value}')
        raise Exception
    check_param(class_name=class_name, param_name=f'lower {param_name}', lower_limit=lower_limit, upper_limit=upper_limit, value=value[0])
    check_param(class_name=class_name, param_name=f'upper {param_name}', lower_limit=lower_limit, upper_limit=upper_limit, value=value[1])

class Fliplr(BaseMode['Fliplr']):
    def __init__(self, p: float=0.5):
        check_param(
            class_name=self.__class__.__name__,
            param_name='p',
            lower_limit=0.0,
            upper_limit=1.0,
            value=p
        )
        self.p = p
        super().__init__(aug=iaa.Fliplr(p))

    @classmethod
    def from_dict(cls, mode_dict: dict) -> Fliplr:
        working_dict = mode_dict.copy()
        if working_dict['class_name'] != cls.__name__:
            logger.error(f"working_dict['class_name'] == {working_dict['class_name']} != {cls.__name__}")
            raise Exception
        del working_dict['class_name']
        check_required_keys(
            item_dict=working_dict,
            required_keys=['p']
        )
        return Fliplr(p=working_dict['p'])

class Flipud(BaseMode['Flipud']):
    def __init__(self, p: float=0.5):
        check_param(
            class_name=self.__class__.__name__,
            param_name='p',
            lower_limit=0.0,
            upper_limit=1.0,
            value=p
        )
        self.p = p
        super().__init__(aug=iaa.Flipud(p))

    @classmethod
    def from_dict(cls, mode_dict: dict) -> Flipud:
        working_dict = mode_dict.copy()
        if working_dict['class_name'] != cls.__name__:
            logger.error(f"working_dict['class_name'] == {working_dict['class_name']} != {cls.__name__}")
            raise Exception
        del working_dict['class_name']
        check_required_keys(
            item_dict=working_dict,
            required_keys=['p']
        )
        return Flipud(p=working_dict['p'])

class Resize(BaseMode['Resize']):
    def __init__(self, width: int=900, height: int=900):
        check_param(
            class_name=self.__class__.__name__,
            param_name='width',
            value=width,
            lower_limit=0
        )
        check_param(
            class_name=self.__class__.__name__,
            param_name='height',
            value=height,
            lower_limit=0
        )
        self.width = width
        self.height = height
        super().__init__(aug=iaa.Resize(size=(width, height)))
    
    @classmethod
    def from_dict(cls, mode_dict: dict) -> Resize:
        working_dict = mode_dict.copy()
        if working_dict['class_name'] != cls.__name__:
            logger.error(f"working_dict['class_name'] == {working_dict['class_name']} != {cls.__name__}")
            raise Exception
        del working_dict['class_name']
        check_required_keys(
            item_dict=working_dict,
            required_keys=['width', 'height']
        )
        return Resize(width=working_dict['width'], height=working_dict['height'])

class Crop(BaseMode['Crop']):
    def __init__(self, percent: List[float]=[0, 0.3]):
        check_param_range(
            class_name=self.__class__.__name__,
            param_name='percent',
            lower_limit=0.0,
            upper_limit=1.0,
            value=percent
        )
        self.percent = percent
        super().__init__(aug=iaa.Crop(percent=tuple(percent)))

    @classmethod
    def from_dict(cls, mode_dict: dict) -> Crop:
        working_dict = mode_dict.copy()
        if working_dict['class_name'] != cls.__name__:
            logger.error(f"working_dict['class_name'] == {working_dict['class_name']} != {cls.__name__}")
            raise Exception
        del working_dict['class_name']
        check_required_keys(
            item_dict=working_dict,
            required_keys=['percent']
        )
        return Crop(percent=working_dict['percent'])

class Superpixels(BaseMode['Superpixels']):
    def __init__(self, p_replace: List[float]=[0, 1.0], n_segments: List[int]=[20,200]):
        check_param_range(
            class_name=self.__class__.__name__,
            param_name='p_replace',
            lower_limit=0,
            upper_limit=1.0,
            value=p_replace
        )
        check_param_range(
            class_name=self.__class__.__name__,
            param_name='n_segments',
            lower_limit=20,
            upper_limit=200,
            value=n_segments
        )
        self.p_replace = p_replace
        self.n_segments = n_segments
        super().__init__(aug=iaa.Superpixels(p_replace=tuple(p_replace), n_segments=tuple(n_segments)))
        
    @classmethod
    def from_dict(cls, mode_dict: dict) -> Superpixels:
        working_dict = mode_dict.copy()
        if working_dict['class_name'] != cls.__name__:
            logger.error(f"working_dict['class_name'] == {working_dict['class_name']} != {cls.__name__}")
            raise Exception
        del working_dict['class_name']
        check_required_keys(
            item_dict=working_dict,
            required_keys=['n_segments', 'p_replace']
        )
        return Superpixels(p_replace=working_dict['p_replace'], n_segments=working_dict['n_segments'])

class Affine(BaseMode['Affine']):
    def __init__(self, scale: dict = {"x": tuple([0.8, 1.2]), "y":tuple([0.8, 1.2])}, translate_percent: dict = {"x": tuple([0, 0]), "y":tuple([0, 0])}, rotate: list[float] = [-180, 180], order: list[float] = [0, 1], cval: list[float] = [0, 255], shear: list[float] = [0,1]):
        if len(rotate) == 2:
            check_param_range(
                class_name=self.__class__.__name__,
                param_name='rotate',
                lower_limit=-180,
                upper_limit=180,
                value=rotate
            )
        check_param_range(
            class_name=self.__class__.__name__,
            param_name='order',
            lower_limit=0,
            upper_limit=1,
            value=order
        )
        check_param_range(
            class_name=self.__class__.__name__,
            param_name='cval',
            lower_limit=0,
            upper_limit=255,
            value=cval
        )
        check_param_range(
            class_name=self.__class__.__name__,
            param_name='shear',
            lower_limit=0,
            upper_limit=10,
            value=shear
        )
        self.scale = scale
        self.translate_percent = translate_percent
        self.rotate = rotate
        self.order = order
        self.cval = cval
        self.shear = shear
        for item in translate_percent:
            if any([item >0.2 for item in translate_percent[item]]):
                logger.yellow(f"high translation on {item} detected, object could be out of image")
        if len(rotate) == 2:
            super().__init__(aug=iaa.Affine(scale = scale, translate_percent = translate_percent, rotate = tuple(rotate), order = order, cval = tuple(cval), shear=tuple(shear)))
        else:
            super().__init__(aug=iaa.Affine(scale = scale, translate_percent = translate_percent, rotate = rotate, order = order, cval = tuple(cval), shear=tuple(shear)))

    def change_rotate_to_right_angle(self) -> Affine:
        self.rotate = [0,90,180,270]
        return Affine(scale = self.scale, translate_percent= self.translate_percent, rotate = self.rotate, order = self.order, cval=self.cval, shear=self.shear)

    @classmethod
    def from_dict(cls, mode_dict: dict) -> Affine:
        working_dict = mode_dict.copy()

        if working_dict['class_name'] != cls.__name__:
            logger.error(f"working_dict['class_name'] == {working_dict['class_name']} != {cls.__name__}")
            raise Exception
        del working_dict['class_name']
        check_required_keys(
            item_dict=working_dict,
            required_keys=['scale', 'translate_percent', 'rotate', 'order', 'cval', 'shear']
        )
        working_dict["scale"]["x"] = tuple( working_dict["scale"]["x"])
        working_dict["scale"]["y"] = tuple( working_dict["scale"]["y"])
        working_dict["translate_percent"]["x"] = tuple( working_dict["translate_percent"]["x"])
        working_dict["translate_percent"]["x"] = tuple( working_dict["translate_percent"]["x"])
        return Affine(scale = working_dict['scale'], translate_percent= working_dict['translate_percent'], rotate = working_dict['rotate'], order = working_dict['order'], cval=tuple(working_dict['cval']), shear=tuple(working_dict['shear']))

class Sharpen(BaseMode['Sharpen']):
    def __init__(self, alpha: list[float]= [0,1.0], lightness:list[float]=[0.75,1.5]):
        check_param_range(
            class_name=self.__class__.__name__,
            param_name='alpha',
            lower_limit=0,
            upper_limit=1,
            value=alpha
        )
        check_param_range(
            class_name=self.__class__.__name__,
            param_name='lightness',
            lower_limit=0.75,
            upper_limit=1.5,
            value=lightness
        )
        self.alpha = alpha
        self.lightness = lightness
        super().__init__(aug=iaa.Sharpen(alpha=tuple(alpha), lightness=tuple(lightness)))
    
    @classmethod
    def from_dict(cls, mode_dict:dict) -> Sharpen:
        working_dict = mode_dict.copy()
        if working_dict['class_name'] != cls.__name__:
            logger.error(f"working_dict['class_name'] == {working_dict['class_name']} != {cls.__name__}")
            raise Exception
        del working_dict['class_name']
        check_required_keys(
            item_dict=working_dict,
            required_keys=['alpha', 'lightness']
        )
        return Sharpen(alpha=working_dict["alpha"], lightness=working_dict["lightness"])

class Emboss(BaseMode['Emboss']):
    def __init__(self, alpha:list[float] = [0, 1.0], strength: list[float] = [0, 2.0]):
        check_param_range(
            class_name=self.__class__.__name__,
            param_name='alpha',
            lower_limit=0,
            upper_limit=1.0,
            value=alpha
        )
        check_param_range(
            class_name=self.__class__.__name__,
            param_name='strength',
            lower_limit=0,
            upper_limit=2.0,
            value=strength
        )
        self.strength = strength
        self.alpha = alpha
        super().__init__(aug=iaa.Emboss(alpha=tuple(alpha), strength=tuple(strength)))
    
    @classmethod
    def from_dict(cls, mode_dict:dict) -> Emboss:
        working_dict = mode_dict.copy()
        if working_dict['class_name'] != cls.__name__:
            logger.error(f"working_dict['class_name'] == {working_dict['class_name']} != {cls.__name__}")
            raise Exception
        del working_dict['class_name']
        check_required_keys(
            item_dict=working_dict,
            required_keys=['alpha', 'strength']
        )
        return Emboss(alpha=working_dict["alpha"], strength=working_dict["strength"])

class AdditiveGaussianNoise(BaseMode['AdditiveGaussianNoise']):
    def __init__(self, loc:float = 0, scale: list[float] = [0.0, 12.75], per_channel: float = 0.5):
        check_param(
            class_name=self.__class__.__name__,
            param_name='loc',
            lower_limit=0,
            upper_limit=1.0,
            value=loc
        )
        check_param_range(
            class_name=self.__class__.__name__,
            param_name='scale',
            lower_limit=0,
            upper_limit=15.0,
            value=scale
        )
        check_param(
            class_name=self.__class__.__name__,
            param_name='per_channel',
            lower_limit=0,
            upper_limit=1.0,
            value=per_channel
        )
        self.loc = loc
        self.scale = scale
        self.per_channel = per_channel
        super().__init__(aug=iaa.AdditiveGaussianNoise(loc=loc, scale=tuple(scale), per_channel=per_channel))
    
    @classmethod
    def from_dict(cls, mode_dict:dict) -> AdditiveGaussianNoise:
        working_dict = mode_dict.copy()
        if working_dict['class_name'] != cls.__name__:
            logger.error(f"working_dict['class_name'] == {working_dict['class_name']} != {cls.__name__}")
            raise Exception
        del working_dict['class_name']
        check_required_keys(
            item_dict=working_dict,
            required_keys=['loc', 'scale', 'per_channel']
        )
        return AdditiveGaussianNoise(loc=working_dict["loc"], scale=working_dict["scale"], per_channel=working_dict["per_channel"])

class Invert(BaseMode['Invert']):
    def __init__(self, p:float = 0, per_channel: float = 0.5):
        check_param(
            class_name=self.__class__.__name__,
            param_name='p',
            lower_limit=0,
            upper_limit=1.0,
            value=p
        )
        check_param(
            class_name=self.__class__.__name__,
            param_name='per_channel',
            lower_limit=0,
            upper_limit=1.0,
            value=per_channel
        )
        self.p = p
        self.per_channel = per_channel
        super().__init__(aug=iaa.Invert(p=p, per_channel=per_channel))
    
    @classmethod
    def from_dict(cls, mode_dict:dict) -> Invert:
        working_dict = mode_dict.copy()
        if working_dict['class_name'] != cls.__name__:
            logger.error(f"working_dict['class_name'] == {working_dict['class_name']} != {cls.__name__}")
            raise Exception
        del working_dict['class_name']
        check_required_keys(
            item_dict=working_dict,
            required_keys=['p', 'per_channel']
        )
        return Invert(p=working_dict["p"], per_channel=working_dict["per_channel"])

class Add(BaseMode['Add']):
    def __init__(self, value: list[float] = [-20,20], per_channel: float = 0.5):
        check_param_range(
            class_name=self.__class__.__name__,
            param_name='value',
            lower_limit=-20,
            upper_limit=20,
            value=value
        )
        check_param(
            class_name=self.__class__.__name__,
            param_name='per_channel',
            lower_limit=0,
            upper_limit=1.0,
            value=per_channel
        )
        self.value = value
        self.per_channel = per_channel
        super().__init__(aug=iaa.Add(value=value, per_channel=per_channel))
    
    @classmethod
    def from_dict(cls, mode_dict:dict) -> Add:
        working_dict = mode_dict.copy()
        if working_dict['class_name'] != cls.__name__:
            logger.error(f"working_dict['class_name'] == {working_dict['class_name']} != {cls.__name__}")
            raise Exception
        del working_dict['class_name']
        check_required_keys(
            item_dict=working_dict,
            required_keys=['value', 'per_channel']
        )
        return Add(value=working_dict["value"], per_channel=working_dict["per_channel"])

class Multiply(BaseMode['Multiply']):
    def __init__(self, mul: list[float] = [-20,20], per_channel: float = 0.5):
        check_param_range(
            class_name=self.__class__.__name__,
            param_name='mul',
            lower_limit=0,
            upper_limit=2,
            value=mul
        )
        check_param(
            class_name=self.__class__.__name__,
            param_name='per_channel',
            lower_limit=0,
            upper_limit=1.0,
            value=per_channel
        )
        self.mul = mul
        self.per_channel = per_channel
        super().__init__(aug=iaa.Multiply(mul= tuple(mul),per_channel=per_channel) )
    
    @classmethod
    def from_dict(cls, mode_dict:dict) -> Multiply:
        working_dict = mode_dict.copy()
        if working_dict['class_name'] != cls.__name__:
            logger.error(f"working_dict['class_name'] == {working_dict['class_name']} != {cls.__name__}")
            raise Exception
        del working_dict['class_name']
        check_required_keys(
            item_dict=working_dict,
            required_keys=['mul', 'per_channel']
        )
        return Multiply(mul=working_dict["mul"], per_channel=working_dict["per_channel"])

class LinearContrast(BaseMode['LinearContrast']):
    def __init__(self, alpha: list[float] = [0.6,1.4], per_channel: float = 0.5):
        check_param_range(
            class_name=self.__class__.__name__,
            param_name='alpha',
            lower_limit=0.6,
            upper_limit=1.4,
            value=alpha
        )
        check_param(
            class_name=self.__class__.__name__,
            param_name='per_channel',
            lower_limit=0,
            upper_limit=1.0,
            value=per_channel
        )
        self.alpha = alpha
        self.per_channel = per_channel
        super().__init__(aug=iaa.LinearContrast(alpha=tuple(alpha), per_channel=per_channel))
    
    @classmethod
    def from_dict(cls, mode_dict:dict) -> LinearContrast:
        working_dict = mode_dict.copy()
        if working_dict['class_name'] != cls.__name__:
            logger.error(f"working_dict['class_name'] == {working_dict['class_name']} != {cls.__name__}")
            raise Exception
        del working_dict['class_name']
        check_required_keys(
            item_dict=working_dict,
            required_keys=['alpha', 'per_channel']
        )
        return LinearContrast(alpha=working_dict["alpha"], per_channel=working_dict["per_channel"])

class Grayscale(BaseMode['Grayscale']):
    def __init__(self, alpha: float = 1):
        check_param(
            class_name=self.__class__.__name__,
            param_name='alpha',
            lower_limit=0,
            upper_limit=1.0,
            value=alpha
        )
        self.alpha = alpha
        super().__init__(aug=iaa.Grayscale(alpha=alpha))
    
    @classmethod
    def from_dict(cls, mode_dict:dict) -> LinearContrast:
        working_dict = mode_dict.copy()
        if working_dict['class_name'] != cls.__name__:
            logger.error(f"working_dict['class_name'] == {working_dict['class_name']} != {cls.__name__}")
            raise Exception
        del working_dict['class_name']
        check_required_keys(
            item_dict=working_dict,
            required_keys=['alpha']
        )
        return Grayscale(alpha=working_dict["alpha"])


class ElasticTransformation(BaseMode['ElasticTransformation']):
    def __init__(self, alpha: list[float] = [0,40.0], sigma: list[float] = [4.0,8.0]):
        check_param_range(
            class_name=self.__class__.__name__,
            param_name='alpha',
            lower_limit=0,
            upper_limit=40.0,
            value=alpha
        )
        check_param_range(
            class_name=self.__class__.__name__,
            param_name='sigma',
            lower_limit=4.0,
            upper_limit=8.0,
            value=sigma
        )
        self.alpha = alpha
        self.sigma = sigma
        super().__init__(aug=iaa.ElasticTransformation(alpha=tuple(alpha), sigma=tuple(sigma)))
    
    @classmethod
    def from_dict(cls, mode_dict:dict) -> ElasticTransformation:
        working_dict = mode_dict.copy()
        if working_dict['class_name'] != cls.__name__:
            logger.error(f"working_dict['class_name'] == {working_dict['class_name']} != {cls.__name__}")
            raise Exception
        del working_dict['class_name']
        check_required_keys(
            item_dict=working_dict,
            required_keys=['alpha']
        )
        return ElasticTransformation(alpha=working_dict["alpha"], sigma=working_dict["sigma"])

class PiecewiseAffine(BaseMode['PiecewiseAffine']):
    def __init__(self, scale: list[float] = [0,0.05]):
        check_param_range(
            class_name=self.__class__.__name__,
            param_name='scale',
            lower_limit=0,
            upper_limit=0.05,
            value=scale
        )
        self.scale = scale
        super().__init__(aug=iaa.PiecewiseAffine(scale=tuple(scale)))
    
    @classmethod
    def from_dict(cls, mode_dict:dict) -> PiecewiseAffine:
        working_dict = mode_dict.copy()
        if working_dict['class_name'] != cls.__name__:
            logger.error(f"working_dict['class_name'] == {working_dict['class_name']} != {cls.__name__}")
            raise Exception
        del working_dict['class_name']
        check_required_keys(
            item_dict=working_dict,
            required_keys=['scale']
        )
        return PiecewiseAffine(scale=working_dict["scale"])

class ContrastNormalization(BaseMode['ContrastNormalization']):
    def __init__(self, alpha: list[float] = [0.6,1.4], per_channel:float= 0.5):
        check_param_range(
            class_name=self.__class__.__name__,
            param_name='alpha',
            lower_limit=0.6,
            upper_limit=1.4,
            value=alpha
        )
        check_param(
            class_name=self.__class__.__name__,
            param_name='per_channel',
            lower_limit=0,
            upper_limit=1,
            value=per_channel
        )
        self.alpha = alpha
        self.per_channel = per_channel
        super().__init__(aug=iaa.contrast.LinearContrast(alpha=tuple(alpha), per_channel=per_channel))
    
    @classmethod
    def from_dict(cls, mode_dict:dict) -> ContrastNormalization:
        working_dict = mode_dict.copy()
        if working_dict['class_name'] != cls.__name__:
            logger.error(f"working_dict['class_name'] == {working_dict['class_name']} != {cls.__name__}")
            raise Exception
        del working_dict['class_name']
        check_required_keys(
            item_dict=working_dict,
            required_keys=['alpha', 'per_channel']
        )
        return ContrastNormalization(alpha=working_dict["alpha"], per_channel=working_dict["per_channel"])

class GaussianBlur(BaseMode['GaussianBlur']):
    def __init__(self, sigma: list[float] = [0.0,3.0]):
        check_param_range(
            class_name=self.__class__.__name__,
            param_name='sigma',
            lower_limit=0.0,
            upper_limit=3.0,
            value=sigma
        )
        self.sigma = sigma
        super().__init__(aug=iaa.GaussianBlur(sigma=tuple(sigma)))
    
    @classmethod
    def from_dict(cls, mode_dict:dict) -> GaussianBlur:
        working_dict = mode_dict.copy()
        if working_dict['class_name'] != cls.__name__:
            logger.error(f"working_dict['class_name'] == {working_dict['class_name']} != {cls.__name__}")
            raise Exception
        del working_dict['class_name']
        check_required_keys(
            item_dict=working_dict,
            required_keys=['sigma']
        )
        return GaussianBlur(sigma=working_dict["sigma"])

class AverageBlur(BaseMode['AverageBlur']):
    def __init__(self, k: list[int] = [1,7]):
        check_param_range(
            class_name=self.__class__.__name__,
            param_name='k',
            lower_limit=1,
            upper_limit=7,
            value=k
        )
        self.k = k
        super().__init__(aug=iaa.AverageBlur(k=tuple(k)))
    
    @classmethod
    def from_dict(cls, mode_dict:dict) -> AverageBlur:
        working_dict = mode_dict.copy()
        if working_dict['class_name'] != cls.__name__:
            logger.error(f"working_dict['class_name'] == {working_dict['class_name']} != {cls.__name__}")
            raise Exception
        del working_dict['class_name']
        check_required_keys(
            item_dict=working_dict,
            required_keys=['k']
        )
        return AverageBlur(k=working_dict["k"])

class MedianBlur(BaseMode['MedianBlur']):
    def __init__(self, k: list[int] = [1,7]):
        check_param_range(
            class_name=self.__class__.__name__,
            param_name='k',
            lower_limit=1,
            upper_limit=7,
            value=k
        )
        self.k = k
        super().__init__(aug=iaa.MedianBlur(k=tuple(k)))
    
    @classmethod
    def from_dict(cls, mode_dict:dict) -> MedianBlur:
        working_dict = mode_dict.copy()
        if working_dict['class_name'] != cls.__name__:
            logger.error(f"working_dict['class_name'] == {working_dict['class_name']} != {cls.__name__}")
            raise Exception
        del working_dict['class_name']
        check_required_keys(
            item_dict=working_dict,
            required_keys=['k']
        )
        return MedianBlur(k=working_dict["k"])

class MotionBlur(BaseMode['MotionBlur']):
    def __init__(self, k: list[int] = [3,7], angle: list[float] = [0,360]):
        check_param_range(
            class_name=self.__class__.__name__,
            param_name='k',
            lower_limit=1,
            upper_limit=7,
            value=k
        )
        check_param_range(
            class_name=self.__class__.__name__,
            param_name='angle',
            lower_limit=0,
            upper_limit=360,
            value=angle
        )
        self.k = k
        self.angle = angle
        super().__init__(aug=iaa.MotionBlur(k=tuple(k), angle=tuple(angle)))
    
    @classmethod
    def from_dict(cls, mode_dict:dict) -> MotionBlur:
        working_dict = mode_dict.copy()
        if working_dict['class_name'] != cls.__name__:
            logger.error(f"working_dict['class_name'] == {working_dict['class_name']} != {cls.__name__}")
            raise Exception
        del working_dict['class_name']
        check_required_keys(
            item_dict=working_dict,
            required_keys=['k', 'angle']
        )
        return MotionBlur(k=working_dict["k"], angle=working_dict["angle"])

class BilateralBlur(BaseMode['BilateralBlur']):
    def __init__(self, d: list[int] = [1,9], sigma_color: list[float] = [10,250], sigma_space: list[float]=[10,250]):
        check_param_range(
            class_name=self.__class__.__name__,
            param_name='d',
            lower_limit=1,
            upper_limit=9,
            value=d
        )
        check_param_range(
            class_name=self.__class__.__name__,
            param_name='sigma_color',
            lower_limit=10,
            upper_limit=250,
            value=sigma_color
        )
        check_param_range(
            class_name=self.__class__.__name__,
            param_name='sigma_space',
            lower_limit=10,
            upper_limit=250,
            value=sigma_space
        )
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        super().__init__(aug=iaa.BilateralBlur(d=tuple(d), sigma_color=tuple(sigma_color), sigma_space=tuple(sigma_space)))
    
    @classmethod
    def from_dict(cls, mode_dict:dict) -> BilateralBlur:
        working_dict = mode_dict.copy()
        if working_dict['class_name'] != cls.__name__:
            logger.error(f"working_dict['class_name'] == {working_dict['class_name']} != {cls.__name__}")
            raise Exception
        del working_dict['class_name']
        check_required_keys(
            item_dict=working_dict,
            required_keys=['sigma_color', 'd', 'sigma_space']
        )
        return BilateralBlur(d=working_dict["d"], sigma_color=working_dict["sigma_color"], sigma_space=working_dict["sigma_space"])

class EdgeDetect(BaseMode['EdgeDetect']):
    def __init__(self, alpha: list[float] = [0,0.75]):
        check_param_range(
            class_name=self.__class__.__name__,
            param_name='alpha',
            lower_limit=0,
            upper_limit=0.75,
            value=alpha
        )
        self.alpha = alpha
        super().__init__(aug=iaa.EdgeDetect(alpha=tuple(alpha)))
    
    @classmethod
    def from_dict(cls, mode_dict:dict) -> EdgeDetect:
        working_dict = mode_dict.copy()
        if working_dict['class_name'] != cls.__name__:
            logger.error(f"working_dict['class_name'] == {working_dict['class_name']} != {cls.__name__}")
            raise Exception
        del working_dict['class_name']
        check_required_keys(
            item_dict=working_dict,
            required_keys=['alpha']
        )
        return EdgeDetect(alpha=working_dict["alpha"])

class DirectedEdgeDetect(BaseMode['DirectedEdgeDetect']):
    def __init__(self, alpha: list[float] = [0,0.75], direction: list[float] = [0,1.0]):
        check_param_range(
            class_name=self.__class__.__name__,
            param_name='alpha',
            lower_limit=0,
            upper_limit=0.75,
            value=alpha
        )
        check_param_range(
            class_name=self.__class__.__name__,
            param_name='direction',
            lower_limit=0,
            upper_limit=1.0,
            value=direction
        )
        self.alpha = alpha
        self.direction = direction
        super().__init__(aug=iaa.DirectedEdgeDetect(alpha=tuple(alpha), direction=tuple(direction)))
    
    @classmethod
    def from_dict(cls, mode_dict:dict) -> DirectedEdgeDetect:
        # working_dict = super().from_dict(mode_dict = mode_dict)
        working_dict = mode_dict.copy()
        if working_dict['class_name'] != cls.__name__:
            logger.error(f"working_dict['class_name'] == {working_dict['class_name']} != {cls.__name__}")
            raise Exception
        del working_dict['class_name']
        check_required_keys(
            item_dict=working_dict,
            # required_keys=list(cls().__dict__.keys())[:-2]
            required_keys=["alpha", "direction"]
        )
        return DirectedEdgeDetect(alpha=working_dict["alpha"])

class Dropout(BaseMode['Dropout']):
    def __init__(self, p: list[float] = [0,0.05], per_channel = 0.5):
        check_param_range(
            class_name=self.__class__.__name__,
            param_name='p',
            lower_limit=0,
            upper_limit=1,
            value=p
        )
        check_param(
            class_name=self.__class__.__name__,
            param_name='per_channel',
            lower_limit=0,
            upper_limit=1.0,
            value=per_channel
        )
        self.p = p
        self.per_channel = per_channel
        super().__init__(aug=iaa.Dropout(p=tuple(p), per_channel=per_channel))
    
    @classmethod
    def from_dict(cls, mode_dict:dict) -> Dropout:
        working_dict = mode_dict.copy()
        if working_dict['class_name'] != cls.__name__:
            logger.error(f"working_dict['class_name'] == {working_dict['class_name']} != {cls.__name__}")
            raise Exception
        del working_dict['class_name']
        check_required_keys(
            item_dict=working_dict,
            required_keys=["p", "per_channel"]
        )
        return Dropout(p=working_dict["p"], per_channel=working_dict["per_channel"])

class CoarseDropout(BaseMode['CoarseDropout']):
    def __init__(self, p: list[float] = [0,0.05]):
        check_param_range(
            class_name=self.__class__.__name__,
            param_name='p',
            lower_limit=0,
            upper_limit=0.5,
            value=p
        )
        self.p = p
        super().__init__(aug=iaa.CoarseDropout(p=tuple(p)))
    
    @classmethod
    def from_dict(cls, mode_dict:dict) -> CoarseDropout:
        working_dict = mode_dict.copy()
        if working_dict['class_name'] != cls.__name__:
            logger.error(f"working_dict['class_name'] == {working_dict['class_name']} != {cls.__name__}")
            raise Exception
        del working_dict['class_name']
        check_required_keys(
            item_dict=working_dict,
            required_keys=["p"]
        )
        return CoarseDropout(p=working_dict["p"])

class AugHandler(BaseModeHandler['AugHandler', 'Any']):
    def __init__(self, aug_modes: list=None):
        super().__init__(
            obj_types=[
                Fliplr, Flipud, Resize, Crop, Superpixels,
                Affine, Sharpen, Emboss, AdditiveGaussianNoise, Invert,
                Add, Multiply, LinearContrast, Grayscale, ElasticTransformation,
                PiecewiseAffine, ContrastNormalization, GaussianBlur, AverageBlur, MedianBlur,
                MotionBlur, BilateralBlur, EdgeDetect, DirectedEdgeDetect, Dropout,
                CoarseDropout
            ],
            obj_list=aug_modes
        )
        self.aug_modes = self.obj_list

    def __call__(self, *args, **kwargs):

        if "polygons" not in kwargs:
            logger.red("polygons not found. Only rotate 90, 180, 270, 360")
            for i in range(len(self.aug_modes)) :
                items = self.aug_modes[i]
                if isinstance(items, Affine):
                    logger.red("change affine")
                    self.aug_modes[i] = self.aug_modes[i].change_rotate_to_right_angle()
        
        for k,v in kwargs.items():
            if k == "image":
                imgaug_kpts = KeypointsOnImage(keypoints=[], shape=kwargs["image"].shape)
                imgaug_bboxes = BoundingBoxesOnImage(bounding_boxes=[], shape=kwargs["image"].shape)
                imgaug_polys = PolygonsOnImage(polygons=[], shape=kwargs["image"].shape)
            if k == "keypoints":
                for item in v:
                    # keypoints_iaa = v.to_imgaug(img_shape=kwargs["image"].shape).keypoints
                    imgaug_kpts.keypoints.extend(item.to_imgaug(img_shape=kwargs["image"].shape).keypoints)
                kwargs["keypoints"] = imgaug_kpts
            if k == "polygons":
                for item in v:
                    # polygons_iaa =  v.to_imgaug(img_shape=kwargs["image"].shape).polygons
                    imgaug_polys.polygons.extend(item.to_imgaug(img_shape=kwargs["image"].shape).polygons)
                kwargs["polygons"] = imgaug_polys
            if k == "bounding_boxes":
                for item in v:
                    # bboxes_iaa = v.to_imgaug()
                    imgaug_bboxes.bounding_boxes.append(item.to_imgaug())
                kwargs["bounding_boxes"] = imgaug_bboxes
        
        for item in self.aug_modes:

            print(item)

        seq = iaa.Sequential([aug_mode.aug for aug_mode in self.aug_modes])
        a = seq(*args, **kwargs)
        image = a[0]

        for items in a:
            if isinstance(items, KeypointsOnImage):
                kpts_aug0 = Keypoint2D_List.from_imgaug(imgaug_kpts=items)
                kpts_aug_list = kpts_aug0.to_numpy(demarcation=True)[:, :2].reshape(1, len(kpts_aug0), 2)
                kpts_aug_list = [[[x, y, 2] for x, y in kpts_aug] for kpts_aug in kpts_aug_list]
                kpts_aug_list = [Keypoint2D_List.from_list(kpts_aug, demarcation=True) for kpts_aug in kpts_aug_list]
            if isinstance(items, BoundingBoxesOnImage):
                bbox_aug_list = [BBox.from_imgaug(bbox_aug) for bbox_aug in items.bounding_boxes]
            if isinstance(items, PolygonsOnImage):
                poly_aug_list = [Polygon.from_imgaug(imgaug_polygon) for imgaug_polygon in items.polygons]
                bbox_aug_list_from_poly = [poly_aug.to_bbox() for poly_aug in poly_aug_list]
                # Adjust BBoxes when Segmentation BBox does not contain all keypoints
                # TODO need to consider this method
                # for i in range(len(bbox_aug_list_from_poly)):
                #     kpt_points_aug = [kpt_aug.point for kpt_aug in kpts_aug_list[i]]
                #     kpt_points_aug_contained = [kpt_point_aug.within(bbox_aug_list_from_poly[i]) for kpt_point_aug in kpt_points_aug]
                #     if not np.any(np.array(kpt_points_aug_contained)):
                #         logger.error(f"Keypoints not contained in corresponding bbox.")
                #     else:
                #         if not np.all(np.array(kpt_points_aug_contained)):
                #             kpt_points_aug_arr = np.array([kpt_point_aug.to_list() for kpt_point_aug in kpt_points_aug])
                #             kpt_points_aug_arr = kpt_points_aug_arr[~np.all(kpt_points_aug_arr == 0, axis=1)]
                #             kpt_xmin, kpt_ymin = np.min(kpt_points_aug_arr, axis=0).tolist()
                #             kpt_xmax, kpt_ymax = np.max(kpt_points_aug_arr, axis=0).tolist()
                #             bbox_aug_list_from_poly[i].xmin = kpt_xmin if kpt_xmin < bbox_aug_list_from_poly[i].xmin and kpt_xmin != 0 else bbox_aug_list_from_poly[i].xmin
                #             bbox_aug_list_from_poly[i].ymin = kpt_ymin if kpt_ymin < bbox_aug_list_from_poly[i].ymin and kpt_xmin != 0 else bbox_aug_list_from_poly[i].ymin
                #             bbox_aug_list_from_poly[i].xmax = kpt_xmax if kpt_xmax > bbox_aug_list_from_poly[i].xmax and kpt_xmin != 0 else bbox_aug_list_from_poly[i].xmax
                #             bbox_aug_list_from_poly[i].ymax = kpt_ymax if kpt_ymax > bbox_aug_list_from_poly[i].ymax and kpt_xmin != 0 else bbox_aug_list_from_poly[i].ymax
                #         break

                seg_aug_list = [Segmentation([poly_aug]) for poly_aug in poly_aug_list]

        if 'bbox_aug_list_from_poly' in locals():
            if 'bbox_aug_list' in locals():
                bbox_aug_list = bbox_aug_list_from_poly

        if 'poly_aug_list' in locals():
            a = (image, bbox_aug_list, poly_aug_list)
            if 'kpts_aug_list' in locals():
                a = (image, kpts_aug_list, bbox_aug_list, poly_aug_list)
        elif 'bbox_aug_list' in locals():
            a = (image, bbox_aug_list)
            if 'kpts_aug_list' in locals():
                a = (image, kpts_aug_list, bbox_aug_list)

        return a

    @classmethod
    def from_dict_list(cls, dict_list: List[dict]) -> AugHandler:
        aug_modes = []
        for dict_item in dict_list:
            if dict_item['class_name'] == 'Fliplr':
                aug_modes.append(Fliplr.from_dict(dict_item))
            elif dict_item['class_name'] == 'Flipud':
                aug_modes.append(Flipud.from_dict(dict_item))
            elif dict_item['class_name'] == 'Crop':
                aug_modes.append(Crop.from_dict(dict_item))
            elif dict_item['class_name'] == 'Superpixels':
                aug_modes.append(Superpixels.from_dict(dict_item))
            elif dict_item['class_name'] == 'Affine':
                aug_modes.append(Affine.from_dict(dict_item))
            elif dict_item['class_name'] == 'Sharpen':
                aug_modes.append(Sharpen.from_dict(dict_item))
            elif dict_item['class_name'] == 'Emboss':
                aug_modes.append(Emboss.from_dict(dict_item))
            elif dict_item['class_name'] == 'AdditiveGaussianNoise':
                aug_modes.append(AdditiveGaussianNoise.from_dict(dict_item))
            elif dict_item['class_name'] == 'Invert':
                aug_modes.append(Invert.from_dict(dict_item))
            elif dict_item['class_name'] == 'Add':
                aug_modes.append(Add.from_dict(dict_item))
            elif dict_item['class_name'] == 'Multiply':
                aug_modes.append(Multiply.from_dict(dict_item))
            elif dict_item['class_name'] == 'LinearContrast':
                aug_modes.append(LinearContrast.from_dict(dict_item))
            elif dict_item['class_name'] == 'Grayscale':
                aug_modes.append(Grayscale.from_dict(dict_item))
            elif dict_item['class_name'] == 'ElasticTransformation':
                aug_modes.append(ElasticTransformation.from_dict(dict_item))
            elif dict_item['class_name'] == 'PiecewiseAffine':
                aug_modes.append(PiecewiseAffine.from_dict(dict_item))
            elif dict_item['class_name'] == 'ContrastNormalization':
                aug_modes.append(ContrastNormalization.from_dict(dict_item))
            elif dict_item['class_name'] == 'GaussianBlur':
                aug_modes.append(GaussianBlur.from_dict(dict_item))
            elif dict_item['class_name'] == 'AverageBlur':
                aug_modes.append(AverageBlur.from_dict(dict_item))
            elif dict_item['class_name'] == 'BilateralBlur':
                aug_modes.append(BilateralBlur.from_dict(dict_item))
            elif dict_item['class_name'] == 'MedianBlur':
                aug_modes.append(MedianBlur.from_dict(dict_item))
            elif dict_item['class_name'] == 'MotionBlur':
                aug_modes.append(MotionBlur.from_dict(dict_item))
            elif dict_item['class_name'] == 'EdgeDetect':
                aug_modes.append(EdgeDetect.from_dict(dict_item))
            elif dict_item['class_name'] == 'DirectedEdgeDetect':
                aug_modes.append(DirectedEdgeDetect.from_dict(dict_item))
            elif dict_item['class_name'] == 'Dropout':
                aug_modes.append(Dropout.from_dict(dict_item))
            elif dict_item['class_name'] == 'CoarseDropout':
                aug_modes.append(CoarseDropout.from_dict(dict_item))
            elif dict_item['class_name'] == 'Resize':
                aug_modes.append(Resize.from_dict(dict_item))
            else:
                logger.error(f"Invalid class_name: {dict_item['class_name']}")
                raise Exception

        return AugHandler(aug_modes=aug_modes)

    @classmethod
    def load_from_path(cls, json_path: str) -> AugHandler:
        check_file_exists(json_path)
        json_data = json.load(open(json_path, 'r'))
        return AugHandler.from_dict_list(json_data)

class Augmenter:
    Fliplr = Fliplr
    Flipud = Flipud
    Resize = Resize
    Crop = Crop
    Superpixels = Superpixels
    Affine = Affine
    Sharpen = Sharpen
    Emboss = Emboss
    AdditiveGaussianNoise = AdditiveGaussianNoise
    Invert = Invert
    Add = Add
    Multiply = Multiply
    LinearContrast = LinearContrast
    Grayscale = Grayscale
    ElasticTransformation = ElasticTransformation
    PiecewiseAffine = PiecewiseAffine
    ContrastNormalization = ContrastNormalization
    GaussianBlur = GaussianBlur
    AverageBlur = AverageBlur
    MedianBlur = MedianBlur
    MotionBlur = MotionBlur
    BilateralBlur = BilateralBlur
    EdgeDetect = EdgeDetect
    DirectedEdgeDetect = DirectedEdgeDetect
    Dropout = Dropout
    CoarseDropout = CoarseDropout
