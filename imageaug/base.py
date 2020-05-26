from __future__ import annotations
from typing import TypeVar, Generic, List
import json
import random
import imgaug.augmenters as iaa
from logger import logger
from common_utils.check_utils import check_list_length, \
    check_type, check_type_from_list, check_file_exists, \
    check_required_keys
from common_utils.file_utils import file_exists


T = TypeVar('T')
H = TypeVar('H')

# sample to raise error using meta class if a method not found in child class
class BaseModeMeta(type):
    def __new__(cls, name, bases, body):
        if name != 'BaseMode' and not 'from_dict' in body:
            raise TypeError(f"Please overwrite method from_dict")
        return super().__new__(cls,name,bases,body)

class BaseMode(Generic[T], metaclass= BaseModeMeta):
    def __init__(self, aug: iaa.Augmenter, frequency: float = None):
        if frequency is not None:
            if frequency < 0 or frequency > 1:
                raise TypeError(f"frequency should be from 0 to 1")
        self.aug = aug
        self.class_name = self.__class__.__name__
        self.frequency = frequency
        
        
        
    # sample to raise error when init subclass
    def __init_subclass__(self, *a, **kw):
        if 'from_dict' not in dir(self):
            raise TypeError(f"Please overwrite method from_dict to {self.__name__} class")
        return super().__init_subclass__(*a, **kw)

    def _get_param_dict(self: T) -> dict:
        param_dict = self.__dict__.copy()
        del param_dict['aug']
        del param_dict['class_name']
        return param_dict

    def __str__(self: T) -> str:
        param_dict = self._get_param_dict()
        param_str_list = [f'{key}={value}' for key, value in param_dict.items()]
        param_str = ', '.join(param_str_list)
        return f'{self.class_name}({param_str})'
    
    def __repr__(self: T) -> str:
        return self.__str__()

    def __call__(self: T, *args, **kwargs):
        return self.aug(*args, **kwargs)

    @classmethod
    def buffer(cls: T, obj) -> T:
        return obj

    def copy(self: T) -> T:
        if type(self) is BaseMode:
            return BaseMode(aug=self.aug)
        else:
            return type(self)(*self._get_param_dict().values())

    def to_dict(self: T) -> dict:
        result = self._get_param_dict()
        result['class_name'] = self.class_name
        if result['frequency'] == None:
            del result['frequency']
        return result

    def save_to_path(self: T, save_path: str, overwrite: bool=False):
        if file_exists(save_path) and not overwrite:
            logger.error(f'File already exists at save_path: {save_path}')
            raise Exception
        elif file_exists(save_path) and overwrite: 
            json_dict_list = json.load(open(save_path, 'r'))
            json_dict_list = [item for item in json_dict_list if item['class_name'] != self.class_name]
            json_dict_list.append(self.to_dict())
        else:
            json_dict = self.to_dict()
            json_dict_list = [json_dict]
        json.dump(json_dict_list, open(save_path, 'w'), indent=2, ensure_ascii=False)

    @classmethod
    def load_from_path(cls: T, json_path:str) -> T:
        check_file_exists(json_path)
        json_dict_list = json.load(open(json_path, 'r'))
        json_dict = [item for item in json_dict_list if item['class_name'] == cls.__name__]
        if len(json_dict) == 0:
            raise Exception('Mode settings is not found inside json path')
        else:
            json_dict = json_dict[0]
        return cls.from_dict(json_dict)

class BaseModeHandler(Generic[H, T]):
    def __init__(self: H, obj_types: List[type], obj_list: List[T]=None, random_order: bool = False):
        check_type(obj_types, valid_type_list=[list])
        check_type_from_list(obj_types, valid_type_list=[type, BaseModeMeta])
        self.obj_types = obj_types
        if obj_list is not None:
            check_type_from_list(obj_list, valid_type_list=obj_types)
        self.obj_list = obj_list if obj_list is not None else []
        self.random_order = random_order

    def __str__(self: H):
        print_str = ""
        for obj in self.obj_list:
            print_str += f"{obj}\n"

        return print_str

    def __repr__(self: H):
        return self.__str__()

    def __len__(self: H) -> int:
        return len(self.obj_list)

    def __getitem__(self: H, idx: int) -> T:
        if type(idx) is int:
            if len(self.obj_list) == 0:
                logger.error(f"{type(self).__name__} is empty.")
                raise IndexError
            elif idx < 0 or idx >= len(self.obj_list):
                logger.error(f"Index out of range: {idx}")
                raise IndexError
            else:
                return self.obj_list[idx]
        elif type(idx) is slice:
            return self.obj_list[idx.start:idx.stop:idx.step]
        else:
            logger.error(f'Expected int or slice. Got type(idx)={type(idx)}')
            raise TypeError

    def __setitem__(self: H, idx: int, value: T):
        check_type(value, valid_type_list=self.obj_types)
        if type(idx) is int:
            self.obj_list[idx] = value
        elif type(idx) is slice:
            self.obj_list[idx.start:idx.stop:idx.step] = value
        else:
            logger.error(f'Expected int or slice. Got type(idx)={type(idx)}')
            raise TypeError

    def __delitem__(self: H, idx):
        if type(idx) is int:
            if len(self.obj_list) == 0:
                logger.error(f"{type(self).__name__} is empty.")
                raise IndexError
            elif idx < 0 or idx >= len(self.obj_list):
                logger.error(f"Index out of range: {idx}")
                raise IndexError
            else:
                del self.obj_list[idx]
        elif type(idx) is slice:
            del self.obj_list[idx.start:idx.stop:idx.step]
        else:
            logger.error(f'Expected int or slice. Got type(idx)={type(idx)}')
            raise TypeError

    def __iter__(self: H):
        self.n = 0
        return self

    def __next__(self: H) -> T:
        if self.n < len(self.obj_list):
            result = self.obj_list[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration

    def copy(self: H) -> H:
        if type(self) is BaseModeHandler:
            return BaseModeHandler(
                obj_types=self.obj_types.copy(),
                obj_list=self.obj_list.copy()
            )
        else:
            return type(self)(self.obj_list.copy())

    def append(self: H, item: T):
        check_type(item, valid_type_list=self.obj_types)
        self.obj_list.append(item)

    def shuffle(self: H):
        random.shuffle(self.obj_list)

    def to_dict_list(self: H) -> List[dict]:
        return [item.to_dict() for item in self]

    def save_to_path(self: H, save_path: str, overwrite: bool=False):
        if file_exists(save_path) and not overwrite:
            logger.error(f'File already exists at save_path: {save_path}')
            raise Exception
        json_data = self.to_dict_list()
        json.dump(json_data, open(save_path, 'w'), indent=2, ensure_ascii=False)
