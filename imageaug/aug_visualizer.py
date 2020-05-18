from typing import Tuple
import cv2
import numpy as np
import math

from logger import logger
from common_utils.image_utils import concat_n_images

class AugVisualizer:
    def __init__(self, vis_save_path: str, n_rows: int=3, n_cols: int=5, save_dims: Tuple[int]=(3*500, 5*500), wait: int=None):
        self.vis_save_path = vis_save_path
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.save_step = n_rows * n_cols
        self.save_h, self.save_w = save_dims
        self.item_h, self.item_w = int(self.save_h / self.n_rows), int(self.save_w / self.n_cols)

        self.vis_buffer = []
        self.wait = wait
        self.wait_count = 0

    def step(self, aug_img: np.ndarray):
        if len(self.vis_buffer) == 0:
            if self.wait is not None and self.wait_count < self.wait:
                self.wait_count += 1
                return
            else:
                self.wait_count = 0
        aug_img_resized = cv2.resize(aug_img, dsize=(self.item_w, self.item_h))
        self.vis_buffer.append(aug_img_resized)
        if len(self.vis_buffer) == self.save_step:
            blank_vis = np.zeros_like(self.vis_buffer[0])
            self.vis_buffer = self.vis_buffer + [blank_vis]*(self.n_rows * self.n_cols - len(self.vis_buffer))
            target_shape = [self.n_rows, self.n_cols] + list(self.vis_buffer[0].shape)
            vis_matrix = np.array(self.vis_buffer).reshape(target_shape)
            result = concat_n_images(
                img_list=[
                    concat_n_images(
                        img_list=vis_row,
                        orientation=0
                    ) for vis_row in vis_matrix
                ],
                orientation=1
            )
            cv2.imwrite(self.vis_save_path, result)
            self.vis_buffer = []

aug_visualizer = AugVisualizer(
    vis_save_path='aug_vis.png',
    wait=None
)
