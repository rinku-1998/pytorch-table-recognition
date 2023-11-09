import cv2
import numpy as np
from modules.base import BaseModule
from modules.unet_wrapper import UNetWrapper
from utils.img_util import np_to_pil, smart_resize
from typing import Tuple


class TLineMapRecognizer(BaseModule):

    def __init__(self, config: dict) -> None:

        # 1. 設定參數
        self.weight_path = None
        self.threshold = 0.5
        self.num_classes = 4
        self.device = 'cpu'
        self.img_size = (512, 512)

        # 2. 更新參數
        self.update_config(config)

        # 3. 建立模型 Wrapper
        self.wrapper = UNetWrapper(self.weight_path,
                                   self.num_classes,
                                   device=self.device,
                                   threshold=self.threshold)

    def _preprocess(
            self,
            img: np.ndarray) -> Tuple[np.ndarray, int, int, int, int, float]:

        # 1. 侵蝕(加深線條)
        # NOTE: 2023-07-03 取消侵蝕，修正影像縮放公用程式的插值方式避免鋸齒
        # kernel = np.ones((3, 3), np.uint8)
        # img_erode = cv2.erode(img, kernel, iterations = 1)

        # 2. 計算新的圖片尺寸比例
        h, w, _ = img.shape
        w_target, h_target = self.img_size

        # 計算內部尺寸
        # NOTE: 2023-08-08 在填充圖片時再向內縮 6px，避免表格線條太靠邊緣導致檢測不出來的問題
        h_inner = h_target - 6
        w_inner = w_target - 6
        h_ratio = (h_inner) / h
        w_ratio = (w_inner) / w
        ratio = min(h_ratio, w_ratio)

        new_h = int(h * ratio)
        new_w = int(w * ratio)

        # 3. 縮放圖片
        img_resizenp = smart_resize(img, (new_w, new_h))

        # 4. 計算邊界
        top = int((h_target - new_h) / 2)
        bottom = h_target - new_h - top
        left = int((w_target - new_w) / 2)
        right = w_target - new_w - left

        # 5. 填充邊界影像
        img_filled = cv2.copyMakeBorder(img_resizenp,
                                        top,
                                        bottom,
                                        left,
                                        right,
                                        cv2.BORDER_CONSTANT,
                                        value=(127, 127, 127))

        return img_filled, top, bottom, left, right, ratio

    def recognize(self, img: np.ndarray) -> Tuple[np.ndarray, float]:

        # 1. 前處理
        img_filled, top, bottom, left, right, ratio = self._preprocess(img)

        # 2. 推論模型
        img_pil = np_to_pil(img_filled)
        mask = self.wrapper.predict_img(img_pil)
        mask = mask.astype(np.uint8)

        # 3. 還原影像(移除邊框)
        w, h = self.img_size
        mask_nb = mask[top:h - bottom, left:w - right]

        return mask_nb, ratio
