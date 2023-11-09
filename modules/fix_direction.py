import cv2
import numpy as np
from enums.stage_name import StageName
from modules.base import BaseModule
from modules.efficientnet_wrapper import EfficientNetWrapper
from utils.img_util import np_to_pil
from typing import Optional


class DirectionFixer(BaseModule):

    def __init__(self, config: dict) -> None:

        # 1. 建立模型 Wrapper
        model_config = config.get('model')
        self.wrapper = EfficientNetWrapper(**model_config)

    def fix(self,
            img: np.ndarray,
            output_dir: Optional[str] = None) -> np.ndarray:
        """修正圖片方向

        Args:
            img (np.ndarray): 影像
            output_dir (Optional[str]) 輸出路徑，defaults to None

        Returns:
            np.ndarray: 修正方向後的影像
        """

        # 1. 設定輸出路徑
        super().update_output(output_dir)

        # 2. 檢查直向或橫向，如果是直的就先轉成橫的
        h, w = img.shape[:2]
        if h > w:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        # 3. 推論模型
        img_pil = np_to_pil(img)
        direction = self.wrapper.predict_img(img_pil)

        # 4. 修正方向，0=正向，1=反向
        img_fix = img.copy()
        if direction == 1:
            img_fix = cv2.rotate(img_fix, cv2.ROTATE_180)

        # 5. 存檔
        self.save_rimg(img_fix, StageName.DIRECTION_FIX.value)

        return img_fix