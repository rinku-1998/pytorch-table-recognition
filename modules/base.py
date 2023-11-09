import numpy as np
from pathlib import Path
from typing import Optional
from utils.img_util import save_img


class BaseModule:

    def __init__(self, output_dir: Optional[str] = None) -> None:

        # 1. 更新輸出路徑
        self.update_output(output_dir)

    def update_output(self, output_dir: Optional[str] = None) -> None:

        # 1. 設定資料
        self.output_dir = output_dir
        self.save = True if output_dir else False

    def update_config(self, config: dict) -> None:

        # 1. 檢查資料是否為空
        if config is None:
            return None

        # 2. 更新參數
        for k, v in config.items():

            # 檢查參數名稱是否存在
            if not hasattr(self, k):
                continue

            # 設定參數
            setattr(self, k, v)

        return None

    def save_rimg(self, img: np.ndarray, stage_name: str) -> None:

        # 1. 檢查是否需要存檔
        if not self.save:
            return None

        # 2. 存檔
        fname = f'IMR{stage_name}.jpg'
        save_path = Path(self.output_dir, fname)
        save_img(img, str(save_path))