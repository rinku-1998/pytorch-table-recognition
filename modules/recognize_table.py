import cv2
import numpy as np
from enums.stage_name import StageName
from dto.table import Table
from modules.base import BaseModule
from modules.fix_direction import DirectionFixer
from modules.fix_shape import ShapeFixer
from modules.recognize_tline import TLineRecognizer
from pathlib import Path
from typing import Optional
from utils.img_util import smart_resize
from utils.json_util import save_json


class TableRecognizer(BaseModule):

    def __init__(self, config: dict) -> None:

        # 1. 設定參數
        self.resize_size = (1280, 960)

        # 2. 更新參數
        preprocess_config = config.get('preprocess')
        self.update_config(preprocess_config)

        # 3. 建立方向修正模組
        df_config = config.get('paper_direction')
        self.df = DirectionFixer(df_config)

        # 4. 建立變形修正模組
        sf_config = config.get('paper')
        self.sf = ShapeFixer(sf_config)

        # 5. 建立表格辨識模組
        tlr_config = config.get('table_line')
        self.tlr = TLineRecognizer(tlr_config)

    def normalize_resize(self, img: np.ndarray) -> np.ndarray:

        # 1. 計算縮放比例
        h, w = img.shape[:2]
        new_w, new_h = self.resize_size

        h_ratio = new_h / h
        w_ratio = new_w / w

        ratio = min(h_ratio, w_ratio)

        # 2. 計算新比例
        new_w = round(w * ratio)
        new_h = round(h * ratio)

        # 3. 縮放
        target_shape = (new_w, new_h)
        img_resize = smart_resize(img, (target_shape))

        return img_resize

    def save_table(self, table: Table) -> None:

        # 1. 檢查是否需要存檔
        if not self.save:
            return None

        # 2. 存檔
        save_path = Path(self.output_dir, 'table.json')
        save_json(table, str(save_path))

        return None

    def run(self, img: np.ndarray, output_dir: Optional[str] = None) -> Table:

        # 1. 設定輸出路徑
        super().update_output(output_dir)

        # 2. 儲存原始照片
        self.save_rimg(img, StageName.ORIGINAL.value)

        # 3. 修正方向
        img_direction_fix = self.df.fix(img, output_dir)

        # 4. 修正尺寸
        img_resize = self.normalize_resize(img_direction_fix)
        self.save_rimg(img_resize, StageName.RESIZE.value)

        # 5. 紙張變形修正
        img_paper_fix = self.sf.fix(img_resize, output_dir)

        # 6. 辨識表格線條
        table = self.tlr.recognize(img_paper_fix, output_dir)

        # 7. 儲存表格資料
        self.save_table(table)

        return table
