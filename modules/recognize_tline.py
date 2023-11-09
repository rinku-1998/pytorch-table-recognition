import numpy as np
from collections import defaultdict
from enums.stage_name import StageName
from exceptions.table_format import EmptyColException, EmptyRowException
from dto.line import Line
from dto.table import Table
from skimage import measure
from modules.base import BaseModule
from modules.recognize_tlmap import TLineMapRecognizer
from typing import Dict, List, Optional, Tuple
from utils.img_util import draw_line
from utils.line_util import points_by_minarea, distance


class TLineRecognizer(BaseModule):

    def __init__(self, config: dict) -> None:

        # 1. 設定參數
        self.min_row_height = 10  # 最小水平線高度像素
        self.min_col_width = 10  # 最小垂直線寬度像素

        self.min_row_len = 100  # 水平線最小允許長度像素
        self.min_col_len = 50  # 垂直線最小允許長度像素

        self.max_merge_distance = 45  # 最大可合併的線段長度差
        self.max_merge_parallel_diff = 10  # 最大允許合併的線段平行距離

        self.ot_w_percentage = 0.8  # 移除表格外的水平線長度倍數(大於才保留)
        self.ot_h_percentage = 0.8  # 移除表格外的垂直線長度倍數(大於才保留)

        self.row_color = (50, 168, 78)  # 水平線顏色
        self.col_color = (50, 125, 168)  # 垂直線顏色

        # 2. 更新參數
        build_config = config.get('build')
        self.update_config(build_config)

        # 3. 建立表格線條分割圖辨識模組
        model_config = config.get('model')
        self.tlmr = TLineMapRecognizer(model_config)

    def separate_row_col(self,
                         img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """分離水平與垂直線像素圖

        Args:
            img (np.ndarray): 線條像素圖

        Returns:
            Tuple[np.ndarray, np.ndarray]: 水平像素圖, 垂直像素圖
        """

        # 1. 取出水平線
        row_img = img.copy()
        row_img[row_img == 0] = 0
        row_img[row_img == 1] = 255
        row_img[row_img == 2] = 0
        row_img[row_img == 3] = 255

        # 2. 取出垂直線
        col_img = img.copy()
        col_img[col_img == 0] = 0
        col_img[col_img == 1] = 0
        col_img[col_img == 2] = 255
        col_img[col_img == 3] = 255

        return row_img, col_img

    def pixel_to_line(
            self,
            img: np.ndarray,
            min_thickness: int = 10) -> Tuple[float, float, float, float]:
        """像素圖轉線條

        Args:
            img (np.ndarray): 線條影像
            min_thickness (int, optional): 最小厚度像素. Defaults to 10.

        Returns:
            Tuple[float, float, float, float]: 線條列表[x_start, y_start, x_end, y_end]
        """

        # 1. 計算相鄰區域
        labels = measure.label(img, connectivity=2)
        regions = measure.regionprops(labels)

        # 2. 使用最小矩形計算線條座標點
        # TODO: 等權重好訓練好之後再決定要不要過濾粗細度
        lines = [points_by_minarea(line.coords) for line in regions]

        return lines

    def restore_point(self, lines: List[List[int]],
                      ratio: float) -> List[List[int]]:

        # 1. 還原縮放比例
        restored_lines: List[List[int]] = list()
        for line in lines:
            x_start, y_start, x_end, y_end = line
            x_start = round(x_start / ratio)
            y_start = round(y_start / ratio)
            x_end = round(x_end / ratio)
            y_end = round(y_end / ratio)

            restored_line = [x_start, y_start, x_end, y_end]
            restored_lines.append(restored_line)

        return restored_lines

    def normalize_point(
            self, row_lines: List[List[int]], col_lines: List[List[int]]
    ) -> Tuple[List[List[int]], List[List[int]]]:

        # 1. 調整水平線的起迄位置
        normalized_rlines: List[List[int]] = list()
        for row_line in row_lines:

            x_start, y_start, x_end, y_end = row_line
            if x_end < x_start:
                x_start, x_end = x_end, x_start
                y_start, y_end = y_end, y_start

            normalized_line = [x_start, y_start, x_end, y_end]
            normalized_rlines.append(normalized_line)

        # 2. 檢查垂直線的起迄位置
        normalized_clines: List[List[int]] = list()
        for col_line in col_lines:

            x_start, y_start, x_end, y_end = col_line
            if y_end < y_start:
                x_start, x_end = x_end, x_start
                y_start, y_end = y_end, y_start

            normalized_line = [x_start, y_start, x_end, y_end]
            normalized_clines.append(normalized_line)

        return normalized_rlines, normalized_clines

    def build_line_obj(self, lines: Tuple[int, int, int, int]) -> List[Line]:
        """建立線條物件

        Args:
            lines (Tuple[int, int, int, int]): 線條列表

        Returns:
            List[Line]: 線條物件列表
        """

        line_objs: List[Line] = list()
        for l in lines:

            x_start, y_start, x_end, y_end = l
            line_obj = Line(x_start=x_start,
                            y_start=y_start,
                            x_end=x_end,
                            y_end=y_end)
            line_objs.append(line_obj)

        return line_objs

    def remove_short(self, lines: List[Line], min_len: int = 50) -> List[Line]:
        """移除過短的線條

        Args:
            lines (List[Line]): 線條物件列表
            min_len (int, optional): 最短長度像素. Defaults to 50.

        Returns:
            List[Line]: 移除後的線條物件列表
        """

        return [l for l in lines if l.length >= min_len]

    def merge_line(self, table: Table, max_distance,
                   max_parallel_diff) -> Table:

        # 1. 延長水平線
        row_lines = table.row_lines
        row_lines.sort(key=lambda l: l.x_start)
        rsrc_to_dsts: Dict[int, List[int]] = defaultdict(list)
        used_ridxs: List[int] = list()
        for src_idx, src_line in enumerate(row_lines):

            # 檢查來源線條是否有被連結過
            if src_idx in used_ridxs:
                continue

            for dst_idx, dst_line in enumerate(row_lines):

                # 檢查目標
                if (src_idx == dst_idx) or (src_idx in used_ridxs):
                    continue

                # 取得合併後線條的結束座標
                dst_idxs = rsrc_to_dsts.get(src_idx)
                x_end = row_lines[max(
                    dst_idxs)].x_end if dst_idxs else src_line.x_end
                y_end = row_lines[max(
                    dst_idxs)].y_end if dst_idxs else src_line.y_end

                # 計算來源線條距離與目標線條的距離
                d = distance((x_end, y_end),
                             (dst_line.x_start, dst_line.y_start))
                if d > max_distance:
                    continue

                if abs(y_end - dst_line.y_start) > max_parallel_diff:
                    continue

                used_ridxs.append(src_idx)
                used_ridxs.append(dst_idx)
                rsrc_to_dsts[src_idx].append(dst_idx)

        # 2. 整理水平線資料
        # 建立合併線條的資料
        merged_rlines: List[Line] = list()
        for src_idx, dst_idxs in rsrc_to_dsts.items():

            x_start = row_lines[src_idx].x_start
            y_start = row_lines[src_idx].y_start
            x_end = row_lines[max(
                dst_idxs)].x_end if dst_idxs else src_line.x_end
            y_end = row_lines[max(
                dst_idxs)].y_end if dst_idxs else src_line.y_end
            merged_rline = Line(x_start=x_start,
                                y_start=y_start,
                                x_end=x_end,
                                y_end=y_end)
            merged_rlines.append(merged_rline)

        # 加入沒有合併的線條
        for idx, row_line in enumerate(row_lines):
            if idx in used_ridxs:
                continue

            merged_rlines.append(row_line)

        # 排序
        merged_rlines.sort(key=lambda l: (l.y_start + l.y_end) / 2)

        # 3. 延長垂直線
        col_lines = table.col_lines
        col_lines.sort(key=lambda l: l.y_start)
        csrc_to_dsts: Dict[int, List[int]] = defaultdict(list)
        used_cidxs: List[int] = list()
        for src_idx, src_line in enumerate(col_lines):

            # 檢查來源線條是否有被連結過
            if src_idx in used_cidxs:
                continue

            for dst_idx, dst_line in enumerate(col_lines):

                # 檢查目標
                if (src_idx == dst_idx) or (src_idx in used_cidxs):
                    continue

                # 取得合併後線條的結束座標
                dst_idxs = csrc_to_dsts.get(src_idx)
                x_end = col_lines[max(
                    dst_idxs)].x_end if dst_idxs else src_line.x_end
                y_end = col_lines[max(
                    dst_idxs)].y_end if dst_idxs else src_line.y_end

                # 計算來源線條距離與目標線條的距離
                d = distance((x_end, y_end),
                             (dst_line.x_start, dst_line.y_start))
                if d > max_distance:
                    continue

                if abs(x_end - dst_line.x_start) > max_parallel_diff:
                    continue

                used_cidxs.append(src_idx)
                used_cidxs.append(dst_idx)
                csrc_to_dsts[src_idx].append(dst_idx)

        # 4. 整理垂直線資料
        # 建立合併線條的資料
        merged_clines: List[Line] = list()
        for src_idx, dst_idxs in csrc_to_dsts.items():

            x_start = col_lines[src_idx].x_start
            y_start = col_lines[src_idx].y_start
            x_end = col_lines[max(
                dst_idxs)].x_end if dst_idxs else src_line.x_end
            y_end = col_lines[max(
                dst_idxs)].y_end if dst_idxs else src_line.y_end
            merged_rline = Line(x_start=x_start,
                                y_start=y_start,
                                x_end=x_end,
                                y_end=y_end)
            merged_clines.append(merged_rline)

        # 加入沒有合併的線條
        for idx, col_line in enumerate(col_lines):
            if idx in used_cidxs:
                continue

            merged_clines.append(col_line)

        # 排序
        merged_clines.sort(key=lambda l: (l.x_start + l.x_end) / 2)

        # 5. 設定資料
        merge_table = Table(row_lines=merged_rlines, col_lines=merged_clines)

        return merge_table

    def build_table_obj(self, row_lines: List[Line],
                        col_lines: List[Line]) -> Table:

        return Table(row_lines=row_lines, col_lines=col_lines)

    def remove_out_table(self, table: Table, w_percentage: float,
                         h_percentage: float) -> Table:

        # 1. 取得表格尺寸
        h, w = table.size

        # 2. 排序線條
        row_lines = table.row_lines
        row_lines.sort(key=lambda l: (l.y_start + l.y_end) / 2)
        col_lines = table.col_lines
        col_lines.sort(key=lambda l: (l.x_start + l.x_end) / 2)

        # 3. 過濾表格外的上方水平線條
        for idx, l in enumerate(row_lines):
            if l.length >= w * w_percentage:
                row_lines = row_lines[idx:]
                break

        # 4. 過濾表格外的下方水平線條
        for idx, l in enumerate(row_lines[::-1]):
            if l.length >= w * w_percentage:
                row_lines = row_lines[:len(row_lines) - idx]
                break

        # 5. 過濾表格外的左方垂直線條
        for idx, l in enumerate(col_lines):
            if l.length >= h * h_percentage:
                col_lines = col_lines[idx:]
                break

        # 6. 過濾表格外的右方垂直線條
        for idx, l in enumerate(col_lines[::-1]):
            if l.length >= h * h_percentage:
                col_lines = col_lines[:len(col_lines) - idx]
                break

        # NOTE: 要先確保表格外沒有多餘的水平與垂直線後，在交叉過濾
        # 7. 過濾水平線最上與最下方的垂直線
        filtered_clines: List[Line] = list()
        y_min = min(row_lines[0].y_start, row_lines[0].y_end)
        y_max = max(row_lines[-1].y_start, row_lines[-1].y_end)
        for col in col_lines:
            if col.y_end < y_min:
                continue
            if col.y_start > y_max:
                continue

            filtered_clines.append(col)

        # 8. 過濾垂直線最上與最下方的水平線
        filtered_rlines: List[Line] = list()
        x_min = min(col_lines[0].x_start, col_lines[0].x_end)
        x_max = max(col_lines[-1].x_start, col_lines[-1].x_end)
        for row in row_lines:
            if row.x_end < x_min:
                continue
            if row.x_start > x_max:
                continue

            filtered_rlines.append(row)

        # 9. 建立新的表格物件
        # NOTE: 為了保證表格尺寸可以重新計算，所以重新鑑裡一個新物件
        removed_table = Table(row_lines=filtered_rlines,
                              col_lines=filtered_clines)

        return removed_table

    def check_table(self, table: Table) -> None:

        if len(table.row_lines) == 0:
            raise EmptyRowException
        if len(table.col_lines) == 0:
            raise EmptyColException

    def recognize(self, img: np.ndarray, output_dir: Optional[str] = None):

        # 1. 設定輸出路徑
        super().update_output(output_dir)

        # 2. 辨識表格線條分跟圖
        mask, ratio = self.tlmr.recognize(img)

        # 3. 分割水平線與垂直線像素圖
        row_img, col_img = self.separate_row_col(mask)

        # 存檔
        if self.save:
            smask = row_img + col_img
            self.save_rimg(smask, StageName.TABLE_MAP.value)
            self.save_rimg(row_img, StageName.TABLE_MAP.value + 'R')
            self.save_rimg(col_img, StageName.TABLE_MAP.value + 'C')

        # 4. 將像素圖轉為線條
        raw_rlines = self.pixel_to_line(row_img)
        raw_clines = self.pixel_to_line(col_img)

        # 5. 還原線條座標
        restored_rlines = self.restore_point(raw_rlines, ratio)
        restored_clines = self.restore_point(raw_clines, ratio)

        # 6. 調整線條座標起訖位置
        normalized_rlines, normalized_clines = self.normalize_point(
            restored_rlines, restored_clines)

        # 7. 建立線條物件
        row_lines = self.build_line_obj(normalized_rlines)
        col_lines = self.build_line_obj(normalized_clines)

        # 排序
        row_lines.sort(key=lambda l: (l.y_start + l.y_end) / 2)
        col_lines.sort(key=lambda l: (l.x_start + l.x_end) / 2)

        # 存檔
        if self.save:
            img_line = img.copy()
            img_line = draw_line(img_line, row_lines, self.row_color)
            img_line = draw_line(img_line, col_lines, self.col_color)
            self.save_rimg(img_line, StageName.LINE_ORIGINAL.value)

        # 8. 過濾過短線條
        row_lines = self.remove_short(row_lines, self.min_row_len)
        col_lines = self.remove_short(col_lines, self.min_col_len)

        # 存檔
        if self.save:
            img_rmshort = img.copy()
            img_rmshort = draw_line(img_rmshort, row_lines, self.row_color)
            img_rmshort = draw_line(img_rmshort, col_lines, self.col_color)
            self.save_rimg(img_rmshort, StageName.LINE_RMSHORT.value)

        # 9. 建立表格物件
        table = self.build_table_obj(row_lines, col_lines)
        self.check_table(table)

        # 10. 合併表格線條
        table_merged = self.merge_line(table, self.max_merge_distance,
                                       self.max_merge_parallel_diff)
        self.check_table(table_merged)

        if self.save:
            img_merged = img.copy()
            img_merged = draw_line(img_merged, table_merged.row_lines,
                                   self.row_color)
            img_merged = draw_line(img_merged, table_merged.col_lines,
                                   self.col_color)
            self.save_rimg(img_merged, StageName.LINE_MERGED.value)

        # 11. 移除表格外的線條
        table_removed = self.remove_out_table(table_merged,
                                              self.ot_w_percentage,
                                              self.ot_h_percentage)
        self.check_table(table_removed)

        if self.save:
            img_rmouttable = img.copy()
            img_rmouttable = draw_line(img_rmouttable, table_removed.row_lines,
                                       self.row_color)
            img_rmouttable = draw_line(img_rmouttable, table_removed.col_lines,
                                       self.col_color)
            self.save_rimg(img_rmouttable, StageName.LINE_RMOUTTABLE.value)

        return table_removed
