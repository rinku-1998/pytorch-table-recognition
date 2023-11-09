import cv2
import numpy as np
from enums.stage_name import StageName
from modules.base import BaseModule
from modules.unet_wrapper import UNetWrapper
from utils.img_util import np_to_pil, smart_resize
from utils.line_util import distance, angle
from utils.point_util import sort_points_anticlockwise
from typing import Optional, List, Tuple


class ShapeFixer(BaseModule):

    def __init__(self, config: dict) -> None:

        # 1. 建立模型 Wrapper
        model_config = config.get('model')
        self.wrapper = UNetWrapper(**model_config)

        # 2. 設定資料
        self.img_size = (512, 512)
        self.lower_percentage = 0.55
        self.upper_percentage = 0.96
        self.max_diagonal_diff = 30

        # 3. 更新參數
        self.update_config(config.get('build'))

    def _preprocess(
            self,
            img: np.ndarray) -> Tuple[np.ndarray, int, int, int, int, float]:

        # 1. 計算新的圖片尺寸比例
        h, w, _ = img.shape
        w_target, h_target = self.img_size

        h_ratio = h_target / h
        w_ratio = w_target / w
        ratio = min(h_ratio, w_ratio)

        new_h = int(h * ratio)
        new_w = int(w * ratio)

        # 2. 縮放圖片
        img_resizenp = smart_resize(img, (new_w, new_h))

        # 3. 計算邊界
        top = int((h_target - new_h) / 2)
        bottom = h_target - new_h - top
        left = int((w_target - new_w) / 2)
        right = w_target - new_w - left

        # 4. 填充邊界影像
        img_filled = cv2.copyMakeBorder(img_resizenp,
                                        top,
                                        bottom,
                                        left,
                                        right,
                                        cv2.BORDER_CONSTANT,
                                        value=(127, 127, 127))

        return img_filled, top, bottom, left, right, ratio

    def _find_shape(self, mask: np.ndarray, lower_percentage: float,
                    upper_percentage: float) -> Optional[np.ndarray]:

        # 1. 找紙張輪廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # 2. 計算近似多邊形，從面積最大的開始計算，直到找到四邊形為止。
        # NOTE: 只計算前五個，避免計算到過小的圖形
        max_contours = 5
        sorted_contours = sorted(contours, key=cv2.contourArea,
                                 reverse=True)[:max_contours]
        poly = None
        for c in sorted_contours:

            # 判斷輪廓大小是否有介於上下限，避免使用到過小的輪廓當作紙張
            counter_area = cv2.contourArea(c)
            if counter_area < mask.size * lower_percentage:
                continue
            if counter_area >= mask.size * upper_percentage:
                continue

            contour_len = cv2.arcLength(c, True)

            approx = cv2.approxPolyDP(c, 0.02 * contour_len, True)
            if len(approx) == 4:
                poly = approx
                break

        return poly

    def _restore_point(self, box_points: List[List[int]],
                       ratio: float) -> List[List[int]]:

        restored_bps: List[List[int]] = list()
        for box_point in box_points:
            x, y = box_point
            x = round(x / ratio)
            y = round(y / ratio)

            restored_bps.append([x, y])

        return restored_bps

    def _expand_shape(self, box_points: List[List[int]],
                      img_shape: Tuple[int]) -> List[List[int]]:

        # 1. 設定向外擴張像素
        offset = 30

        # 2. 四個頂點向外擴張
        h, w = img_shape
        epbox_points: List[List[int]] = list()
        for i, box_point in enumerate(box_points):

            x, y = box_point
            if i == 0:

                x = max(0, x - offset)
                y = max(0, y - offset)
                epbox_point = [x, y]
                epbox_points.append(epbox_point)

                continue

            if i == 1:

                x = min(w - 1, x + offset)
                y = max(0, y - offset)
                epbox_point = [x, y]
                epbox_points.append(epbox_point)

                continue

            if i == 2:

                x = min(w - 1, x + offset)
                y = min(h - 1, y + offset)
                epbox_point = [x, y]
                epbox_points.append(epbox_point)

                continue

            else:
                x, y = box_point
                x = max(0, x - offset)
                y = min(h - 1, y + offset)
                epbox_point = [x, y]
                epbox_points.append(epbox_point)

                continue

        return epbox_points

    def _check_shape(self, points: List[List[int]],
                     max_diagonal_diff: int) -> bool:
        """檢查形狀，判斷是否是平行四邊形、梯形、長方形、正方形

        Args:
            points (List[List[int]]): 四邊形座標列表(左上、右上、右下、左下)
            max_diagonal_diff: 最大允許的兩對角線長度差

        Returns:
            bool: 是否是平行四邊形、梯形、長方形、正方形
        """

        # 1. 檢查對角線是否相等，若相等代表形狀是平行四邊形、梯形、長方形、正方形
        point1, point2, point3, point4 = points

        diagonal1 = distance(point1, point3)
        diagonal2 = distance(point2, point4)

        if abs(diagonal1 - diagonal2) > max_diagonal_diff:
            return False

        return True

    def _rotate_incomplete(self, img: np.ndarray,
                           points: List[List[int]]) -> np.ndarray:

        # 1. 計算上下2條水平線的角度
        point1, point2, point3, point4 = points
        angle1 = angle(point1, point2)
        angle2 = angle(point4, point3)

        # 2. 檢查角度是否有任一條水平邊是水平的，+-5度之間
        top_is_horizontal = angle1 <= 5.0 and angle1 >= -5.0
        bottom_is_horizontal = angle2 <= 5.0 and angle2 >= -5.0

        if not (top_is_horizontal ^ bottom_is_horizontal):
            return img

        # 3. 計算旋轉角度
        rotate_degree = None
        if top_is_horizontal:
            rotate_degree = round(angle1)
        if bottom_is_horizontal:
            rotate_degree = round(angle2)

        # 4. 旋轉圖片
        # NOTE: 第一個參數旋轉中心，第二個參數旋轉角度(-順時針/+逆時針)，第三個參數縮放比例
        h, w = img.shape[:2]
        center = (w // 2, h // 2)  # 找到圖片中心
        M = cv2.getRotationMatrix2D(center, rotate_degree, 1.0)

        # 旋轉
        img_rotate = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

        return img_rotate

    def _calculate_pers_shape(
            self,
            epbox_points: List[List[int]],
            img_shape: Tuple[int],
            shape_percentage: float = 1.0) -> Tuple[int, int]:

        # 1. 計算變形前的表格尺寸
        shape_h = (abs(epbox_points[0][1] - epbox_points[3][1]) +
                   abs(epbox_points[1][1] - epbox_points[2][1])) / 2
        shape_w = (abs(epbox_points[0][0] - epbox_points[1][0]) +
                   abs(epbox_points[2][0] - epbox_points[3][0])) / 2

        # 2. 計算變形後表格的尺吋
        # NOTE: 變換後的表格大小長邊要佔原始圖片的長邊的90%

        # 計算長邊縮放比例
        new_h, new_w = img_shape[:2]
        ratio = None
        if shape_w >= shape_h:
            ratio = round(new_w * shape_percentage) / shape_w
        else:
            ratio = round(new_h * shape_percentage) / shape_h

        # 計算表格新的尺寸
        pers_w = round(shape_w * ratio)
        pers_h = round(shape_h * ratio)

        return pers_w, pers_h

    def fix(self,
            img: np.ndarray,
            output_dir: Optional[str] = None) -> np.ndarray:

        # 1. 設定輸出路徑
        super().update_output(output_dir)

        # 2. 圖片前處理
        img_filled, top, bottom, left, right, ratio = self._preprocess(img)

        # 3. 推論模型
        img_pil = np_to_pil(img_filled)
        mask = self.wrapper.predict_img(img_pil)
        mask = mask.astype(np.uint8)

        # 存檔
        mask_save = mask.copy()
        mask_save[mask_save == 1] = 255
        self.save_rimg(mask_save, StageName.PAPER_MAP.value)

        # 4. 找紙張輪廓
        # 裁切邊界填充
        h, w = mask.shape
        mask = mask[top:h - bottom, left:w - right]

        # 計算四邊形
        poly = self._find_shape(mask, self.lower_percentage,
                                self.upper_percentage)
        if poly is None:
            self.save_rimg(img, StageName.SHAPE_FIX.value)
            return img

        # 5. 逆時針(按照象限)排序
        poly = np.reshape(poly, (4, 2))
        box_points = sort_points_anticlockwise(poly)

        # 6. 還原座標點
        restored_points = self._restore_point(box_points, ratio)

        # NOTE: 2023-06-20 因為分割表格效果不好，先暫時不使用這一段去處理表格的分割，改回使用紙張分割。
        '''
        # 7. 調整遮罩尺寸(向外擴張)
        epbox_points = self._expand_shape(restored_points, img.shape[:2])
        '''

        # 8. 檢查四邊形形狀
        # NOTE: 如果形狀不符合平行四邊形、梯形、長方形以外的，就檢查是不是需要人工旋轉沒有拍完整的照片(一邊水平、一邊歪斜)
        if not self._check_shape(restored_points, self.max_diagonal_diff):

            img_rotate = self._rotate_incomplete(img, restored_points)
            self.save_rimg(img_rotate, StageName.SHAPE_FIX.value)
            return img

        # 9. 計算變換後的尺寸
        pers_w, pers_h = self._calculate_pers_shape(restored_points,
                                                    img.shape[:2])

        # 10. 透視變換
        # 計算座標
        points1 = np.float32(restored_points)
        points2 = np.float32([
            [0, 0],
            [pers_w, 0],
            [pers_w, pers_h],
            [0, pers_h],
        ])

        # 變換
        M = cv2.getPerspectiveTransform(points1, points2)
        img_pers = cv2.warpPerspective(img,
                                       M, (pers_w, pers_h),
                                       flags=cv2.INTER_CUBIC)

        # 11. 存檔
        self.save_rimg(img_pers, StageName.SHAPE_FIX.value)

        return img_pers
