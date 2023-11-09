import cv2
import math
import numpy as np
from dto.line import Line
from typing import List, Optional, Tuple
from utils.point_util import sort_points_anticlockwise


def distance(point1: List[int], point2: List[int]) -> float:
    """計算兩點之間距離

    Args:
        point1 (List[int]): 座標點1(x, y)
        point2 (List[int]): 座標點2(x, y)

    Returns:
        float: 兩點距離
    """

    square = math.pow((point1[0] - point2[0]), 2) + math.pow(
        (point1[1] - point2[1]), 2)
    distance = math.pow(square, 0.5)

    return distance


def angle(point1: List[int], point2: List[int]) -> float:
    """計算線段的角度

    Args:
        point1 (List[int]): 座標點1(x, y)
        point2 (List[int]): 座標點2(x, y)

    Returns:
        float: 角度[0, 180]
    """

    return math.atan2((point2[1] - point1[1]),
                      (point2[0] - point1[0])) / math.pi * 180


def shape(points: List[List[int]]) -> Tuple[int, int]:
    """計算兩點間的尺寸(長寬)

    Args:
        points (List[List[int]]): 座標列表[[x, y]]

    Returns:
        Tuple[int, int]: 長, 寬
    """

    # 1. 計算長寬
    w = round(
        (distance(points[0], points[1]) + distance(points[2], points[3])) / 2)
    h = round(
        distance(points[1], points[2]) + distance(points[0], points[3]) / 2)

    return h, w


def points_by_minarea(coords: List[Tuple[int]]) -> List[int]:
    """使用最小矩形計算線條座標

    Args:
        coords (List[Tuple[int, int]]): 座標點列表

    Returns:
        List[int, int, int, int]: 線段座標
    """

    # 1. 找出該線段像素區域最小的矩形
    rect = cv2.minAreaRect(coords[:, ::-1])
    box = cv2.boxPoints(rect)

    # 2. 排序矩形的四個座標點，分為左上(P1)、右上(P2)、右下(P3)、左下(P4)
    sorted_box = sort_points_anticlockwise(box)
    sorted_box = np.array(sorted_box)
    sorted_box = sorted_box.tolist()

    # 3. 計算線條長寬
    h, w = shape(sorted_box)

    # 4. 決定線段起訖座標點
    x1, y1 = sorted_box[0]
    x2, y2 = sorted_box[1]
    x3, y3 = sorted_box[2]
    x4, y4 = sorted_box[3]

    # 依據矩形旋轉的角度判斷是直線還是橫線
    x_start = None
    x_end = None
    y_start = None
    y_end = None

    # 垂直線
    if w < h:
        x_start = (x1 + x2) / 2
        x_end = (x3 + x4) / 2
        y_start = (y1 + y2) / 2
        y_end = (y3 + y4) / 2

    # 水平線
    else:
        x_start = (x1 + x4) / 2
        x_end = (x2 + x3) / 2
        y_start = (y1 + y4) / 2
        y_end = (y2 + y3) / 2

    # 5. 整理格式
    x_start = round(x_start)
    y_start = round(y_start)
    x_end = round(x_end)
    y_end = round(y_end)

    return [x_start, y_start, x_end, y_end]


def is_in_length(source_length: float, target_length: float,
                 scale_percentage: float) -> bool:

    # 1. 檢查縮小後的目標長度
    min_factor = 1 - scale_percentage
    if not (source_length >= target_length * min_factor):
        return False

    # 2. 檢查放大後的目標長度
    max_factor = 1 + scale_percentage
    if not (source_length <= target_length * max_factor):
        return False

    return True


def calculate_cross(line1: Line, line2: Line) -> Tuple[int, int]:

    # 1. 檢查斜率是否一樣
    if line1.m == line2.m:
        return None, None

    # 2. 檢查是否有水平或垂直線可以先決定x, y
    x = None
    y = None
    if line1.m == None:
        x = line1.x_start

    if line2.m == None:
        x = line2.x_start

    if line1.m == 0:
        y = line1.y_start

    if line2.m == 0:
        y = line2.y_start

    if (x is not None) and (y is not None):
        return x, y

    # 3. 計算交點
    # 決定要使用 L1 還是 L2
    use_m = None
    use_b = None
    if (line1.m is not None) and (line1.b is not None):
        use_m = line1.m
        use_b = line1.b

    if all([
            line2.m is not None, line2.b is not None, use_m is None, use_b
            is None
    ]):
        use_m = line1.m
        use_b = line1.b

    if any([use_m is None, use_b is None]):
        return None, None

    # 計算
    if x is None:
        x = (line2.b - line1.b) / (line1.m - line2.m)

    if (y is None) and (x is not None):
        y = (line1.m * x) + line1.b

    return x, y
