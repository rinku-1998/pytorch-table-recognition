import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Line:

    x_start: int = field(init=True)  # x 起始座標
    y_start: int = field(init=True)  # y 起始座標
    x_end: int = field(init=True)  # x 結束座標
    y_end: int = field(init=True)  # y 結束座標
    length: int = field(init=False)  # 線條長度
    m: Optional[float] = field(init=False)  # 點斜式斜率
    b: Optional[float] = field(init=False)  # 點斜式常數

    def calculate_fn_mb(
            self, point1: List[int],
            point2: List[int]) -> Tuple[Optional[float], Optional[float]]:

        #1. 取出資料
        x1, y1 = point1
        x2, y2 = point2

        # 2. 檢查是否是垂直線
        if x1 == x2:
            return None, None

        # 3. 檢查是否是水平線
        if y1 == y2:
            return 0.0, y1

        # 4. 計算點斜式的 m 跟 b
        delta_x = x2 - x1
        delta_y = y2 - y1
        m = delta_y / delta_x if delta_x != 0 else 0.0
        b = y1 - (m * x1)

        return m, b

    def __post_init__(self):

        # 1. 計算線段長度
        square = math.pow((self.x_start - self.x_end), 2) + math.pow(
            (self.y_start - self.y_end), 2)
        length = math.pow(square, 0.5)
        self.length = length

        # 2. 計算點斜式的 m 跟 b
        m, b = self.calculate_fn_mb((self.x_start, self.y_start),
                                    (self.x_end, self.y_end))
        self.m = m
        self.b = b
