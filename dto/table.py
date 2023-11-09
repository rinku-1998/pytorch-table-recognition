from dataclasses import dataclass, field
from dto.line import Line
from typing import List, Tuple


@dataclass
class Table:

    size: Tuple[int, int] = field(init=False)  # 表格尺寸(高, 寬)
    row_lines: List[Line] = field(init=True)  # 水平線線條物件列表
    col_lines: List[Line] = field(init=True)  # 垂直線線條物件列表

    def __post_init__(self):

        # 1. 計算表格尺寸
        xs: List[int] = list()
        xs.extend([l.x_start for l in self.row_lines])
        xs.extend([l.x_end for l in self.row_lines])

        ys: List[int] = list()
        ys.extend([l.y_start for l in self.col_lines])
        ys.extend([l.y_end for l in self.col_lines])

        w = max(xs) - min(xs) if xs else 0
        h = max(ys) - min(ys) if ys else 0
        self.size = (h, w)
