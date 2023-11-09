from enum import Enum


class StageName(Enum):

    ORIGINAL = '01_ORIGINAL'  # 原始照片
    DIRECTION_FIX = '02_DIRECTION_FIX'  # 方向修正
    RESIZE = '03_RESIZE'  # 縮放
    PAPER_MAP = '04_PAPER_MAP'  # 紙張像素圖
    SHAPE_FIX = '05_SHAPE_FIX'  # 變形修正
    TABLE_MAP = '06_TABLE_MAP'  # 表格線條像素圖
    LINE_ORIGINAL = '07_LINE_ORIGINAL'  # 原始線條
    LINE_RMSHORT = '08_LINE_RMSHORT'  # 移除過短線條
    LINE_MERGED = '09_LINE_MERGED'  # 合併線條
    LINE_RMOUTTABLE = '10_LINE_RMOUTTABLE'  # 移除表格外的線條
