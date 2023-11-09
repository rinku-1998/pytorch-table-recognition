import json
import os
from pathlib import Path
from typing import Optional


class SmartJSONEncoder(json.JSONEncoder):

    def default(self, obj):
        if hasattr(obj, '__jsonencode__'):
            return obj.__jsonencode__()

        if hasattr(obj, '__dict__'):
            return obj.__dict__

        if isinstance(obj, set):
            return list(obj)

        return json.JSONEncoder.default(self, obj)


def save_json(data: any, save_path: str) -> None:
    """儲存 json 檔

    Args:
        data (any): 資料物件
        save_path (str): 存檔路徑
    """

    # 1. 檢查檔案是否存在，如果不存在就建立一個
    if not Path(save_path).parent.exists():
        os.makedirs(str(Path(save_path).parent))

    # 2. 存檔
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, cls=SmartJSONEncoder)


def load_json(json_path: str) -> Optional[any]:

    # 1. 檢查路徑是否存在
    if not Path(json_path).exists():
        return None

    # 2. 讀取檔案
    data = None
    with open(str(json_path), 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data
