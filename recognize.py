import os
import shutil
from datetime import datetime
from dto.table import Table
from helpers.config_helper import load_config
from modules.recognize_table import TableRecognizer
from pathlib import Path
from utils.img_util import load_img


def run(config_path: str, img_path: str, output_dir: str) -> Table:

    # 1. 載入設定檔
    config = load_config(config_path)

    # 2. 載入圖片
    img = load_img(img_path)

    # 3. 建立表格辨識模組
    tr = TableRecognizer(config)

    # 4. 執行辨識
    ticket_results = tr.run(img, output_dir=output_dir)

    return ticket_results


if __name__ == '__main__':

    # 1. 設定執行參數
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--img_path',
                        type=str,
                        required=True,
                        help='Path to image')
    parser.add_argument('-c',
                        '--config_path',
                        type=str,
                        default='config.yaml',
                        help='Path to config file in YAML')
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        default='outputs/recognize',
                        help='Directory to output files')

    args = parser.parse_args()

    # 2. 設定輸出目錄
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    folder_name = f'{timestamp}_{Path(args.img_path).stem}'
    output_dir = Path(args.output_dir, folder_name)

    # 新增路徑
    if not output_dir.exists():
        os.makedirs(str(output_dir))

    # 3. 儲存設定檔
    config_spath = output_dir.joinpath('config.yaml')
    shutil.copy(args.config_path, str(config_spath))

    # 4. 執行辨識表格
    run(args.config_path, args.img_path, output_dir)
