from utils.yaml_util import load_yaml


def load_config(config_path: str) -> dict:

    # 1. 讀取 yaml 檔
    config = load_yaml(config_path)

    # 2. 將影像尺寸設定為 tuple
    preprocess = config.get('preprocess')
    if preprocess is not None:
        resize_size = preprocess.get('resize_size')
        resize_size = eval(resize_size)
        config['preprocess']['resize_size'] = resize_size

    # 3. 將顏色尺寸設定為 tuple
    keys = ('row_color', 'col_color', 'cand_color', 'prof_color')
    draw = config.get('draw')
    if draw is not None:
        for k in keys:
            v = draw.get(k)
            tuple_value = eval(v)

            config['draw'][k] = tuple_value

    # 4. 將有使用到模型的設定都加上 device 參數
    device = config.get('model').get('device')
    config['paper_direction']['model']['device'] = device
    config['paper']['model']['device'] = device
    config['table_line']['model']['device'] = device

    # 5. 將有繪圖的模組加上 color 參數
    draw = config.get('draw')
    if draw:
        config['table_line']['build'].update(draw)

    return config
