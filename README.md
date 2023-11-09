# pytorch-table-recognition

Pytorch 表格辨識

## 版本需求

- Python >= `3.9`
- Pytorch >= `1.9.0`

## 安裝

1. 安裝 pytorch，可至 [PyTorch 官網](https://pytorch.org/) 根據對應版本產生對應的安裝指令。
2. 安裝 Python 套件

```shell
# 部署環境
$ pip install -r requirements.txt  # pip
$ poetry install --no-dev  # poetry

# 開發環境
$ pip install -r requirements-dev.txt  # pip
$ poetry install  # poetry
```

## 啟動服務

1. 啟動 API 服務，`python3 start_api.py`

## 使用方法

- 辨識表格

```shell
$ python3 recognize.py -i [IMAGE_PATH]
```

| 參數名稱             | 型態   | 必填 | 預設值                                             | 說明       | 備註 |
| -------------------- | ------ | ---- | -------------------------------------------------- | ---------- | ---- |
| `-i`, `--img_path`   | String | Y    |                                                    | 影像路徑   |      |
| `-c`, `--config`     | String | N    | `./config.yaml`                                    | 設定檔路徑 |      |
| `-o`, `--output_dir` | String | N    | `./outputs/recognition/{timestamp}_{img_filename}` | 輸出路徑   |      |

- 啟動 API 服務

```shell
# 預設會以 uvicorn 來啟動服務
$ python start_api.py -hs [HOST_IP] -p [PORT]
```

| 參數名稱        | 型態   | 必填 | 預設值    | 說明      | 備註 |
| --------------- | ------ | ---- | --------- | --------- | ---- |
| `-hs`, `--host` | String | N    | `0.0.0.0` | Host IP   |      |
| `-p`, `--port`  | String | N    | 8000      | Host Port |      |
