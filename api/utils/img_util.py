import cv2
import numpy as np
import io
from PIL import Image


def load_form_image(byte_file: bytes) -> np.ndarray:
    """載入 Form 表單的圖片

    Args:
        byte_file (bytes): Form 表單的圖片

    Returns:
        np.ndarray: 圖片物件
    """

    # 1. 讀取圖片
    img_pil = Image.open(io.BytesIO(byte_file))

    # 2. 轉為 Numpy 物件
    img = np.array(img_pil)

    # 3. 調整色彩通道(RGB -> BGR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img
