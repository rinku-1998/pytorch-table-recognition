import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ai_models.unet_model.unet import UNet
from PIL import Image
from torchvision.transforms import ToTensor
from typing import Tuple


class UNetWrapper(object):

    def __init__(self,
                 weight_path: str,
                 num_classes: int,
                 img_size: Tuple[int, int] = (512, 512),
                 device='cpu',
                 threshold: float = 0.5) -> None:

        # 1 載入模型
        model = UNet(n_channels=3, n_classes=num_classes, bilinear=False)
        model.to(device)

        # 2. 載入權重
        state_dict = torch.load(weight_path, map_location=device)
        _ = state_dict.pop('mask_values', [0, 1])
        model.load_state_dict(state_dict)

        self.device = device
        self.model = model
        self.img_size = img_size
        self.threshold = threshold

    def _preprocess(self, img: Image) -> torch.Tensor:
        """前處理

        Args:
            img (Image): 影像

        Returns:
            torch.Tensor: 影像 Tensor
        """

        # 1. 整理格式
        to_tensor = ToTensor()
        img_torch = to_tensor(img)

        return img_torch

    def predict_img(self, img: Image) -> np.ndarray:

        # 1. 調整影像維度
        img_input = self._preprocess(img)
        img_input = img_input.unsqueeze(0)

        # 2. 推論
        self.model.eval()
        img_input = img_input.to(self.device, dtype=torch.float32)

        with torch.no_grad():

            output = self.model(img_input).cpu()
            output = F.interpolate(output, (img.size[1], img.size[0]),
                                   mode='bilinear')

        mask = None
        if self.model.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > self.threshold

        # 3. 還原影像尺寸
        mask_np = mask[0].long().squeeze().numpy()
        mask_np = mask_np.astype(np.uint8)

        return mask_np
