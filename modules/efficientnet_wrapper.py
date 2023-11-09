import torch
from efficientnet_pytorch import EfficientNet
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from utils import torch_util


class EfficientNetWrapper:

    def __init__(self,
                 weight_path: str,
                 num_classes: int,
                 device: str = 'cpu') -> None:

        # 1. 載入模型
        model = EfficientNet.from_name('efficientnet-b0',
                                       num_classes=num_classes)

        # 2. 載入權重
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)

        # 3. 設定前處理
        transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.model = model
        self.device = device
        self.transform = transform

    def preprocess(self, img: Image) -> torch.Tensor:
        """前處理

        Args:
            img (Image): 影像

        Returns:
            torch.Tensor: 前處理後的影像
        """

        img_tensor = self.transform(img)
        return img_tensor

    def predict_img(self, img: Image) -> int:
        """預測影像

        Args:
            img (Image): 影像

        Returns:
            int: 預測標籤
        """

        # 1. 前處理
        img_input = self.preprocess(img)

        # 2. 推論
        img_input = torch.unsqueeze(img_input, 0)

        self.model.eval()
        img_input = img_input.to(self.device)
        outputs = self.model(img_input)
        _, predict_labels = torch.max(outputs, 1)

        # 3. 調整格式
        labels = torch_util.tensor_to_numpy(predict_labels)

        return labels[0]