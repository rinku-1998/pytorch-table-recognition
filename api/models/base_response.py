from api.enums.status_code import StatusCode
from api.enums.status_msg import StatusMsg
from pydantic import BaseModel
from typing import Any, Optional


class BaseResponse(BaseModel):

    code: Optional[int] = StatusCode.SUCCESS.value  # 狀態代碼
    msg: Optional[str] = StatusMsg.SUCCESS.value  # 狀態訊息
    data: Optional[Any] = None  # 資料
