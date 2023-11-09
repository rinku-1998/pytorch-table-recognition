from api.exceptions.file_excpetion import UploadEmptyFileException
from api.models.base_response import BaseResponse
from api.utils.img_util import load_form_image
from exceptions.table_format import EmptyColException, EmptyRowException
from fastapi import APIRouter, UploadFile, File, Request
from loguru import logger

# 設定物件
router = APIRouter()


@router.get('/health', response_model=BaseResponse)
def get_health():

    # 1. 整理資料
    base_res = BaseResponse()

    return base_res


@router.post('/recognize', response_model=BaseResponse)
def recognize_table(request: Request, file: UploadFile = File()):

    # 1. 讀取圖片
    byte_file = file.file.read()

    # 檢查圖片是否為空
    if len(byte_file) == 0:
        raise UploadEmptyFileException

    img = load_form_image(byte_file)

    # 3. 辨識表格
    tr = request.app.state.table_recognizer
    table = None
    try:
        table = tr.run(img)
    except Exception as e:

        # 取得錯誤訊息
        isin_defined_exception = any(
            isinstance(e, _) for _ in (EmptyColException, EmptyRowException))

        err_msg = e.msg if isin_defined_exception else str(e)
        logger.error(f'辨識失敗，錯誤為 {err_msg}')
        raise e

    # 4. 整理資料
    base_res = BaseResponse(data=table)
    logger.success('辨識成功！')

    return base_res
