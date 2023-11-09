import traceback
from api.enums.status_code import StatusCode
from api.enums.status_msg import StatusMsg
from api.exceptions.file_excpetion import UploadEmptyFileException
from api.models.base_response import BaseResponse
from exceptions.table_format import EmptyRowException, EmptyColException
from fastapi import FastAPI, Request, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def attach_exception_handlers(app: FastAPI) -> FastAPI:

    # TODO: 未來可以從這個事件攔截 401、 402 等已經被定義好的例外
    # HTTP 例外(FastAPI 預設攔截的例外)
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request,
                                     exception: HTTPException) -> JSONResponse:
        pass

    # Request 錯誤例外
    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
            request: Request,
            exception: RequestValidationError) -> JSONResponse:

        # 1. 格式化錯誤訊息
        msg = ''
        for error in exception.errors():

            location = error.get('loc')[0]
            field_name = error.get('loc')[1]

            msg += f'Request {location} 缺少 {field_name}，'

        # 2. 整理資料
        res = BaseResponse(
            code=StatusCode.REQUEST_MISSING_REQUIRED_PARAMETER.value)
        res.msg = msg

        return JSONResponse(jsonable_encoder(res))

    # 上傳檔案為空
    @app.exception_handler(UploadEmptyFileException)
    async def upload_empty_file_exception(
            request: Request,
            exception: UploadEmptyFileException) -> JSONResponse:

        res = BaseResponse(code=StatusCode.UPLOAD_EMPTY_FILE.value,
                           msg=StatusMsg.UPLOAD_EMPTY_FILE.value)
        return JSONResponse(jsonable_encoder(res))

    # 找不到水平線
    @app.exception_handler(EmptyRowException)
    async def empty_row_exception(
            request: Request, exception: EmptyRowException) -> JSONResponse:

        res = BaseResponse(code=StatusCode.TABLE_EMPTY_ROWS.value,
                           msg=StatusMsg.TABLE_EMPTY_ROWS.value)
        return JSONResponse(jsonable_encoder(res))

    # 找不到垂直線
    @app.exception_handler(EmptyColException)
    async def empty_col_exception(
            request: Request, exception: EmptyColException) -> JSONResponse:

        res = BaseResponse(code=StatusCode.TABLE_EMPTY_COLS.value,
                           msg=StatusMsg.TABLE_EMPTY_COLS.value)
        return JSONResponse(jsonable_encoder(res))

    # 全局例外
    @app.exception_handler(Exception)
    async def base_exception_handler(request: Request,
                                     exception: Exception) -> JSONResponse:

        # 1. 設定錯誤訊息
        msg = traceback.format_exc()

        # 2. 整理資料
        res = BaseResponse(code=StatusCode.UNDEFINED_EXCEPTION.value, msg=msg)
        return JSONResponse(jsonable_encoder(res))

    return app
