import logging
import sys
from api.config import Config
from api.logging_handler import InterceptHandler, format_record
from api.exception_handler import attach_exception_handlers
from helpers.config_helper import load_config
from fastapi import FastAPI
from loguru import logger
from modules.recognize_table import TableRecognizer


def create_app() -> FastAPI:

    # 1. 建立 App 物件
    title = 'Table recognition'
    ver = '0.1.0'
    app = FastAPI(title=title, version=ver)

    # 2. 設定日誌格式
    # 格式
    logger.configure(handlers=[{
        "sink": sys.stdout,
        "level": logging.DEBUG,
        "format": format_record
    }])

    # Uvicorn
    logging.getLogger().handlers = [InterceptHandler()]
    logging.getLogger("uvicorn.access").handlers = [InterceptHandler()]

    # 新增儲存位置
    logger.add("logs/run.log",
               rotation="500MB",
               encoding="utf-8",
               enqueue=True,
               retention="15 days")

    # 3. 初始化模型
    @app.on_event('startup')
    async def startup_event():

        # 設定物件
        module_config = load_config(Config.module_config_path)
        tr = TableRecognizer(module_config)
        app.state.table_recognizer = tr

    # 4. 掛載路由
    from api.route import router
    app.include_router(router)

    # 4. 設定例外攔截器
    app = attach_exception_handlers(app)

    return app
