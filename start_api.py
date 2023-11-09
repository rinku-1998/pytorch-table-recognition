from api import create_app

app = create_app()

if __name__ == '__main__':

    import argparse
    import uvicorn

    # 1. 設定參數
    parser = argparse.ArgumentParser(
        description='Report recognition 2024 API Service')
    parser.add_argument('-hs',
                        '--host',
                        type=str,
                        default='0.0.0.0',
                        help='Host server listen')
    parser.add_argument('-p',
                        '--port',
                        type=int,
                        default=8000,
                        help='Port where server listen')

    args = parser.parse_args()

    # 2. 啟動服務
    uvicorn.run(app, host=args.host, port=args.port)