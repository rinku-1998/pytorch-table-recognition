FROM python:3.9-slim
LABEL MAINTAINER="ais"

# 1. 安裝 Ubuntu 套件
RUN apt update && \
    apt install -y python3-pip python3-dev && \
    apt install -y ffmpeg libsm6 libxext6 

# 2. 設定
EXPOSE 8000

# 3. 複製專案檔案
COPY . /root/report-recognition-2024

# 4. 安裝 Python 套件
WORKDIR /root/report-recognition-2024
RUN pip3 install -r requirements.txt   

# 5. 啟動服務
CMD ["uvicorn", "start_api:app", "--host", "0.0.0.0", "--port", "8000"]