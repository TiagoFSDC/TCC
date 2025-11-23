FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependências do sistema necessárias para OpenCV, Torch Hub (git) e suporte a vídeo
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
        libglib2.0-0 \
        libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Instala PyTorch CPU separadamente para garantir download do repositório oficial
RUN pip install --no-cache-dir --pre \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.2.2 torchvision==0.17.2

# Instala demais dependências
RUN pip install --no-cache-dir -r requirements.txt

# Baixa o repositório YOLOv5 localmente para uso offline e instala dependências específicas
RUN git clone --depth 1 https://github.com/ultralytics/yolov5.git /opt/yolov5 \
    && pip install --no-cache-dir -r /opt/yolov5/requirements.txt

# Garante versões compatíveis de Numpy/Pandas após instalação do YOLOv5
RUN pip install --no-cache-dir --force-reinstall \
        numpy==1.26.4 \
        pandas==2.2.3

COPY . .

CMD ["python", "TCC.py"]





