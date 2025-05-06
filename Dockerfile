FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update -y && apt-get install -y python3-pip

COPY requirements.txt .

RUN pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

RUN pip install --no-cache-dir -r requirements.txt

COPY test-voice.py .

ENTRYPOINT [ "python3", "-m", "test-voice" ]