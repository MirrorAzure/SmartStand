FROM mirrorazure/cuda12.3.2-torch2.8.0-ubuntu22.04 AS base

WORKDIR /app

RUN apt-get update -y && apt-get install -y python3-pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base

COPY test-voice.py .

ENTRYPOINT [ "python3", "-m", "test-voice" ]