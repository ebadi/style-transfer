FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    requests==2.31.0 \
    Pillow==10.2.0 \
    PyYAML==6.0.1 \
    tqdm==4.66.2 \
    numpy==1.26.4 \
    opencv-python-headless==4.9.0.80

RUN mkdir -p /scripts /data /workflows

CMD ["python3", "/scripts/run_pipeline.py"]
