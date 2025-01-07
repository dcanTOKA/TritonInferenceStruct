FROM nvcr.io/nvidia/tritonserver:23.05-py3

COPY requirements.txt /opt/tritonserver/requirements.txt

RUN apt-get update && apt-get install -y libgl1

RUN pip install --upgrade pip && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install 'git+https://github.com/facebookresearch/detectron2.git' && \
    pip install paddleocr && \
    pip install paddlepaddle-gpu && \
    pip install -r /opt/tritonserver/requirements.txt

ENV PYTHONUNBUFFERED=1

CMD ["tritonserver", "--model-repository=/model_repository", "--model-control-mode=poll", "--repository-poll-secs=10"]