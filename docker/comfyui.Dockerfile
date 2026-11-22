FROM python:3.12-slim

# Pin ComfyUI so every machine runs the same version.
ARG COMFYUI_VERSION=v0.46.2
# PyTorch CUDA build. cu128 needs an NVIDIA driver >= 570;
# for older drivers build with --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${COMFYUI_VERSION} \
        https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI

WORKDIR /opt/ComfyUI
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url ${TORCH_INDEX_URL} \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 8188
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--preview-method", "none"]
