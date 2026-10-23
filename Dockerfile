# Inspection API on CPU. Mount a trained run and its dataset:
#   docker build -t dpf-preinspect .
#   docker run -p 8000:8000 -v "$PWD/runs:/app/runs" -v "$PWD/data:/app/data" dpf-preinspect
FROM python:3.11-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PREINSPECT_DEVICE=cpu PREINSPECT_ROOT=/app PREINSPECT_OFFLINE_WEIGHTS=1
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -e .
EXPOSE 8000
CMD ["preinspect", "api", "--config", "demo", "--host", "0.0.0.0"]
