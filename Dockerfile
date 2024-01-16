FROM python:3.8

RUN apt-get update && apt-get install build-essential ffmpeg libsm6 libxext6 -y

RUN pip install \
  --disable-pip-version-check \
  --no-cache-dir \
  imageio>=2.25 \
  imageio-ffmpeg>=0.4.6 \
  numpy \
  tqdm \
  scikit-image \
  opencv-python \
  fastapi \
  python-multipart \
  'uvicorn[standard]'

COPY *.py .
COPY *.onnx .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]