# docker run --shm-size 64G --gpus all -it -d -p 8801:8801 -v /data:/app  --privileged=true --name [docker container name] [docker container id]
FROM pytorch:2.1.2-cuda12.1-cudnn8-devel

WORKDIR /app

COPY . /app
COPY /data/models/SpeechGPT-7B-cm /app/SpeechGPT-7B-cm
COPY /data/models/SpeechGPT-7B-com /app/SpeechGPT-7B-com
COPY /data/xtpan/models/speech2unit /app/speech2unit
COPY /data/xtpan/models/vocoder /app/vocoder

RUN mkdir -p /app/input
RUN mkdir -p /app/output

RUN pip3 install -r requirements.txt --index-url http://mirrors.cloud.tencent.com/pypi/simple --trusted-host mirrors.cloud.tencent.com

RUN echo -e '\n\n\n\n\n\n\n' | openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 365 -nodes

CMD ["sh", "-c", "CUDA_VISIBLE_DEVICES=0 python3 -m speechgpt.src.infer.web_infer --model-name-or-path /app/SpeechGPT-7B-cm --lora-weights /app/SpeechGPT-7B-com --s2u-dir /app/speech2unit --vocoder-dir /app/vocoder --output-dir /app/output --input-dir /app/input/audio.wav --port 8801 --https-dir /app/data/"]