FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel
USER root

RUN sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list && \
    sed -i "s@http://.*security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list

RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg curl vim wget htop

WORKDIR /app
RUN pip install uv requests torchaudio==2.2.1+cu121 -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html -i https://pypi.tuna.tsinghua.edu.cn/simple 

COPY speechgpt/requirements.txt /app/requirements.txt
RUN /opt/conda/bin/uv --version
RUN uv -v pip  install  --system -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN uv -v pip  install  --system gradio modelscope -i https://mirrors.aliyun.com/pypi/simple/ --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple

COPY ./speechgpt /app/speechgpt
