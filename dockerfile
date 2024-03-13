FROM apache/spark-py:v3.3.2

USER root
WORKDIR /workspace

# (Optional)
RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN apt-get update \
    && apt-get install -y gcc ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libxrender-dev vim \
    build-essential libboost-all-dev cmake curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install img2dataset huggingface_hub[hf_transfer]
CMD bash
