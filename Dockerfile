FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /root

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential pkg-config locales tzdata \
    python3 python3-dev python3-pip python3-wheel python3-venv \
    python3-neovim git curl jq less \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN locale-gen ja_JP.UTF-8

ENV LANG=ja_JP.UTF-8
ENV TZ=Asia/Tokyo
ENV PATH=${PATH}:/root/.local/bin

RUN ln -sf /usr/bin/python3.10 /usr/bin/python3
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip3 install --upgrade pip
RUN pip3 --no-cache-dir install tensorflow==2.17.1
RUN pip3 --no-cache-dir install \
        'numpy<2.0.0' \
        'Cython<3.1' \
        sgf \
        pandas \
        pyarrow \
        pytest \
        tqdm

RUN ["bash"]

