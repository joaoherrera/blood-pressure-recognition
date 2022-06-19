FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04

# Workaround to deal with nvidia obsolete apt keys.
RUN rm /etc/apt/sources.list.d/cuda.list && \
    rm /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-key del 7fa2af80 && \
    apt-get update && apt-get install -y --no-install-recommends wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb

# Install locale and set language
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y locales \
    && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && dpkg-reconfigure --frontend=noninteractive locales \
    && update-locale LANG=en_US.UTF-8

ENV LANG en_US.UTF-8 
ENV LC_ALL en_US.UTF-8

# Install base dependencies.
RUN apt-get update -y && apt-get install -y \
    software-properties-common \
    build-essential \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \    
    curl \
    unzip \
    git \
    g++ \
    cmake \
    wget \
    bash-completion \
    python3.8 \
    python3.8-dev \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Make Python 3.8 the default Python
RUN update-alternatives --install \
    /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --set python /usr/bin/python3.8

COPY ./requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && rm requirements.txt

ARG USERNAME=blood_pressure_recognizer
ARG UID=1000
ARG GID=1000

RUN groupadd -g ${GID} -o ${USERNAME} && \
    useradd -ms /bin/bash -u ${UID} -g ${GID} -o ${USERNAME}

USER ${USERNAME}

WORKDIR /src