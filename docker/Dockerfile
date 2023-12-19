FROM nvcr.io/nvidia/tensorrt:22.10-py3

ARG CUDA=11.3
ARG PYTHON_VERSION=3.8
ARG TORCH_VERSION=1.10.0
ARG TORCHVISION_VERSION=0.11.0
ENV FORCE_CUDA="1"

ENV DEBIAN_FRONTEND=noninteractive

### change the system source for installing libs
ARG USE_SRC_INSIDE=false
RUN if [ ${USE_SRC_INSIDE} == true ] ; \
    then \
        sed -i s/archive.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list ; \
        sed -i s/security.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list ; \
        echo "Use aliyun source for installing libs" ; \
    else \
        echo "Keep the download source unchanged" ; \
    fi

### update apt and install libs
RUN sed -i s:/archive.ubuntu.com:/mirrors.tuna.tsinghua.edu.cn/ubuntu:g /etc/apt/sources.list
RUN cat /etc/apt/sources.list
RUN chmod 777 /tmp
RUN apt-get clean && apt-get update &&\
    apt-get install -y vim libsm6 libxext6 libxrender-dev libgl1-mesa-glx git wget libssl-dev libopencv-dev libspdlog-dev --no-install-recommends &&\
    rm -rf /var/lib/apt/lists/*

### install miniconda
RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} conda-build pyyaml numpy ipython cython typing typing_extensions mkl mkl-include ninja && \
    /opt/conda/bin/conda clean -ya

### configue ssh and set root password to 123
RUN \
  DEBIAN_FRONTEND=noninteractive apt-get update && \ 
  DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y vim python3 python3-pip && \
  sed -i "s/UsePAM yes/#UsePAM yes/" /etc/ssh/sshd_config && \
  sed -i "s/PermitRootLogin/#PermitRootLogin/" /etc/ssh/sshd_config && \
  echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config && \
  echo 'UsePAM no' >> /etc/ssh/sshd_config && \
  echo 'service ssh start' >> /etc/bash.bashrc && \
  echo 'root:123' | chpasswd  

### pytorch
RUN /opt/conda/bin/conda install pytorch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} cudatoolkit=${CUDA} -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/linux-64/
ENV PATH /opt/conda/bin:$PATH

ENV BACKUP_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real/:$LD_LIBRARY_PATH

### add path
RUN echo 'export PATH=/opt/conda/bin:/opt/tensorrt/bin:/usr/local/mpi/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:$PATH' >> ~/.bashrc

### add others packages
RUN pip install pycuda \
    numpy \
    scikit-image \
    tensorboard==2.6.0 \
    pandas\
    opencv-python \
    timm \
    matplotlib \


