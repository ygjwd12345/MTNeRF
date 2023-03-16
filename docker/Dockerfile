#FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5"

RUN apt-get clean && apt-get update && apt-get install -y \
    tree \
    vim \
    python3-tk \
    git \
    unzip \
    wget \
    screen \
    libgl1-mesa-dev\
    libgtk2.0-dev\
    cmake\
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev \
    imagemagick

RUN pip install pycocotools \
    opencv-contrib-python
RUN apt-get install libatlas-base-dev libsuitesparse-dev
WORKDIR "/home"
RUN git clone https://ceres-solver.googlesource.com/ceres-solver
RUN cd ceres-solver && cmake . && make -j && make install
#RUN cd ./ceres-solver

#RUN mkdir build
#RUN cd build
#RUN cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF
#RUN make -j
#RUN make install
WORKDIR "/home"
RUN git clone -b pointvisibility https://github.com/B1ueber2y/colmap.git
RUN cd colmap && cmake . && make -j && make install
#RUN git checkout dev
#RUN mkdir build
#RUN cd build
#RUN cmake ..
#RUN make -j
#RUN make install
WORKDIR "/home"
RUN pip install --no-cache-dir Cython \
    wandb \
    torchvision \
    ninja \
    scikit-image==0.15.0 \
    opencv-python \
    pandas \
    tqdm \
    scikit-learn \
    inplace_abn \
    tensorboard \
    tensorboardX \
    lpips==0.1.3 \
    wget \
    configargparse \
    h5py \
    imageio-ffmpeg \
    imageio