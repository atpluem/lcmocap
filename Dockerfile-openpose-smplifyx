FROM nvidia/cuda:10.0-cudnn7-devel

# Install dependencies | Caffe-cuda (Asia: 6 Bangkok: 12)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git wget nano \
    python3-dev python3-pip g++ make libprotobuf-dev protobuf-compiler \
    libopencv-dev libgoogle-glog-dev libboost-all-dev libcaffe-cuda-dev \
    libhdf5-dev libatlas-base-dev

# For python api
RUN pip3 install numpy scikit-build \
    --upgrade pip setuptools wheel opencv-python

# Cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.16.0/cmake-3.16.0-Linux-x86_64.tar.gz && \
    tar xzf cmake-3.16.0-Linux-x86_64.tar.gz -C /opt && \
    rm cmake-3.16.0-Linux-x86_64.tar.gz
ENV PATH="/opt/cmake-3.16.0-Linux-x86_64/bin:${PATH}"

# ===== Openpose =====
WORKDIR /lcmocap-env
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git

# Openpose - build
WORKDIR /lcmocap-env/openpose/build
RUN cmake -DBUILD_PYTHON=ON .. && \
    sed -ie 's/set(AMPERE "80 86")/#&/g'  ../cmake/Cuda.cmake && \
    sed -ie 's/set(AMPERE "80 86")/#&/g'  ../3rdparty/caffe/cmake/Cuda.cmake && \
    make -j `nproc`

# Openpose - setup env. for pyopenpose
WORKDIR /lcmocap-env/openpose/build/python/openpose
RUN make install && \
    cd /usr/local/python && \
    cp ./pyopenpose.cpython-36m-x86_64-linux-gnu.so /usr/local/lib/python3.6/dist-packages && \
    cd /usr/local/lib/python3.6/dist-packages && \
    ln -s pyopenpose.cpython-36m-x86_64-linux-gnu.so pyopenpose && \
    export LD_LIBRARY_PATH=/openpose/build/python/openpose

# ===== Smplify-x =====
WORKDIR /lcmocap-env/
RUN git clone https://github.com/vchoutas/smplify-x.git

# Smplify-x - config
WORKDIR /lcmocap-env/smplify-x
RUN pip3 install -r requirements.txt && \
    pip3 install -r optional-requirements.txt && \
    pip3 install smplx[all] && \
    git clone https://github.com/nghorbani/human_body_prior.git && \
    git clone https://github.com/nghorbani/homogenus && \
    git clone https://github.com/NVIDIA/cuda-samples.git && \
    git clone https://github.com/vchoutas/torch-mesh-isect

# Smplify-x - install human_body_prior (VPoser)
WORKDIR /lcmocap-env/smplify-x/human_body_prior/
RUN pip3 install -r requirements.txt && \
    python3 setup.py develop && \
    cd ..

# Smplify-x - install homogenus
WORKDIR /lcmocap-env/smplify-x/homogenus
RUN python3 setup.py install && \
    cd ..

# Smplify-x - install torch-mesh-isect
WORKDIR /lcmocap-env/smplify-x/
RUN pip3 install -r requirements.txt && \
    pip install -r optional-requirements.txt && \
    python3 setup.py install