FROM python:3.7

WORKDIR /retargeting
ADD https://github.com/TylerGubala/blenderpy/releases/download/v2.91a0/bpy-2.91a0-cp37-cp37m-manylinux2014_x86_64.whl bpy-2.91a0-cp37-cp37m-manylinux2014_x86_64.whl
RUN python -m pip install -U pip setuptools wheel
RUN python -m pip install bpy-2.91a0-cp37-cp37m-manylinux2014_x86_64.whl && \
    bpy_post_install && rm bpy-2.91a0-cp37-cp37m-manylinux2014_x86_64.whl
RUN apt-get update && apt-get install libgl1 -y

# install addition python library
RUN python -m pip install torch opencv-python tqdm loguru omegaconf trimesh \
    scipy matplotlib seaborn networkx sklearn

# COPY ./lcmocap ./lcmocap
# COPY ./config_files ./config_files
# COPY ./input ./input
# COPY ./output ./output

# CMD ["python", "lcmocap/main.py", "--config", "config_files/config.yaml"]