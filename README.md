# AUTOMATIC HUMAN POSE RETARGETING ALGORITHM FOR MOTION CAPTURE SYSTEM

## Table of Contents
- [Description](#description)
- [Dependencies](#dependencies)

## Description
This repository contains the code used for transfer pose, the pose result from SMPLify-X (.pkl), to another model that has different rigging orientations.
### Data
The require data consist of ```config.yaml```, ```destination_model.fbx``` , and ```pose.pkl``` then place to the right folder. You should have the following folder structure:
```
lcmocap
├── config_files
│   ├── config.yaml
├── input
│   ├── destination
│       ├── destination_model.fbx
│   ├── source
│       ├── pose.pkl
```
### Fitting
Run the following command to execute the code:
```
python lcmocap/main.py --config config_files/config.yaml
```
where the **output/retargeting** is the folder that contains the resulting pose

## Dependencies
- [Blender Python API](https://docs.blender.org/api/current/index.html)
- [Numpy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
### Alternative way
You can using this on container by building Docker [Here](Dockerfile)
The code has been tested with python 3.7 and Blender API 2.91 on Ubuntu 18.04
