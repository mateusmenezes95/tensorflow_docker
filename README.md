# Tensorflow Docker Container

This repository contains Dockerfiles and helper bash scripts to create images to run ROS2. The bash scripts were developed for the user to choose the parameters used in the construction of Docker images and in the execution of the containers.

This repository is ready to create images ROS2 Foxy Fitzroy.

## Nvidia Docker

Check which version of the Nvidia driver is installed on your machine.

```console
nvidia-smi
```

Thus, pull the CUDA image that is compatible with the version of the Nvidia driver installed on your machine. For example, if the Nvidia driver version shows the CUDA version 11.7, run the following command.

```console
docker pull nvidia/cuda:11.1.1-runtime-ubuntu20.04
```

To use the Nvidia Docker, you need to install the Nvidia Container Toolkit. Follow the instructions in the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/user-guide.html#daemon-configuration-file).

Then, run the following command to check if the Nvidia Docker is working properly.

```console
docker run --gpus all --rm nvidia/cuda:11.1.1-runtime-ubuntu20.04 nvidia-smi
```

## Usage

Inside the folder containing the bash scripts and Dockerfile, build the image by running

```console
./build_image.sh
```

After the build phase, just run the container

```console
./run-container.sh
```
