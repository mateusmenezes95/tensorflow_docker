#!/bin/bash

. parameters.sh

docker build --force-rm -t $IMAGE_NAME \
    --build-arg USERNAME=$CONTAINER_USER \
    --build-arg ROS_DISTRO_ARG=$DISTRO \
    --build-arg UUID=$UUID \
    --build-arg UGID=$UGID \
    .
