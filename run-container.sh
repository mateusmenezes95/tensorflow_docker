#!/bin/bash

. parameters.sh

if [ ! -d "${VOLUME_PATH}" ]; then
    mkdir -p "${VOLUME_PATH}"

    if [ ! -d "${VOLUME_PATH}" ]; then
        echo "Could not create ${PROJECT}'s home"
        exit 1
    fi
fi

if [ ! -d "${VOLUME_PATH}/.ssh" ]; then
    if [ ! -d "${HOME}/.ssh" ]; then
        printf "Please, setup ssh keys before running the container\n"
        exit 1
    fi
    cp -R "${HOME}/.ssh" "${VOLUME_PATH}/"
fi

if [ ! -f "${VOLUME_PATH}/.gitconfig" ]; then
    if [ ! -f "${HOME}/.gitconfig" ]; then
        printf "Please, setup git before running the container\n\n"
        printf "Use: \n"
        printf "git config --global user.name \"John Doe\"\n"
        printf "git config --global user.email johndoe@example.com\n"

        exit 1
    fi
    cp "${HOME}/.gitconfig" "${VOLUME_PATH}/"
fi

# Solution found in https://answers.ros.org/question/301056/ros2-rviz-in-docker-container/
XSOCK=/tmp/.X11-unix

if [ ! "$(docker ps -q -f name=${IMAGE_NAME}_container)" ]; then
    docker run -it --rm \
        --env="SSH_AUTH_SOCK=$SSH_AUTH_SOCK" \
        --env="TERM" \
        --env=DISPLAY=$DISPLAY \
        --gpus="all"\
        --name="${IMAGE_NAME}_container" \
        --net=host \
        --privileged \
        --user="${CONTAINER_USER}" \
        --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
        --volume="$XSOCK:$XSOCK:rw" \
        --volume="/etc/localtime:/etc/localtime:ro" \
        --volume="$(dirname $SSH_AUTH_SOCK):$(dirname $SSH_AUTH_SOCK)" \
        --volume="${VOLUME_PATH}:/home/${CONTAINER_USER}:rw" \
        --workdir="/home/${CONTAINER_USER}" \
        "${IMAGE_NAME}:latest"
else
    docker exec -ti ${IMAGE_NAME}_container /bin/bash
fi
