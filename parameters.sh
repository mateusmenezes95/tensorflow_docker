UUID="$(id -u)"
UGID="$(id -g)"

# Change when desirable
IMAGE_NAME="tf_container"
CONTAINER_USER="tensorflow"

VOLUME_PATH=${HOME}/container_volumes/${IMAGE_NAME}
