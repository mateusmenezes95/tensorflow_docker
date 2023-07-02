FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update \
    && apt-get install -q -y --no-install-recommends \
    dirmngr \
    sudo \
    locales \
    gnupg2 \
    lsb-release \
    && apt-get -y autoremove \
    && apt-get clean autoclean \
    && rm -fr /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*

# # setup timezone
# RUN echo 'Etc/UTC' > /etc/timezone && \
#     ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
#     apt-get update && \
#     apt-get install -q -y --no-install-recommends tzdata \
#     && apt-get -y autoremove \
#     && apt-get clean autoclean \
#     && rm -fr /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*

# Set env variables
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    XVFB_WHD="1920x1080x24"\
    LIBGL_ALWAYS_SOFTWARE="1"

ARG USERNAME
ARG UUID
ARG UGID

RUN useradd -m $USERNAME && \
    echo "$USERNAME:$USERNAME" | chpasswd && \
    usermod --shell /bin/bash $USERNAME && \
    usermod -aG sudo $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME && \
    usermod --uid $UUID $USERNAME && \
    groupmod --gid $UGID $USERNAME

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    git \
    vim \
    wget \
    bash-completion \
    build-essential \
    tree \
    python3-argcomplete \
    && apt-get -y autoremove \
    && apt-get clean autoclean \
    && rm -fr /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*

RUN pip3 install -U \
    matplotlib \
    setuptools \
    && rm -rf /root/.cache/pip

# Solution found in https://answers.ros.org/question/301056/ros2-rviz-in-docker-container/
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# setup entrypoint
COPY ./entrypoint.sh /

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
