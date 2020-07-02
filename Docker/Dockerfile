FROM nvidia/cudagl:10.1-devel-ubuntu18.04

ARG SOURCEFORGE=https://sourceforge.net/projects
ARG TURBOVNC_VERSION=2.1.2
ARG VIRTUALGL_VERSION=2.5.2
ARG LIBJPEG_VERSION=1.5.2
ARG WEBSOCKIFY_VERSION=0.8.0
ARG NOVNC_VERSION=1.0.0
ARG LIBARMADILLO_VERSION=6

#---------------------------------------------------------------------
# Install Linux stuff
#---------------------------------------------------------------------
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates curl wget less sudo lsof git net-tools nano psmisc xz-utils nemo vim net-tools iputils-ping traceroute htop \
    lubuntu-core chromium-browser xterm terminator zenity make cmake gcc libc6-dev \
    x11-xkb-utils xauth xfonts-base xkb-data \
    mesa-utils xvfb libgl1-mesa-dri libgl1-mesa-glx libglib2.0-0 libxext6 libsm6 libxrender1 \
    libglu1 libglu1:i386 libxv1 libxv1:i386 \
    python python-numpy libpython-dev libsuitesparse-dev libgtest-dev \
    libeigen3-dev libsdl1.2-dev libignition-math2-dev  libsdl-image1.2-dev libsdl-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

#---------------------------------------------------------------------
# Install gtest
#---------------------------------------------------------------------
RUN /bin/bash -c "cd /usr/src/gtest && cmake CMakeLists.txt && make && cp *.a /usr/lib"

#---------------------------------------------------------------------
# Install VirtualGL and TurboVNC
#---------------------------------------------------------------------
RUN cd /tmp && \
    curl -fsSL -O ${SOURCEFORGE}/turbovnc/files/${TURBOVNC_VERSION}/turbovnc_${TURBOVNC_VERSION}_amd64.deb \
    -O ${SOURCEFORGE}/libjpeg-turbo/files/${LIBJPEG_VERSION}/libjpeg-turbo-official_${LIBJPEG_VERSION}_amd64.deb \
    -O ${SOURCEFORGE}/virtualgl/files/${VIRTUALGL_VERSION}/virtualgl_${VIRTUALGL_VERSION}_amd64.deb \
    -O ${SOURCEFORGE}/virtualgl/files/${VIRTUALGL_VERSION}/virtualgl32_${VIRTUALGL_VERSION}_amd64.deb && \
    dpkg -i *.deb && \
    rm -f /tmp/*.deb && \
    sed -i 's/$host:/unix:/g' /opt/TurboVNC/bin/vncserver
ENV PATH ${PATH}:/opt/VirtualGL/bin:/opt/TurboVNC/bin

#---------------------------------------------------------------------
# Install noVNC
#---------------------------------------------------------------------
RUN curl -fsSL https://github.com/novnc/noVNC/archive/v${NOVNC_VERSION}.tar.gz | tar -xzf - -C /opt && \
    curl -fsSL https://github.com/novnc/websockify/archive/v${WEBSOCKIFY_VERSION}.tar.gz | tar -xzf - -C /opt && \
    mv /opt/noVNC-${NOVNC_VERSION} /opt/noVNC && \
    chmod -R a+w /opt/noVNC && \
    mv /opt/websockify-${WEBSOCKIFY_VERSION} /opt/websockify && \
    cd /opt/websockify && make && \
    cd /opt/noVNC/utils && \
    ln -s /opt/websockify

COPY requirements/xorg.conf /etc/X11/xorg.conf
COPY requirements/index.html /opt/noVNC/index.html

# Expose whatever port NoVNC will serve from. In our case it will be 40001, see ./start_desktop.sh
EXPOSE 40001
ENV DISPLAY :1

#---------------------------------------------------------------------
# Install desktop files for this user
#---------------------------------------------------------------------
RUN mkdir -p /root/Desktop
COPY ./requirements/terminator.desktop /root/Desktop
RUN mkdir -p /root/.config/terminator
COPY ./requirements/terminator_config /root/.config/terminator/config
COPY ./requirements/self.pem /root/self.pem

# Precede bash on all new terminator shells with vglrun so that 3d graphics apps will use the GPU
RUN perl -pi -e 's/^Exec=terminator$/Exec=terminator -e "vglrun bash"/g' /usr/share/applications/terminator.desktop

# Start setups for TurboVNC
RUN mkdir -p /root/.vnc
COPY ./requirements/xstartup.turbovnc /root/.vnc/xstartup.turbovnc
RUN chmod a+x /root/.vnc/xstartup.turbovnc

# Install Sound ASLA and Pulse and setup sound
RUN DEBIAN_FRONTEND=noninteractive apt-get -qq update && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y pulseaudio pavucontrol && DEBIAN_FRONTEND=noninteractive apt update && apt-get install -y openjdk-8-jdk python3  python3-pip unzip && pip3 install numpy gym py4j port_for opencv-python && pip3 install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html && pip3 install tensorboard && wget http://www.ice.ci.ritsumei.ac.jp/~ftgaic/Downloadfiles/FTG4.50.zip && git clone -b ACER https://github.com/Byron-Edwards/FTG4.50.git  && unzip FTG4.50.zip && cd FTG4.50 && git clone https://github.com/Byron-Edwards/Gym-FightingICE.git && cd Gym-FightingICE && pip3 install -e .



#---------------------------------------------------------------------
# Startup
#---------------------------------------------------------------------
COPY requirements/launch.sh /opt/noVNC/utils/launch.sh
COPY requirements/start_desktop.sh /usr/local/bin/start_desktop.sh

# Uncomment for autostart of the VNC server
#CMD /usr/local/bin/start_desktop.sh
#  pulseaudio -D --exit-idle-time=-1 && pactl load-module module-null-sink sink_name=SpeakerOutput sink_properties=device.description="Dummy_Output" 

CMD /bin/bash
