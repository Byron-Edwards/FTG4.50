#/bin/bash

# Create X11 Instance

# Start the container
docker run -it \
--name=super-container \
--gpus all  \
--init \
--net=host \
--privileged=true \
--rm=true \
-e DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v /home/byron/Repos/FTG4.50:/workspace      \
recipe-wiz:latest

# Kill X11 Server
#pkill xinit

# For sound?  
#-v /run/user/1000/pulse:/run/user/1000/pulse 
#-e PULSE_SERVER=unix:/run/user/1000/pulse/native 

# Stuff
#-e QT_X11_NO_MITSHM=1   
#-e XAUTHORITY=/tmp/.docker.xauth 
#-v /tmp/.docker.xauth:/tmp/.docker.xauth 


