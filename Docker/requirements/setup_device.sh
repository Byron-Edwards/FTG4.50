#/bin/bash
cd /dev
MAKEDEV input
pulseaudio -D --exit-idle-time=-1
pactl load-module module-null-sink sink_name=SpeakerOutput sink_properties=device.description="Dummy_Output"

