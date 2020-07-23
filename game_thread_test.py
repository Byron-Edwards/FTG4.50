import os
import subprocess
from subprocess import DEVNULL
import tempfile
import time
temp =tempfile.TemporaryFile()
p = []
for i in range(4):
    java_env = subprocess.Popen(['java', '-Dsun.reflect.inflationThreshold=2147483647', '-Xmx2g', '-Xms1g',
                                 '-cp', 'FightingICE.jar:./lib/lwjgl/*:./lib/natives/linux/*:./lib/*',
                                 'Main', '--port', '600{}'.format(i),
                                 '--fastmode', '-r', '1000',
                                 '--inverted-player', '1', '--mute',
                                 '--limithp', '400', '400', '--grey-bg',
                                 '--a1', 'ReiwaThunder', '--a2', 'Toothless', '--c1', 'ZEN', '--c2', 'ZEN'],
                                stdout=temp, stderr=temp)
    p.append(java_env)
    time.sleep(10)


