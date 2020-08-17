import gym
import gym_fightingice
import signal, functools
from random import randint
from os import listdir
from os.path import isfile, join


class TimeoutError(Exception): pass


def timeout(seconds):
    def decorated(func):
        result = ""

        def _handle_timeout(signum, frame):
            global result
            raise TimeoutError()

        def wrapper(*args, **kwargs):
            global result
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
                return result
            return result

        return functools.wraps(func)(wrapper)

    return decorated


@timeout(10)
def rand_step(env, ainame):
    env.step(randint(0, 55))
    f.write("{} success \n".format(ainame))


@timeout(10)
def env_reset(env,ainame):
    env.reset(p2=ainame)

mypath = 'data/ai'
fileNames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
aiNames = []
for i in fileNames:
    if len(i.split(".")[0])>0:
        aiNames.append(i.split(".")[0])

f = open("test_results.txt", "a")
envs = [gym.make("FightingiceDataNoFrameskip-v0", java_env_path="..", port=4000 + i) for i in range(len(aiNames))]

for env, ainame in zip(envs,aiNames):
    try:
        env_reset(env, ainame)
    except Exception:
        f.write("{} failed \n".format(ainame))
        continue

for env, ainame in zip(envs,aiNames):
    try:
        rand_step(env, ainame)
    except Exception:
        f.write("{} failed \n".format(ainame))
        continue

f.close()
for env in envs:
    env.close()
    del env
