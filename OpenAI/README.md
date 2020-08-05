#BUTCHER_PUDGE

##INTRODUCTION
ButcherPudge is implemented based on the Deep Reinforcement Learning Algorithm SAC (Soft-Actor-Critic). It was mainly trained to fight with the 2019 Award winners’ AIs  (ReiwaThunder, RHEA_PI and Toothless..) progressively.  This AI was trained directly on the delayed RAM data, and did not utilized the simulator provided by the platform.
 
The Q-network and Policy network both contain 3 layers with 256 hidden units respectively. Reward Tuning was also used during training to help the training converge faster.

##HOW TO RUN

Although this AI was trained with the OpenAI gym interface and pytorch library, it could be run directly as the pure Python AI because the network was deployed using pure python.

Only `numpy` library is needed to support the forward propagation. No deep learning library needed, No Gym needed, No GPU needed. 

Because of the device and time, this AI class was only tested on ubuntu 18.04 

To run this AI, please use the similar method to run the example Python AI `Machete.py` with `Main_PyAIvsPyAI.py` or `Main_PyAIvsJavaAI.py`

My submisssion is the zip file, which is a folder named `OpenAI`. The uncompressed `OpenAI` folder should be directly put under the `FTG4.50` folder. It contains:

```
OpenAI/
├── ButcherPudge/
│   ├── sac_GARNET.pkl
│   ├── sac_LUD.pkl
│   ├── sac_ZEN.pkl
│   ├── sac_GARNET_speed.pkl
│   ├── sac_LUD_speed.pkl
│   └── sac_ZEN_speed.pkl
├── ButcherPudge.py
├── ButcherPudge.pptx
└── README.md
```

ButcherPudge class need the correct **folder** path of the parameter to initialize. For example, If `Main_PyAIvsJavaAI.py` is put under `FTG4.50/python/`, the code in running ButcherPudge:

with relative path is:

```python
from OpenAI.ButcherPudge import ButcherPudge
manager.registerAI(ButcherPudge.__class__.__name__, ButcherPudge(gateway=gateway, para_folder="../OpenAI/ButcherPudge/"))
game = manager.createGame("ZEN", "ZEN", ButcherPudge.__class__.__name__, "p2", GAME_NUM)
```
 
With abs path is:

```python
from OpenAI.ButcherPudge import ButcherPudge
manager.registerAI(ByronAI.__class__.__name__, ByronAI(gateway=gateway, parametes="/home/baiwen/Repos/FTG4.50/OpenAI/ButcherPudge/"))
game = manager.createGame("ZEN", "ZEN", ButcherPudge.__class__.__name__, "p2", GAME_NUM)
```

if the startup script (like `Main_PyAIvsJavaAI.py`) is placed the other place, the folder path need to adjust accordingly

##DEPENDENCY

- numpy
- py4j


##FUTURE WORK
The current version is train with the last years winners, It might overfit the opponent and the character, which means it might master the fight with some specific opponent but not master the fighting itself. Our next work is to make the AI more robust to different style opponents and characters. 

##CONTACTS
Please let me know any questions

personal <byron_edwards@outlook.com>
campus <wbai001@e.ntu.edu.sg>
