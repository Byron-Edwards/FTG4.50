#BUTCHER_PUDGE

##INTRODUCTION
 


##HOW TO RUN
Although this AI was trained with the OpenAI gym interface and pytorch library, it could be run directly as the pure Python AI because the network was deployed using pure python.

Only `numpy` library is needed to support the forward propagation. No deep learning library needed, No GPU needed. 

To run this AI, please use the similar method to run the example Python AI `Machete.py` with `Main_PyAIvsPyAI.py` or `Main_PyAIvsJavaAI.py`

My submisssion is the zip file, which is a folder named `OpenAI`. It contains:
- AI class
- AI parameters
- This README.md

The uncompressed `OpenAI` folder should be directly put under the `FTG4.50` folder
The AI class need the correct file path of the parameter to initialize. For example:
My `Main_PyAIvsJavaAI.py` is put under `FTG4.50/python/`, 
so the code in running my AI with relative path is:
```python
manager.registerAI(ByronAI.__class__.__name__, ByronAI(gateway=gateway, parametes="../OpenAI/SAC/sac.pkl"))
``` 
With abs path is:
```python
manager.registerAI(ByronAI.__class__.__name__, ByronAI(gateway=gateway, parametes="/home/baiwen/Repos/FTG4.50/OpenAI/SAC/sac.pkl"))
```
if the startup script (like `Main_PyAIvsJavaAI.py`) is placed the other place, the file path need to adjust accordingly

##DEPENDENCY
numpy

##FUTURE WORK
The current version is train with the last years winners, It might overfit the opponent and the character, which means it might master the fight with some specific opponent but not master the fighting itself. Our next work is to make the AI more robust to different style opponents and characters. 

##CONTACTS
Please let me know any questions

personal <byron_edwards@outlook.com>
campus <wbai001@e.ntu.edu.sg>
