import sys
from time import sleep
from py4j.java_gateway import JavaGateway, GatewayParameters, CallbackServerParameters, get_field
from OpenAI.ByronAI import ByronAI
from OpenAI.StylizedAI import StylizedAI

# java -cp FightingICE.jar:./lib/lwjgl/*:./lib/natives/linux/*:./lib/*  Main  --mute --grey-bg --limithp 400 400 --py4j --fastmode
def check_args(args):
	for i in range(argc):
		if args[i] == "-n" or args[i] == "--n" or args[i] == "--number":
			global GAME_NUM
			GAME_NUM = int(args[i+1])

def start_game():
	manager.registerAI(ByronAI.__class__.__name__, ByronAI(gateway=gateway, parametes="../OpenAI/SAC/sac.pkl"))
	print("Start game")
	
	game = manager.createGame("ZEN", "ZEN", ByronAI.__class__.__name__, "ReiwaThunder", GAME_NUM)
	manager.runGame(game)
	
	print("After game")
	sys.stdout.flush()

def close_gateway():
	gateway.close_callback_server()
	gateway.close()
	
def main_process():
	check_args(args)
	start_game()
	close_gateway()

args = sys.argv
argc = len(args)
GAME_NUM = 2
gateway = JavaGateway(gateway_parameters=GatewayParameters(port=4242), callback_server_parameters=CallbackServerParameters());
manager = gateway.entry_point

main_process()

