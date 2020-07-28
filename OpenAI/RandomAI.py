import numpy as np
import pickle
import random
import os
from collections import OrderedDict

class RandomAI(object):
    def __init__(self, gateway, frameskip=True):
        self.gateway = gateway
        self.obs = None
        self.just_inited = True
        self._actions = "AIR AIR_A AIR_B AIR_D_DB_BA AIR_D_DB_BB AIR_D_DF_FA AIR_D_DF_FB AIR_DA AIR_DB AIR_F_D_DFA AIR_F_D_DFB AIR_FA AIR_FB AIR_GUARD AIR_GUARD_RECOV AIR_RECOV AIR_UA AIR_UB BACK_JUMP BACK_STEP CHANGE_DOWN CROUCH CROUCH_A CROUCH_B CROUCH_FA CROUCH_FB CROUCH_GUARD CROUCH_GUARD_RECOV CROUCH_RECOV DASH DOWN FOR_JUMP FORWARD_WALK JUMP LANDING NEUTRAL RISE STAND STAND_A STAND_B STAND_D_DB_BA STAND_D_DB_BB STAND_D_DF_FA STAND_D_DF_FB STAND_D_DF_FC STAND_F_D_DFA STAND_F_D_DFB STAND_FA STAND_FB STAND_GUARD STAND_GUARD_RECOV STAND_RECOV THROW_A THROW_B THROW_HIT THROW_SUFFER"
        self._action_air = "AIR_GUARD AIR_A AIR_B AIR_DA AIR_DB AIR_FA AIR_FB AIR_UA AIR_UB AIR_D_DF_FA AIR_D_DF_FB AIR_F_D_DFA AIR_F_D_DFB AIR_D_DB_BA AIR_D_DB_BB"
        self._action_ground = "STAND_D_DB_BA BACK_STEP FORWARD_WALK DASH JUMP FOR_JUMP  BACK_JUMP STAND_GUARD CROUCH_GUARD THROW_A THROW_B STAND_A STAND_B CROUCH_A CROUCH_B STAND_FA STAND_FB CROUCH_FA CROUCH_FB STAND_D_DF_FA STAND_D_DF_FB STAND_F_D_DFA STAND_F_D_DFB STAND_D_DB_BB"
        self._valid_action = self._action_air + " " + self._action_ground
        self.action_strs = self._actions.split(" ")
        self.valid_action_strs = self._valid_action.split(" ")
        self.frameskip = frameskip

    def close(self):
        pass

    def initialize(self, gameData, player):
        self.inputKey = self.gateway.jvm.struct.Key()
        self.frameData = self.gateway.jvm.struct.FrameData()
        self.cc = self.gateway.jvm.aiinterface.CommandCenter()
        self.player = player
        self.gameData = gameData
        self.isGameJustStarted = True
        return 0

    # please define this method when you use FightingICE version 3.20 or later
    def roundEnd(self, p1hp, p2hp, frames):
        self.just_inited = True
        if p1hp <= p2hp:
            print("Lost, p1hp:{}, p2hp:{}, frame used: {}".format(p1hp,  p2hp, frames))
        elif p1hp > p2hp:
            print("Win!, p1hp:{}, p2hp:{}, frame used: {}".format(p1hp,  p2hp, frames))
        # self.obs = None

    # Please define this method when you use FightingICE version 4.00 or later
    def getScreenData(self, sd):
        self.screenData = sd

    def getInformation(self, frameData, isControl):
        self.frameData = frameData
        self.isControl = isControl
        self.cc.setFrameData(self.frameData, self.player)
        if frameData.getEmptyFlag():
            return

    def input(self):
        return self.inputKey

    def gameEnd(self):
        pass

    def processing(self):
        if self.frameData.getEmptyFlag() or self.frameData.getRemainingTime() <= 0:
            self.isGameJustStarted = True
            return

        if self.frameskip:
            if self.cc.getSkillFlag():
                self.inputKey = self.cc.getSkillKey()
                return
            if not self.isControl:
                return

            self.inputKey.empty()
            self.cc.skillCancel()

        # Perform random valid action
        action = random.choice(self.valid_action_strs)
        self.cc.commandCall(action)
        if not self.frameskip:
            self.inputKey = self.cc.getSkillKey()

    # This part is mandatory
    class Java:
        implements = ["aiinterface.AIInterface"]
