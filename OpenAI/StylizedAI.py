import numpy as np
import pickle
import random
import os
from collections import OrderedDict


class StylizedAI(object):
    def __init__(self, gateway, frameskip=True):
        self.gateway = gateway
        self.obs = None
        self.just_inited = True
        self._actions = "AIR AIR_A AIR_B AIR_D_DB_BA AIR_D_DB_BB AIR_D_DF_FA AIR_D_DF_FB AIR_DA AIR_DB AIR_F_D_DFA AIR_F_D_DFB AIR_FA AIR_FB AIR_GUARD AIR_GUARD_RECOV AIR_RECOV AIR_UA AIR_UB BACK_JUMP BACK_STEP CHANGE_DOWN CROUCH CROUCH_A CROUCH_B CROUCH_FA CROUCH_FB CROUCH_GUARD CROUCH_GUARD_RECOV CROUCH_RECOV DASH DOWN FOR_JUMP FORWARD_WALK JUMP LANDING NEUTRAL RISE STAND STAND_A STAND_B STAND_D_DB_BA STAND_D_DB_BB STAND_D_DF_FA STAND_D_DF_FB STAND_D_DF_FC STAND_F_D_DFA STAND_F_D_DFB STAND_FA STAND_FB STAND_GUARD STAND_GUARD_RECOV STAND_RECOV THROW_A THROW_B THROW_HIT THROW_SUFFER"
        self._action_air = "AIR_GUARD AIR_A AIR_B AIR_DA AIR_DB AIR_FA AIR_FB AIR_UA AIR_UB AIR_D_DF_FA AIR_D_DF_FB AIR_F_D_DFA AIR_F_D_DFB AIR_D_DB_BA AIR_D_DB_BB"
        self._action_ground = "STAND_D_DB_BA BACK_STEP FORWARD_WALK DASH JUMP FOR_JUMP  BACK_JUMP STAND_GUARD CROUCH_GUARD THROW_A THROW_B STAND_A STAND_B CROUCH_A CROUCH_B STAND_FA STAND_FB CROUCH_FA CROUCH_FB STAND_D_DF_FA STAND_D_DF_FB STAND_F_D_DFA STAND_F_D_DFB STAND_D_DB_BB"
        self._action_guard = "AIR_GUARD STAND_GUARD CROUCH_GUARD"
        self.action_strs = self._actions.split(" ")
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
            print("Lost, p1hp:{}, p2hp:{}, frame used: {}".format(p1hp, p2hp, frames))
        elif p1hp > p2hp:
            print("Win!, p1hp:{}, p2hp:{}, frame used: {}".format(p1hp, p2hp, frames))
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

    def ImDown(self):
        if self.myAction.equals(self.gateway.jvm.enumerate.Action.DOWN) \
                or self.myAction.equals(self.gateway.jvm.enumerate.Action.RISE) \
                or self.myAction.equals(self.gateway.jvm.enumerate.Action.CHANGE_DOWN):
            return True
        else:
            return False

    def canHallWall(self, player_number, threshold):
        (my, opp) = (self.my, self.opp) if player_number else (my, opp) = (self.opp, self.my)
        if my.getRight() < opp.getRight() and self.gameData.getStageWidth() - opp.getRight() < threshold: return True
        if opp.getLeft() < my.getLeft() and opp.getLeft() < threshold: return True
        return False

    def defence(self):
        if self.myState.equals(self.gateway.jvm.enumerate.State.AIR):
            return "AIR_GUARD"
        if self.oppAction is not None and (self.oppAction.getAttackType() == 1 or self.oppAction.getAttackType() == 2 ):
            return "STAND_GUARD"
        elif self.oppAction is not None and (self.oppAction.getAttackType() == 3):
            return "CROUCH_GUARD"

    def dodge(self):


    def attack(self):
        pass


    def MoveClose(self):
            return random.choice(["FOR_JUMP","FORWARD_WALK","DASH"])

    def MoveFar(self):
            return random.choice(["BACK_JUMP","BACK_STEP"])

    def NonAction(self):
        return "NEUTRAL"





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

        distance = self.frameData.getDistanceX()
        self.my = self.frameData.getCharacter(self.player)
        self.opp = self.frameData.getCharacter(not self.player)
        self.myProjectiles = self.frameData.getProjectilesByP1() if self.player else self.frameData.getProjectilesByP2()
        self.oppProjectiles = self.frameData.getProjectilesByP2() if not self.player else self.frameData.getProjectilesByP1()

        # my information
        self.myHp = abs(self.my.getHp())  # 400
        self.myEnergy = self.my.getEnergy()  # 300
        self.myLeft = self.my.getLeft()
        self.myRight = self.my.getRight()  # 960
        self.myTop = self.my.getTop()  # 640
        self.myBottom = self.my.getBottom()
        self.mySpeedX = self.my.getSpeedX()  # 15
        self.mySpeedY = self.my.getSpeedY()  # 28
        self.myAction = self.my.getAction()
        self.myState = self.my.getState()
        self.myRemainingFrame = self.my.getRemainingFrame()  # 70

        # opp information
        self.oppHp = abs(self.opp.getHp())  # 400
        self.oppEnergy = self.opp.getEnergy()  # 300
        self.oppLeft = self.opp.getLeft()  # 960
        self.oppRight = self.opp.getRight()  # 960
        self.oppTop = self.opp.getTop()  # 640
        self.oppBottom = self.opp.getBottom()  # 640
        self.oppSpeedX = self.opp.getSpeedX()  # 15
        self.oppSpeedY = self.opp.getSpeedY()  # 28
        self.oppAction = self.opp.getAction()
        self.oppState = self.opp.getState()
        self.oppRemainingFrame = self.opp.getRemainingFrame()  # 70

        # time information
        self.game_frame_num = self.frameData.getFramesNumber()  # 3600

        self.myHitDamage0 = self.myProjectiles[0].getHitDamage() if len(self.myProjectiles) >= 1 else None  # 200.0
        self.myHitAreaNowLeft0 = self.myProjectiles[0].getCurrentHitArea().getLeft() if len(self.myProjectiles) >= 1 else None  # 960.0
        self.myHitAreaNowRight0 = self.myProjectiles[0].getCurrentHitArea().getRight() if len(self.myProjectiles) >= 1 else None  # 960.0
        self.myHitAreaNowTop0 = self.myProjectiles[0].getCurrentHitArea().getTop() if len(self.myProjectiles) >= 1 else None  # 640.0
        self.myHitAreaNowBottom0 = self.myProjectiles[0].getCurrentHitArea().getBottom() if len(self.myProjectiles) >= 1 else None  # 640.0
        self.myHitDamage1 = self.myProjectiles[1].getHitDamage() if len(self.myProjectiles) >= 2 else None  # 200.0
        self.myHitAreaNowLeft1 = self.myProjectiles[1].getCurrentHitArea().getLeft() if len(self.myProjectiles) >= 2 else None  # 960.0
        self.myHitAreaNowRight1 = self.myProjectiles[1].getCurrentHitArea().getRight() if len(self.myProjectiles) >= 2 else None  # 960.0
        self.myHitAreaNowTop1 = self.myProjectiles[1].getCurrentHitArea().getTop() if len(self.myProjectiles) >= 2 else None  # 640.0
        self.myHitAreaNowBottom1 = self.myProjectiles[1].getCurrentHitArea().getBottom() if len(self.myProjectiles) >= 2 else None  # 640.0

        self.oppHitDamage0 = self.oppProjectiles[0].getHitDamage() if len(self.oppProjectiles) >= 1 else None  # 200.0
        self.oppHitAreaNowLeft0 = self.oppProjectiles[0].getCurrentHitArea().getLeft() if len(self.oppProjectiles) >= 1 else None  # 960.0
        self.oppHitAreaNowRight0 = self.oppProjectiles[0].getCurrentHitArea().getRight() if len(self.oppProjectiles) >= 1 else None  # 960.0
        self.oppHitAreaNowTop0 = self.oppProjectiles[0].getCurrentHitArea().getTop() if len(self.oppProjectiles) >= 1 else None  # 640.0
        self.oppHitAreaNowBottom0 = self.oppProjectiles[0].getCurrentHitArea().getBottom() if len(self.oppProjectiles) >= 1 else None  # 640.0
        self.oppHitDamage1 = self.oppProjectiles[1].getHitDamage() if len(self.oppProjectiles) >= 2 else None  # 200.0
        self.oppHitAreaNowLeft1 = self.oppProjectiles[1].getCurrentHitArea().getLeft() if len(self.oppProjectiles) >= 2 else None  # 960.0
        self.oppHitAreaNowRight1 = self.oppProjectiles[1].getCurrentHitArea().getRight() if len(self.oppProjectiles) >= 2 else None  # 960.0
        self.oppHitAreaNowTop1 = self.oppProjectiles[1].getCurrentHitArea().getTop() if len(self.oppProjectiles) >= 2 else None  # 640.0
        self.oppHitAreaNowBottom1 = self.oppProjectiles[1].getCurrentHitArea().getBottom() if len(self.oppProjectiles) >= 2 else None  # 640.0


        # Following is the brain of the reflex agent. It determines distance to the enemy and the energy of our agent and then it performs an action
        if self.ImDown() and self.canHallWall(not self.player,50):


        if (opp.getEnergy() >= 300) and (my.getHp() - opp.getHp() <= 300):
            # If the opp has 300 of energy, it is dangerous, so better jump!!
            # If the health difference is high we are dominating so we are fearless :)
            self.cc.commandCall("FOR_JUMP _B B B")
        elif not my_state.equals(self.gateway.jvm.enumerate.State.AIR) and not my_state.equals(
                self.gateway.jvm.enumerate.State.DOWN):
            # If not in air
            if distance > 150:
                # If its too far, then jump to get closer fast
                self.cc.commandCall("FOR_JUMP")
            elif energy >= 300:
                # High energy projectile
                self.cc.commandCall("STAND_D_DF_FC")
            elif (distance > 100) and (energy >= 50):
                # Perform a slide kick
                self.cc.commandCall("STAND_D_DB_BB")
            elif opp_state.equals(self.gateway.jvm.enumerate.State.AIR):  # If enemy on Air
                # Perform a big punch
                self.cc.commandCall("STAND_F_D_DFA")
            elif distance > 100:
                # Perform a quick dash to get closer
                self.cc.commandCall("6 6 6")
            else:
                # Perform a kick in all other cases, introduces randomness
                self.cc.commandCall("B")
        elif ((distance <= 150) and (my_state.equals(self.gateway.jvm.enumerate.State.AIR) or my_state.equals(
                self.gateway.jvm.enumerate.State.DOWN)) and (
                      ((self.gameData.getStageWidth() - my_x) >= 200) or (xDifference > 0)) and (
                      (my_x >= 200) or xDifference < 0)):
            # Conditions to handle game corners
            if energy >= 5:
                # Perform air down kick when in air
                self.cc.commandCall("AIR_DB")
            else:
                # Perform a kick in all other cases, introduces randomness
                self.cc.commandCall("B")
        else:
            # Perform a kick in all other cases, introduces randomness
            self.cc.commandCall("B")

    # This part is mandatory
    class Java:
        implements = ["aiinterface.AIInterface"]
