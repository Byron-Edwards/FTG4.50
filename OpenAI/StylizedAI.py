import random
import python.action as pyactions
from OpenAI.Calculator import Calculator
import logging
# logging.basicConfig(level=logging.DEBUG)
class StylizedAI(object):
    def __init__(self, gateway, frameskip=True, agent_type=0):
        # Agent type: 1 = Aggressive, 2 = mixing, 3 = defensive
        self.gateway = gateway
        self.java_actions = self.gateway.jvm.enumerate.Action
        self.obs = None
        self.just_inited = True
        self._actions = "AIR AIR_A AIR_B AIR_D_DB_BA AIR_D_DB_BB AIR_D_DF_FA AIR_D_DF_FB AIR_DA AIR_DB AIR_F_D_DFA AIR_F_D_DFB AIR_FA AIR_FB AIR_GUARD AIR_GUARD_RECOV AIR_RECOV AIR_UA AIR_UB BACK_JUMP BACK_STEP CHANGE_DOWN CROUCH CROUCH_A CROUCH_B CROUCH_FA CROUCH_FB CROUCH_GUARD CROUCH_GUARD_RECOV CROUCH_RECOV DASH DOWN FOR_JUMP FORWARD_WALK JUMP LANDING NEUTRAL RISE STAND STAND_A STAND_B STAND_D_DB_BA STAND_D_DB_BB STAND_D_DF_FA STAND_D_DF_FB STAND_D_DF_FC STAND_F_D_DFA STAND_F_D_DFB STAND_FA STAND_FB STAND_GUARD STAND_GUARD_RECOV STAND_RECOV THROW_A THROW_B THROW_HIT THROW_SUFFER"
        self._action_air = "AIR_GUARD AIR_A AIR_B AIR_DA AIR_DB AIR_FA AIR_FB AIR_UA AIR_UB AIR_D_DF_FA AIR_D_DF_FB AIR_F_D_DFA AIR_F_D_DFB AIR_D_DB_BA AIR_D_DB_BB"
        self._action_ground = "STAND_D_DB_BA BACK_STEP FORWARD_WALK DASH JUMP FOR_JUMP BACK_JUMP STAND_GUARD CROUCH_GUARD THROW_A THROW_B STAND_A STAND_B CROUCH_A CROUCH_B STAND_FA STAND_FB CROUCH_FA CROUCH_FB STAND_D_DF_FA STAND_D_DF_FB STAND_F_D_DFA STAND_F_D_DFB STAND_D_DB_BB"
        # self._action_attack = "AIR_A AIR_B AIR_D_DB_BA AIR_D_DB_BB AIR_D_DF_FA AIR_D_DF_FB AIR_DA AIR_DB AIR_F_D_DFA AIR_F_D_DFB AIR_FA AIR_FB AIR_UA AIR_UB CROUCH_A CROUCH_B CROUCH_FA CROUCH_FB STAND_A STAND_B STAND_D_DB_BA STAND_D_DB_BB STAND_D_DF_FA STAND_D_DF_FB STAND_D_DF_FC STAND_F_D_DFA STAND_F_D_DFB STAND_FA STAND_FB THROW_A THROW_B"
        self._action_attack = "AIR_A AIR_B  AIR_DA AIR_DB AIR_FA AIR_FB AIR_UA AIR_UB CROUCH_A CROUCH_B CROUCH_FA CROUCH_FB STAND_A STAND_B STAND_D_DB_BA STAND_D_DB_BB STAND_D_DF_FC STAND_F_D_DFA STAND_F_D_DFB STAND_FA STAND_FB THROW_A THROW_B"
        self._action_down_recover = "STAND_GUARD_RECOV CROUCH_GUARD_RECOV AIR_GUARD_RECOV STAND_RECOV CROUCH_RECOV AIR_RECOV CHANGE_DOWN DOWN RISE LANDING THROW_HIT THROW_SUFFER"
        self._action_defence = "AIR_GUARD STAND_GUARD CROUCH_GUARD"
        self._valid_action = self._action_air + " " + self._action_ground
        self.action_strs = self._actions.split(" ")
        self.agent_type_strs = ["Aggressive", "Mixing", "Defensive"] # First create three type, later extend to 5
        self.agent_type = agent_type
        self.frameskip = frameskip

    def agent_type_random(self):
        new_type = random.randint(1,3)
        self.agent_type_swith(new_type)

    def agent_type_swith(self, target_type):
        self.agent_type = target_type

    def close(self):
        pass

    def initialize(self, gameData, player):
        self.inputKey = self.gateway.jvm.struct.Key()
        self.frameData = self.gateway.jvm.struct.FrameData()
        self.cc = self.gateway.jvm.aiinterface.CommandCenter()
        self.player = player
        self.gameData = gameData
        self.simulator = self.gameData.getSimulator()
        self.my_motion = self.gameData.getMotionData(self.player)
        self.opp_motion = self.gameData.getMotionData(not self.player)
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
        if frameData.getFramesNumber() >= 0 :
            self.myLastAction = self.frameData.getCharacter(self.player).getAction()
            self.oppLastAttack = self.opp_motion.get(self.frameData.getCharacter(not self.player).getAction().ordinal())
        if frameData.getFramesNumber() > 14:
            self.frameData = self.simulator.simulate(frameData, self.player, None, None, 14)
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
        # self.calc = Calculator(self.frameData, self.gameData, self.player, Calculator.NONACT)
        self.get_obs()
        action = self.act()
        if str(action) == "CROUCH_GUARD":
            action = "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
        elif str(action) == "STAND_GUARD":
            action = "4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4"
        elif str(action) == "AIR_GUARD":
            action = "7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7"
        self.cc.commandCall(action)
        print("Stylized AI {} perform ACTION: {}".format(self.agent_type, action))


    def get_obs(self):
        self.distance = self.frameData.getDistanceX()
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
        self.myFront = self.my.isFront()
        self.mySpeedX = self.my.getSpeedX()  # 15
        self.mySpeedY = self.my.getSpeedY()  # 28
        self.myAction = self.my.getAction()
        self.myAttack = self.my.getAttack()
        self.myState = self.my.getState()
        self.myRemainingFrame = self.my.getRemainingFrame()  # 70

        # opp information
        self.oppHp = abs(self.opp.getHp())  # 400
        self.oppEnergy = self.opp.getEnergy()  # 300
        self.oppLeft = self.opp.getLeft()  # 960
        self.oppRight = self.opp.getRight()  # 960
        self.oppTop = self.opp.getTop()  # 640
        self.oppBottom = self.opp.getBottom()  # 640
        self.oppFront = self.opp.isFront()
        self.oppSpeedX = self.opp.getSpeedX()  # 15
        self.oppSpeedY = self.opp.getSpeedY()  # 28
        self.oppAction = self.opp.getAction()
        self.oppAttack = self.opp.getAttack()
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

    def act(self):
        # TODO: need to figure out the proper distance to make sure most of the attack could take effect, \
        # later need to use the precise judgement
        if self.is_down() and self.can_hall_wall(250):
            return self.defence()
        elif self.distance < 200:
            return self.battle_policy()
        else:
            return self.moving_policy()

    def moving_policy(self):
        # 1 = moving closer, 2 = neural, 3 = moving far
        # TODO need to add the stage edge judgement

        if 200 <= self.distance <= 400:
            if self.agent_type == 1:
                return self.move_closer()
            elif self.agent_type == 3:
                return self.move_far() if self.at_edge(50) else self.neutral()
            elif self.agent_type == 2:
                if self.at_advantage():
                    return self.neutral()
                else:
                    return self.move_closer()
            else:
                raise Exception("No such agent type")

        elif self.distance >= 400:
            if self.agent_type == 1:
                return self.move_closer()
            elif self.agent_type == 3:
                return self.neutral()
            elif self.agent_type == 2:
                if self.at_advantage():
                    return self.neutral()
                else:
                    return self.move_closer()
            else:
                raise Exception("No such agent type")
        else:
            raise Exception("The distance is illegal")

    def battle_policy(self):
        # when opp attack first
        if str(self.oppAction) in self._action_attack.split(" "):
            if self.agent_type == 1:
                return self.counter_attack()
            elif self.agent_type == 3:
                action = self.dodge()
                return action if action else self.defence()
            elif self.agent_type == 2:
                if self.at_advantage():
                    action = self.dodge()
                    return action if action else self.counter_attack()
                else:
                    return self.counter_attack()
            else:
                return self.neutral()
        # when opp is not attacking
        else:
            if self.agent_type == 1:
                return self.active_attack()
            elif self.agent_type == 3:
                action = self.dodge()
                return action if action else self.active_attack()
            elif self.agent_type == 2:
                if self.at_advantage():
                    action = self.dodge()
                    return action if action else self.active_attack()
                else:
                    return self.active_attack()
            else:
                return self.neutral()

    # the following functions are used to Judge the current situation
    def at_advantage(self):
        return self.myHp > self.oppHp

    def opp_attack_before_active(self, action):
        pass

    # TODO: implement this function later to replace the simple distance judgement in battle
    def is_in_hit_area(self, action):
        pass

    # TODO: implement this function later to replayce the simple energy judgement in battle
    def is_enough_energy(self, action):
        pass

    def is_down(self):
        print("is_down function")
        return str(self.myLastAction) == "DOWN" or str(self.myLastAction) == "RISE" or str(self.myLastAction) == "CHANGE_DOWN"

    def can_hall_wall(self, threshold):
        print("can_hall_wall function")
        if self.oppLeft > self.myLeft and self.oppLeft < threshold: return True
        if self.oppRight < self.myRight and self.gameData.getStageWidth() - self.oppRight < threshold: return True
        return False

    def at_edge(self, threshold):
        if self.myLeft <= threshold or self.gameData.getStageWidth() - self.myRight <= threshold:
            return True
        else:
            return False


    # the following functions are used to perform the actions
    # dodge response when opp perform attack before attack active frame.
    def dodge(self):
        print("dodge function")
        if (self.myLeft >= 150 and self.myFront) or (self.gameData.getStageWidth() - self.myRight >= 150 and not self.myFront):
            if str(self.oppState) == "AIR":
                return "BACK_STEP"
            else:
                return "BACK_JUMP"
        else:
            return None

    # defence response when opp perform attack before attack active frame.
    def defence(self):
        print("defence function")
        if str(self.myState) == "AIR":
            return "AIR_GUARD"
        elif self.oppLastAttack.getAttackType() == 1 or self.oppLastAttack.getAttackType() == 2:
            return "STAND_GUARD"
        elif self.oppLastAttack.getAttackType() == 3:
            return "CROUCH_GUARD"
        else:
            print("Did not catch the condition, return default Defence Action")
            return "CROUCH_GUARD"

    # counter attack behavior when opp is using attack or r the opp attack active frame
    # TODO add later, use active_attack first
    def counter_attack(self):
        print("counter attack function")
        return self.active_attack()

    # Active attack behavior when opp is not using attack or after the opp attack active frame
    def active_attack(self):
        print("active attack function")
        if (str(self.myState) == "STAND" or str(self.myState) == "CROUCH") and (str(self.oppState) == "STAND" or str(self.oppState) == "CROUCH"):
            print("active attack:STAND CROUCH")
            if self.oppLastAttack.getAttackType() == 1 or self.oppLastAttack.getAttackType() == 2:
                if self.myEnergy >= 50:
                    return "STAND_D_DB_BB"
                else:
                    return "CROUCH_FB"
            elif self.oppLastAttack.getAttackType() == 3:
                return "JUMP"
            if self.distance >= 175:
                return "STAND_FB"
            elif self.myEnergy >= 50:
                    return "STAND_D_DB_BB"
            elif str(self.oppAction) != "CROUCH_GUARD":
                    return "CROUCH_FB"
            else:
                return random.choice(["STAND_A", "STAND_B", "STAND_FA", "STAND_FB","CROUCH_A","CROUCH_B","CROUCH_FA"])

        elif (str(self.myState) == "STAND" or str(self.myState) == "CROUCH") and (str(self.oppState) == "DOWN"):
            print("active attack:STAND DOWN")
            if self.myEnergy >= 200:
                return "STAND_D_DF_FC"
            else:
                return "CROUCH_FB"

        elif (str(self.myState) == "STAND" or str(self.myState) == "CROUCH") and str(self.oppState) == "AIR":
            print("active attack:STAND AIR")
            if (int(self.oppFront) * self.oppSpeedX) > 0:
                if self.distance >= 175:
                    return "STAND_FB"
                elif self.myEnergy >= 100:
                    # L Dragon!
                    return "STAND_F_D_DFB"
                else:
                    return random.choice(["STAND_F_D_DFA", "CROUCH_FA"])
            else:
                return self.neutral()

        elif str(self.myState) == "AIR" and (str(self.oppState) == "STAND" or str(self.oppState) == "CROUCH"):
            print("active attack:AIR STAND")

            return random.choice(["AIR_DA", "AIR_DB"])

        elif str(self.myState) == "AIR" and str(self.oppState) == "AIR":
            print("active attack:AIR AIR")
            if (self.myTop+self.myBottom)/2 < (self.oppTop + self.oppBottom) / 2:
                return random.choice(["AIR_UB", "AIR_UA"])
            elif (self.myTop + self.myBottom) / 2 > (self.oppTop + self.oppBottom) / 2:
                return random.choice(["AIR_DB", "AIR_DA"])
            elif (self.myTop + self.myBottom) / 2 == (self.oppTop + self.oppBottom) / 2:
                return random.choice(["AIR_B", "AIR_A", "AIR_FA", "AIR_FB"])
            else:
                return self.neutral()
        else:
            print("Can not process by rule, return neural")
            return self.neutral()



    # Distance control by moving actions
    def move_closer(self):
        print("Moving closer")
        return random.choice(["FORWARD_WALK", "DASH"])

    def move_far(self):
        print("Moving far")
        return random.choice(["BACK_JUMP", "BACK_STEP"])

    def neutral(self):
        print("neutral function")
        return "NEUTRAL"

    def get_agent_type(self):
        print(self.agent_type_strs[self.agent_type - 1])
        return self.agent_type

    def isEnoughEnergy(self, act, player):
        ch = self.frameData.getCharacter(player)
        motion = self.my_motion if player else self.opp_motion
        return motion.get(pyactions.Action.to_ordinal(act)).getAttackStartAddEnergy() + ch.getEnergy() >= 0

    def getEnoughEnergyActions(self, player, acts):
        moveActs = list()
        for tac in acts:
            if self.isEnoughEnergy(tac, player):
                moveActs.append(tac)
        return moveActs

    def IsInHitArea(self, ac):
        mych = self.frameData.getCharacter(self.player)
        opch = self.frameData.getCharacter(not self.player)
        if not self.isEnoughEnergy(ac, self.player):
            return False
        mo = self.my_motion.get(pyactions.Action.to_ordinal(ac))
        hi = mo.getAttackHitArea()

        top = mych.getY() + hi.getTop()
        bottom = mych.getY() + hi.getBottom()
        bottom += mo.getAttackStartUp() * mo.getSpeedY()
        top += mo.getAttackStartUp() * mo.getSpeedY()

        if mo.getAttackSpeedY() > 0:
            bottom += mo.getAttackActive() * mo.getAttackSpeedY()
        else:
            top += mo.getAttackActive() * mo.getAttackSpeedY()
        if mo.getSpeedY() > 0:
            bottom += mo.getAttackActive() * mo.getSpeedY()
        else:
            top += mo.getAttackActive() * mo.getSpeedY()

        frontfugou = 1
        if mych.isFront():
            left = mych.getX() + hi.getLeft()
            right = mych.getX() + hi.getRight()
        else:
            frontfugou = -1
            left = mych.getX() + mych.getGraphicSizeX() - hi.getRight()
            right = mych.getX() + mych.getGraphicSizeX() - hi.getLeft()

        left += mo.getAttackStartUp() * mo.getSpeedX() * frontfugou
        right += mo.getAttackStartUp() * mo.getSpeedX() * frontfugou

        if mo.getAttackSpeedX() * frontfugou > 0:
            right += mo.getAttackActive() * mo.getAttackSpeedX() * frontfugou
        else:
            left += mo.getAttackActive() * mo.getAttackSpeedX() * frontfugou

        if mo.getSpeedX() * frontfugou > 0:
            right += mo.getAttackActive() * mo.getSpeedX() * frontfugou
        else:
            left += mo.getAttackActive() * mo.getSpeedX() * frontfugou

        oright = opch.getRight()
        oleft = opch.getLeft()
        otop = opch.getTop()
        obottom = opch.getBottom()

        if right < oleft:
            return False
        if oright < left:
            return False
        if bottom < otop:
            return False
        if obottom < top:
            return False
        return True

    def get_hittable_actions(self, actions, sort_by="energy"):
        actions = self.getEnoughEnergyActions(self.player, actions)
        actions = [action for action in actions if self.IsInHitArea(action)]
        # if sort_by == "energy":
        #     actions = sorted(actions, key=lambda action: self.my_motion.get(pyactions.Action.to_ordinal(action)).getAttackStartAddEnergy())
        # else:
        #     actions = sorted(actions, key=lambda action: self.my_motion.get(pyactions.Action.to_ordinal(action)).getAttackHitDamage(),reverse=True)
        return actions[0] if actions else self.neutral()

    # python version of ReiwaThunder ZEN Standard,
    def act2(self):
        # TODO: need to test
        # later need to use the precise judgement
        if self.is_down() and self.can_hall_wall(250):
            return self.defence()
        if str(self.myState != "AIR"):
            moveActs=self.calc.getEnoughEnergyActions(False,"NEUTRAL")
            if self.calc.canHitFromNow("STAND_D_DF_FC", self.player):
                if 100 <= self.distance < 230:
                    return "STAND_D_DF_FC"
                if self.calc.getMinHpScore("STAND_B", moveActs) > 0:
                    return "STAND_B"
                if self.calc.canHameWall(self.player, 50) and self.calc.getMinHpScore("CROUCH_FB", moveActs) > 0:
                    return "CROUCH_FB"

        if self.distance<300:
            movingActions = self.calc.getEnoughEnergyActions(self.player,["FOR_JUMP","FORWARD_WALK","NEUTRAL","JUMP"])
        else:
            movingActions = self.calc.getEnoughEnergyActions(self.player,["FORWARD_WALK", "FOR_JUMP", "NEUTRAL", "JUMP"])
        if self.calc.canHameWall(not self.player,50):
            nigeActions = self.calc.getEnoughEnergyActions(self.player,
                                                             ["NEUTRAL", "FOR_JUMP", "FORWARD_WALK", "NEUTRAL","BACK_STEP","JUMP","BACK_JUMP"])
        elif self.distance<300:
            nigeActions = self.calc.getEnoughEnergyActions(self.player,
                                                           ["BACK_STEP", "NEUTRAL", "JUMP", "FOR_JUMP","FORWARD_WALK", "BACK_JUMP"])
        elif self.distance<600:
            nigeActions = self.calc.getEnoughEnergyActions(self.player,
                                                           ["NEUTRAL", "BACK_STEP", "JUMP", "FOR_JUMP","FORWARD_WALK", "BACK_JUMP"])
        else:
            nigeActions = self.calc.getEnoughEnergyActions(self.player,
                                                           ["FORWARD_WALK", "NEUTRAL", "BACK_STEP", "FOR_JUMP", "JUMP","BACK_JUMP"])

        HittingMyActions = []
        bestscore = -9999;
        bestac = None
        myacs = self._action_air.split() if str(self.myState)=="AIR" else self._action_ground.split()
        for ac in myacs:
            hpscore=self.calc.getHpScore(ac)
            mo=self.my_motion.get(pyactions.Action.to_ordinal(ac))
            if (hpscore > 0):
                HittingMyActions.append(ac)
                turnscore= hpscore + 30 if mo.isAttackDownProp() else 0 - mo.getAttackStartUp() * 0.01 - mo. getCancelAbleFrame() * 0.0001
                if turnscore > bestscore:
                    bestscore=turnscore
                    bestac=ac

        HittingOpActions = []
        opacs = self._action_air.split() if str(self.oppState) == "AIR" else self._action_ground.split()
        for ac in opacs:
            hpscore=self.calc.getHpScore("NEUTRAL", ac)
            mo=self.my_motion.get(pyactions.Action.to_ordinal(ac))
            if (hpscore < 0):
                HittingOpActions.append(ac)
        HittingOpActions.append("NEUTRAL")

        if not HittingMyActions:
            return self.calc.getMinMaxHp(HittingMyActions, HittingOpActions)
        elif not self.at_advantage():
            return self.calc.getMinMaxHp(movingActions, HittingOpActions)
        else:
            return self.calc.getMinMaxHp(nigeActions, HittingOpActions)

    def getJavaAction(self, action):
        for Jaction in self.java_actions:
            if str(action) == Jaction.name():
                return Jaction

    # Active attack behavior when opp is not using attack or after the opp attack active frame
    # TODO need to test
    def active_attack2(self):
        print("active attack function")
        if (str(self.myState) == "STAND" or str(self.myState) == "CROUCH") and (str(self.oppState) == "STAND" or str(self.oppState) == "CROUCH"):
            print("active attack:STAND CROUCH")

            if self.oppLastAttack.getAttackType() == 1 or self.oppLastAttack.getAttackType() == 2:
                actions = ["STAND_D_DB_BB", "CROUCH_FB", ]
            elif self.oppLastAttack.getAttackType() == 3:
                actions = ["JUMP"]
            else:
                actions = ["STAND_FB","STAND_D_DB_BB","CROUCH_FB","STAND_A", "STAND_B", "STAND_FA", "STAND_FB","CROUCH_A","CROUCH_B","CROUCH_FA"]
            return self.get_hittable_actions(actions)

        elif (str(self.myState) == "STAND" or str(self.myState) == "CROUCH") and (str(self.oppState) == "DOWN"):
            print("active attack:STAND DOWN")
            actions =["STAND_D_DF_FC","CROUCH_FB"]
            return self.get_hittable_actions(actions)

        elif (str(self.myState) == "STAND" or str(self.myState) == "CROUCH") and str(self.oppState) == "AIR":
            print("active attack:STAND AIR")
            actions = ["STAND_FB","STAND_F_D_DFB","STAND_F_D_DFA", "CROUCH_FA"]
            return self.get_hittable_actions(actions)

        elif str(self.myState) == "AIR" and (str(self.oppState) == "STAND" or str(self.oppState) == "CROUCH"):
            print("active attack:AIR STAND")
            actions = ["AIR_DA", "AIR_DB"]
            return self.get_hittable_actions(actions)

        elif str(self.myState) == "AIR" and str(self.oppState) == "AIR":
            print("active attack:AIR AIR")
            actions = ["AIR_FB", "AIR_FA","AIR_UB", "AIR_UA", "AIR_DB", "AIR_DA", "AIR_B", "AIR_A", ]
            return self.get_hittable_actions(actions)
        else:
            print("Can not be processed by any rule, return neural")
            return self.neutral()


    # This part is mandatory
    class Java:
        implements = ["aiinterface.AIInterface"]
