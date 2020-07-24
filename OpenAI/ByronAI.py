import numpy as np
from collections import OrderedDict
np.set_printoptions(precision=32)


def state_dict_trans(state_dict):
    new_dict = OrderedDict()
    for param_tensor in state_dict:
        new_dict[param_tensor] = state_dict[param_tensor].numpy()
    return new_dict


class ActorCritic:

    class Linear:
        def __init__(self, weight, bias):
            self.weight = weight
            self.bias = bias

        def __call__(self, x):
            x = np.dot(self.weight, x)
            x = x + self.bias
            return x

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def softmax(x, dim):
        return np.exp(x) / np.sum(np.exp(x), axis=dim,)

    def __init__(self, state_dict):
        self.state_dict = state_dict
        self.fc1 = self.Linear(self.state_dict["fc1.weight"], self.state_dict["fc1.bias"])
        self.fc_pi = self.Linear(self.state_dict["fc_pi.weight"], self.state_dict["fc_pi.bias"])
        self.fc_v = self.Linear(self.state_dict["fc_v.weight"], self.state_dict["fc_v.bias"])

    def pi(self, x, softmax_dim=0):
        x = self.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = np.clip(a=self.softmax(x, dim=softmax_dim), a_max=1-1e-20, a_min=1e-20)
        return prob

    def v(self, x):
        x = self.relu(self.fc1(x))
        v = self.fc_v(x)
        return v


class ByronAI(object):
    def __init__(self, gateway, pipe, frameskip=True):
        self.gateway = gateway
        self.pipe = pipe

        self.width = 96  # The width of the display to obtain
        self.height = 64  # The height of the display to obtain
        self.grayscale = True  # The display's color to obtain true for grayscale, false for RGB

        self.obs = None
        self.just_inited = True

        self._actions = "AIR AIR_A AIR_B AIR_D_DB_BA AIR_D_DB_BB AIR_D_DF_FA AIR_D_DF_FB AIR_DA AIR_DB AIR_F_D_DFA AIR_F_D_DFB AIR_FA AIR_FB AIR_GUARD AIR_GUARD_RECOV AIR_RECOV AIR_UA AIR_UB BACK_JUMP BACK_STEP CHANGE_DOWN CROUCH CROUCH_A CROUCH_B CROUCH_FA CROUCH_FB CROUCH_GUARD CROUCH_GUARD_RECOV CROUCH_RECOV DASH DOWN FOR_JUMP FORWARD_WALK JUMP LANDING NEUTRAL RISE STAND STAND_A STAND_B STAND_D_DB_BA STAND_D_DB_BB STAND_D_DF_FA STAND_D_DF_FB STAND_D_DF_FC STAND_F_D_DFA STAND_F_D_DFB STAND_FA STAND_FB STAND_GUARD STAND_GUARD_RECOV STAND_RECOV THROW_A THROW_B THROW_HIT THROW_SUFFER"
        self.action_strs = self._actions.split(" ")
        self.frameskip = frameskip

    def close(self):
        pass

    def initialize(self, gameData, player):
        self.inputKey = self.gateway.jvm.struct.Key()
        self.frameData = self.gateway.jvm.struct.FrameData()
        self.cc = self.gateway.jvm.aiinterface.CommandCenter()

        # self.get_obs()

        self.player = player
        self.gameData = gameData

        return 0

    # please define this method when you use FightingICE version 3.20 or later
    def roundEnd(self, p1hp, p2hp, frames):
        print("send round end to {}".format(self.pipe))
        self.pipe.send([self.obs, self.get_reward(), True, {}])
        self.just_inited = True
        if p1hp <= p2hp:
            self.reward -= 1
            print("Lost, p1hp:{}, p2hp:{}, frame used: {}".format(p1hp,  p2hp, frames))
        elif p1hp > p2hp:
            self.reward += 1
            print("Win!, p1hp:{}, p2hp:{}, frame used: {}".format(p1hp,  p2hp, frames))
        # request = self.pipe.recv()
        # if request == "close":
        #     return
        self.obs = None

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

        action = self.get_obs()

        self.cc.commandCall(self.action_strs[action])


    def get_reward(self):
        try:
            if self.pre_framedata.getEmptyFlag() or self.frameData.getEmptyFlag():
                reward = 0
            else:
                p2_hp_pre = self.pre_framedata.getCharacter(False).getHp()
                p1_hp_pre = self.pre_framedata.getCharacter(True).getHp()
                p2_hp_now = self.frameData.getCharacter(False).getHp()
                p1_hp_now = self.frameData.getCharacter(True).getHp()
                if self.player:
                    reward = (p2_hp_pre-p2_hp_now) - (p1_hp_pre-p1_hp_now)
                    # reward shaping
                    if p2_hp_pre-p2_hp_now <= 0 and self.last_action in self._attacks.split():
                        reward -= 0.1
                else:
                    reward = (p1_hp_pre-p1_hp_now) - (p2_hp_pre-p2_hp_now)
        except:
            reward = 0
        return reward

    def get_obs(self):
        my = self.frameData.getCharacter(self.player)
        opp = self.frameData.getCharacter(not self.player)

        # my information
        myHp = abs(my.getHp() / 400)
        myEnergy = my.getEnergy() / 300
        myX = ((my.getLeft() + my.getRight()) / 2) / 960
        myY = ((my.getBottom() + my.getTop()) / 2) / 640
        mySpeedX = my.getSpeedX() / 15
        mySpeedY = my.getSpeedY() / 28
        myState = my.getAction().ordinal()
        myRemainingFrame = my.getRemainingFrame() / 70

        # opp information
        oppHp = abs(opp.getHp() / 400)
        oppEnergy = opp.getEnergy() / 300
        oppX = ((opp.getLeft() + opp.getRight()) / 2) / 960
        oppY = ((opp.getBottom() + opp.getTop()) / 2) / 640
        oppSpeedX = opp.getSpeedX() / 15
        oppSpeedY = opp.getSpeedY() / 28
        oppState = opp.getAction().ordinal()
        oppRemainingFrame = opp.getRemainingFrame() / 70

        # time information
        game_frame_num = self.frameData.getFramesNumber() / 3600

        observation = []

        # my information
        observation.append(myHp)
        observation.append(myEnergy)
        observation.append(myX)
        observation.append(myY)
        if mySpeedX < 0:
            observation.append(0)
        else:
            observation.append(1)
        observation.append(abs(mySpeedX))
        if mySpeedY < 0:
            observation.append(0)
        else:
            observation.append(1)
        observation.append(abs(mySpeedY))
        for i in range(56):
            if i == myState:
                observation.append(1)
            else:
                observation.append(0)
        observation.append(myRemainingFrame)

        # opp information
        observation.append(oppHp)
        observation.append(oppEnergy)
        observation.append(oppX)
        observation.append(oppY)
        if oppSpeedX < 0:
            observation.append(0)
        else:
            observation.append(1)
        observation.append(abs(oppSpeedX))
        if oppSpeedY < 0:
            observation.append(0)
        else:
            observation.append(1)
        observation.append(abs(oppSpeedY))
        for i in range(56):
            if i == oppState:
                observation.append(1)
            else:
                observation.append(0)
        observation.append(oppRemainingFrame)

        # time information
        observation.append(game_frame_num)

        myProjectiles = self.frameData.getProjectilesByP1()
        oppProjectiles = self.frameData.getProjectilesByP2()

        if len(myProjectiles) == 2:
            myHitDamage = myProjectiles[0].getHitDamage() / 200.0
            myHitAreaNowX = ((myProjectiles[0].getCurrentHitArea().getLeft() + myProjectiles[
                0].getCurrentHitArea().getRight()) / 2) / 960.0
            myHitAreaNowY = ((myProjectiles[0].getCurrentHitArea().getTop() + myProjectiles[
                0].getCurrentHitArea().getBottom()) / 2) / 640.0
            observation.append(myHitDamage)
            observation.append(myHitAreaNowX)
            observation.append(myHitAreaNowY)
            myHitDamage = myProjectiles[1].getHitDamage() / 200.0
            myHitAreaNowX = ((myProjectiles[1].getCurrentHitArea().getLeft() + myProjectiles[
                1].getCurrentHitArea().getRight()) / 2) / 960.0
            myHitAreaNowY = ((myProjectiles[1].getCurrentHitArea().getTop() + myProjectiles[
                1].getCurrentHitArea().getBottom()) / 2) / 640.0
            observation.append(myHitDamage)
            observation.append(myHitAreaNowX)
            observation.append(myHitAreaNowY)
        elif len(myProjectiles) == 1:
            myHitDamage = myProjectiles[0].getHitDamage() / 200.0
            myHitAreaNowX = ((myProjectiles[0].getCurrentHitArea().getLeft() + myProjectiles[
                0].getCurrentHitArea().getRight()) / 2) / 960.0
            myHitAreaNowY = ((myProjectiles[0].getCurrentHitArea().getTop() + myProjectiles[
                0].getCurrentHitArea().getBottom()) / 2) / 640.0
            observation.append(myHitDamage)
            observation.append(myHitAreaNowX)
            observation.append(myHitAreaNowY)
            for t in range(3):
                observation.append(0.0)
        else:
            for t in range(6):
                observation.append(0.0)

        if len(oppProjectiles) == 2:
            oppHitDamage = oppProjectiles[0].getHitDamage() / 200.0
            oppHitAreaNowX = ((oppProjectiles[0].getCurrentHitArea().getLeft() + oppProjectiles[
                0].getCurrentHitArea().getRight()) / 2) / 960.0
            oppHitAreaNowY = ((oppProjectiles[0].getCurrentHitArea().getTop() + oppProjectiles[
                0].getCurrentHitArea().getBottom()) / 2) / 640.0
            observation.append(oppHitDamage)
            observation.append(oppHitAreaNowX)
            observation.append(oppHitAreaNowY)
            oppHitDamage = oppProjectiles[1].getHitDamage() / 200.0
            oppHitAreaNowX = ((oppProjectiles[1].getCurrentHitArea().getLeft() + oppProjectiles[
                1].getCurrentHitArea().getRight()) / 2) / 960.0
            oppHitAreaNowY = ((oppProjectiles[1].getCurrentHitArea().getTop() + oppProjectiles[
                1].getCurrentHitArea().getBottom()) / 2) / 640.0
            observation.append(oppHitDamage)
            observation.append(oppHitAreaNowX)
            observation.append(oppHitAreaNowY)
        elif len(oppProjectiles) == 1:
            oppHitDamage = oppProjectiles[0].getHitDamage() / 200.0
            oppHitAreaNowX = ((oppProjectiles[0].getCurrentHitArea().getLeft() + oppProjectiles[
                0].getCurrentHitArea().getRight()) / 2) / 960.0
            oppHitAreaNowY = ((oppProjectiles[0].getCurrentHitArea().getTop() + oppProjectiles[
                0].getCurrentHitArea().getBottom()) / 2) / 640.0
            observation.append(oppHitDamage)
            observation.append(oppHitAreaNowX)
            observation.append(oppHitAreaNowY)
            for t in range(3):
                observation.append(0.0)
        else:
            for t in range(6):
                observation.append(0.0)

        observation = np.array(observation, dtype=np.float32)
        observation = np.clip(observation, 0, 1)
        return observation

    # This part is mandatory
    class Java:
        implements = ["aiinterface.AIInterface"]
