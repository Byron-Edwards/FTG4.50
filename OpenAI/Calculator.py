# This class is the Python version of Calculator from ReiwaThunder
import python.action as pyactions


class Calculator:
    SIMULATE_LIMIT = 80
    NONACT = pyactions.Action.NEUTRAL
    def __init__(self, motoFrame, gd, player, preAct):
        self.motoFrame = motoFrame
        self.gd = gd
        self.simlator = gd.getSimulator()
        self.player = player
        self.myMotion = gd.getMotionData(self.player)
        self.oppMotion = gd.getMotionData(not self.player)
        self.hadoukenActsAir = list()
        self.hadoukenActsGround = list()
        self.damageActs = list()
        for ac in pyactions.AIR_ACTIONS:
            mo = self.myMotion.get(pyactions.Action.to_ordinal(ac))
            if mo.getAttackSpeedX() != 0 or mo.getAttackSpeedY() != 0:
                self.hadoukenActsAir.append(ac)
            if mo.getAttackHitDamage() > 0:
                self.damageActs.append(ac)
        for ac in pyactions.GROUND_ACTIONS:
            mo = self.myMotion.get(pyactions.Action.to_ordinal(ac))
            if mo.getAttackSpeedX() != 0 or mo.getAttackSpeedY() != 0:
                self.hadoukenActsGround.append(ac)
            if mo.getAttackHitDamage() > 0:
                self.damageActs.append(ac)
        self.map = dict()
        self.nonActionFrame = self.getFrame(preAct, self.NONACT)

    def canHameWall(self, playerNumber, threthold):
        my = self.motoFrame.getCharacter(playerNumber)
        op = self.motoFrame.getCharacter(not playerNumber)
        if my.getRight() < op.getRight() and self.gd.getStageWidth() - op.getRight() < threthold: return True;
        if op.getLeft() < my.getLeft() and op.getLeft() < threthold: return True
        return False

    def getHPfromNow(self, myact, opact, playerNumber):
        frame = self.getFrame(myact, opact)
        bef = self.motoFrame.getCharacter(playerNumber)
        aft = frame.getCharacter(playerNumber)
        return aft.getHp() - bef.getHp()

    def canHitFromNow(self, act, playerNumber):
        myact = act
        opact = self.NONACT
        nonHp = self.getHPfromNow(self.NONACT, self.NONACT, not playerNumber)
        aftHp = self.getHPfromNow(myact, opact, not playerNumber)
        return (aftHp < nonHp)

    def isEnoughEnergy(self, act, player):
        mos = self.myMotion if player else self.oppMotion
        ch = self.motoFrame.getCharacter(self.player) if player else self.motoFrame.getCharacter(not self.player)
        return mos.get(pyactions.Action.to_ordinal(act)).getAttackStartAddEnergy() + ch.getEnergy() >= 0

    def getEnoughEnergyActions(self, player, acts):
        moveActs = list()
        for tac in acts:
            if self.isEnoughEnergy(tac, player):
                moveActs.append(tac)
        return moveActs

    def getFrame(self, myact, opact):
        if self.isEnoughEnergy(myact, True):
            tmyact = myact
        else:
            tmyact = self.NONACT
        if self.isEnoughEnergy(opact, False):
            topact = opact
        else:
            topact = self.NONACT
        key = (tmyact, topact)

        if key not in self.map:
            mAction = list()
            mAction.append(tmyact)
            opAction = list()
            opAction.append(topact)
            value = self.simlator.simulate(self.motoFrame, self.player, mAction, opAction, self.SIMULATE_LIMIT)
            self.map[key] = value
        return self.map[key]

    def getMyFrame(self, myact):
        return self.getFrame(myact, self.NONACT)

    def getHpScore(self, myact, opact=pyactions.Action.NEUTRAL):
        fd = self.getFrame(myact, opact)
        gapMyHp = fd.getCharacter(self.player).getHp() - self.nonActionFrame.getCharacter(self.player).getHp()
        gapOpHp = fd.getCharacter(not self.player).getHp() - self.nonActionFrame.getCharacter(not self.player).getHp()
        return gapMyHp - gapOpHp

    def getMinHpScoreIfDamage(self, myact):
        min = 9999
        for opact in self.damageActs:
            score = self.getHpScore(myact, opact)
            if score < min: min = score
        return min

    def getMinMaxIfDamage(self, acs):
        max = -9999
        maxact = pyactions.Action.NEUTRAL
        for myact in acs:
            score = self.getMinHpScoreIfDamage(myact)
            if score > max:
                max = score
                maxact = myact
        return maxact

    def getMinHpScoreIfHadouken(self, myact):
        min = 9999
        for opact in self.hadoukenActsGround:
            score = self.getHpScore(myact, opact)
            if score < min:
                min = score
        return min

    def getMinMaxIfHadouken(self, acs):
        max = -9999
        maxact = pyactions.Action.FORWARD_WALK
        for myact in acs:
            score = self.getMinHpScoreIfHadouken(myact)
            if score > max:
                max = score
                maxact = myact
        return maxact

    def getMinHpScore(self, myact, opAcs):
        min = 9999
        for opact in opAcs:
            score = self.getHpScore(myact, opact)
            if score < min:
                min = score
        return min

    def getMinMaxHp(self, myAcs, opAcs):
        alpha = -9999
        maxact = pyactions.Action.FORWARD_WALK
        for myact in myAcs:
            min = 9999
            for opact in opAcs:
                score = self.getHpScore(myact, opact)
                if score < min:
                    min = score
                    if min < alpha:
                        break
            if min > alpha:
                alpha = min
                maxact = myact
        return maxact

    def IsInHitArea(self, ac):
        mych = self.motoFrame.getCharacter(self.player)
        opch = self.motoFrame.getCharacter(not self.player)
        if not self.isEnoughEnergy(ac, True): return False
        mo = self.myMotion.get(pyactions.Action.to_ordinal(ac))
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

        if right < oleft: return False
        if oright < left: return False
        if bottom < otop: return False
        if obottom < top: return False
        return True
