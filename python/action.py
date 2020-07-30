from enum import Enum

class Action(Enum):
    NEUTRAL = 'NEUTRAL'
    STAND = 'STAND'
    FORWARD_WALK = 'FORWARD_WALK'
    DASH = 'DASH'
    BACK_STEP = 'BACK_STEP'
    CROUCH = 'CROUCH'
    JUMP = 'JUMP'
    FOR_JUMP = 'FOR_JUMP'
    BACK_JUMP = 'BACK_JUMP'
    AIR = 'AIR'
    STAND_GUARD = 'STAND_GUARD'
    CROUCH_GUARD = 'CROUCH_GUARD'
    AIR_GUARD = 'AIR_GUARD'
    STAND_GUARD_RECOV = 'STAND_GUARD_RECOV'
    CROUCH_GUARD_RECOV = 'CROUCH_GUARD_RECOV'
    AIR_GUARD_RECOV = 'AIR_GUARD_RECOV'
    STAND_RECOV = 'STAND_RECOV'
    CROUCH_RECOV = 'CROUCH_RECOV'
    AIR_RECOV = 'AIR_RECOV'
    CHANGE_DOWN = 'CHANGE_DOWN'
    DOWN = 'DOWN'
    RISE = 'RISE'
    LANDING = 'LANDING'

    THROW_A = 'THROW_A'
    THROW_B = 'THROW_B'
    THROW_HIT = 'THROW_HIT'
    THROW_SUFFER = 'THROW_SUFFER'

    STAND_A = 'STAND_A'
    STAND_B = 'STAND_B'
    CROUCH_A = 'CROUCH_A'
    CROUCH_B = 'CROUCH_B'
    AIR_A = 'AIR_A'
    AIR_B = 'AIR_B'
    AIR_DA = 'AIR_DA'
    AIR_DB = 'AIR_DB'
    STAND_FA = 'STAND_FA'
    STAND_FB = 'STAND_FB'
    CROUCH_FA = 'CROUCH_FA'
    CROUCH_FB = 'CROUCH_FB'
    AIR_FA = 'AIR_FA'
    AIR_FB = 'AIR_FB'
    AIR_UA = 'AIR_UA'
    AIR_UB = 'AIR_UB'

    STAND_D_DF_FA = 'STAND_D_DF_FA'
    STAND_D_DF_FB = 'STAND_D_DF_FB'
    STAND_F_D_DFA = 'STAND_F_D_DFA'
    STAND_F_D_DFB = 'STAND_F_D_DFB'
    STAND_D_DB_BA = 'STAND_D_DB_BA'
    STAND_D_DB_BB = 'STAND_D_DB_BB'
    AIR_D_DF_FA = 'AIR_D_DF_FA'
    AIR_D_DF_FB = 'AIR_D_DF_FB'
    AIR_F_D_DFA = 'AIR_F_D_DFA'
    AIR_F_D_DFB = 'AIR_F_D_DFB'
    AIR_D_DB_BA = 'AIR_D_DB_BA'
    AIR_D_DB_BB = 'AIR_D_DB_BB'

    STAND_D_DF_FC = 'STAND_D_DF_FC'

    @staticmethod
    def to_ordinal(action):

        if action == Action.NEUTRAL or action == Action.NEUTRAL.value: return 0
        if action == Action.STAND or action == Action.STAND.value: return 1
        if action == Action.FORWARD_WALK or action == Action.FORWARD_WALK.value: return 2
        if action == Action.DASH or action == Action.DASH.value: return 3
        if action == Action.BACK_STEP or action == Action.BACK_STEP.value: return 4
        if action == Action.CROUCH or action == Action.CROUCH.value: return 5
        if action == Action.JUMP or action == Action.JUMP.value: return 6
        if action == Action.FOR_JUMP or action == Action.FOR_JUMP.value: return 7
        if action == Action.BACK_JUMP or action == Action.BACK_JUMP.value: return 8
        if action == Action.AIR or action == Action.AIR.value: return 9
        if action == Action.STAND_GUARD or action == Action.STAND_GUARD.value: return 10
        if action == Action.CROUCH_GUARD or action == Action.CROUCH_GUARD.value: return 11
        if action == Action.AIR_GUARD or action == Action.AIR_GUARD.value: return 12
        if action == Action.STAND_GUARD_RECOV or action == Action.STAND_GUARD_RECOV.value: return 13
        if action == Action.CROUCH_GUARD_RECOV or action == Action.CROUCH_GUARD_RECOV.value: return 14
        if action == Action.AIR_GUARD_RECOV or action == Action.AIR_GUARD_RECOV.value: return 15
        if action == Action.STAND_RECOV or action == Action.STAND_RECOV.value: return 16
        if action == Action.CROUCH_RECOV or action == Action.CROUCH_RECOV.value: return 17
        if action == Action.AIR_RECOV or action == Action.AIR_RECOV.value: return 18
        if action == Action.CHANGE_DOWN or action == Action.CHANGE_DOWN.value: return 19
        if action == Action.DOWN or action == Action.DOWN.value: return 20
        if action == Action.RISE or action == Action.RISE.value: return 21
        if action == Action.LANDING or action == Action.LANDING.value: return 22
        if action == Action.THROW_A or action == Action.THROW_A.value: return 23
        if action == Action.THROW_B or action == Action.THROW_B.value: return 24
        if action == Action.THROW_HIT or action == Action.THROW_HIT.value: return 25
        if action == Action.THROW_SUFFER or action == Action.THROW_SUFFER.value: return 26
        if action == Action.STAND_A or action == Action.STAND_A.value: return 27
        if action == Action.STAND_B or action == Action.STAND_B.value: return 28
        if action == Action.CROUCH_A or action == Action.CROUCH_A.value: return 29
        if action == Action.CROUCH_B or action == Action.CROUCH_B.value: return 30
        if action == Action.AIR_A or action == Action.AIR_A.value: return 31
        if action == Action.AIR_B or action == Action.AIR_B.value: return 32
        if action == Action.AIR_DA or action == Action.AIR_DA.value: return 33
        if action == Action.AIR_DB or action == Action.AIR_DB.value: return 34
        if action == Action.STAND_FA or action == Action.STAND_FA.value: return 35
        if action == Action.STAND_FB or action == Action.STAND_FB.value: return 36
        if action == Action.CROUCH_FA or action == Action.CROUCH_FA.value: return 37
        if action == Action.CROUCH_FB or action == Action.CROUCH_FB.value: return 38
        if action == Action.AIR_FA or action == Action.AIR_FA.value: return 39
        if action == Action.AIR_FB or action == Action.AIR_FB.value: return 40
        if action == Action.AIR_UA or action == Action.AIR_UA.value: return 41
        if action == Action.AIR_UB or action == Action.AIR_UB.value: return 42
        if action == Action.STAND_D_DF_FA or action == Action.STAND_D_DF_FA.value: return 43
        if action == Action.STAND_D_DF_FB or action == Action.STAND_D_DF_FB.value: return 44
        if action == Action.STAND_F_D_DFA or action == Action.STAND_F_D_DFA.value: return 45
        if action == Action.STAND_F_D_DFB or action == Action.STAND_F_D_DFB.value: return 46
        if action == Action.STAND_D_DB_BA or action == Action.STAND_D_DB_BA.value: return 47
        if action == Action.STAND_D_DB_BB or action == Action.STAND_D_DB_BB.value: return 48
        if action == Action.AIR_D_DF_FA or action == Action.AIR_D_DF_FA.value: return 49
        if action == Action.AIR_D_DF_FB or action == Action.AIR_D_DF_FB.value: return 50
        if action == Action.AIR_F_D_DFA or action == Action.AIR_F_D_DFA.value: return 51
        if action == Action.AIR_F_D_DFB or action == Action.AIR_F_D_DFB.value: return 52
        if action == Action.AIR_D_DB_BA or action == Action.AIR_D_DB_BA.value: return 53
        if action == Action.AIR_D_DB_BB or action == Action.AIR_D_DB_BB.value: return 54
        if action == Action.STAND_D_DF_FC or action == Action.STAND_D_DF_FC.value: return 55

USELESS_ACTIONS = [
    Action.STAND,
    Action.AIR,
    Action.STAND_GUARD_RECOV,
    Action.CROUCH_GUARD_RECOV,
    Action.AIR_GUARD_RECOV,
    Action.STAND_RECOV,
    Action.CROUCH_RECOV,
    Action.AIR_RECOV,
    Action.CHANGE_DOWN,
    Action.DOWN,
    Action.RISE,
    Action.LANDING,
    Action.THROW_HIT,
    Action.THROW_SUFFER
]

AIR_ACTIONS = [
    Action.AIR_GUARD,
    Action.AIR_A,
    Action.AIR_B,
    Action.AIR_DA,
    Action.AIR_DB,
    Action.AIR_FA,
    Action.AIR_FB,
    Action.AIR_UA,
    Action.AIR_UB,
    Action.AIR_D_DF_FA,
    Action.AIR_D_DF_FB,
    Action.AIR_F_D_DFA,
    Action.AIR_F_D_DFB,
    Action.AIR_D_DB_BA,
    Action.AIR_D_DB_BB
]

GROUND_ACTIONS = [
    Action.STAND_D_DB_BA,
    Action.BACK_STEP,
    Action.FORWARD_WALK,
    Action.DASH,
    Action.JUMP,
    Action.FOR_JUMP,
    Action.BACK_JUMP,
    Action.STAND_GUARD,
    Action.CROUCH_GUARD,
    Action.THROW_A,
    Action.THROW_B,
    Action.STAND_A,
    Action.STAND_B,
    Action.CROUCH_A,
    Action.CROUCH_B,
    Action.STAND_FA,
    Action.STAND_FB,
    Action.CROUCH_FA,
    Action.CROUCH_FB,
    Action.STAND_D_DF_FA,
    Action.STAND_D_DF_FB,
    Action.STAND_F_D_DFA,
    Action.STAND_F_D_DFB,
    Action.STAND_D_DB_BB
]

# ALL_ACTIONS = [a.value for a in Action]
# ALL_USEFUL_ACTIONS = [a.value for a in Action if a not in USELESS_ACTIONS]
# AIR_ACTIONS = [a.value for a in AIR_ACTIONS]
# GROUND_ACTIONS = [a.value for a in GROUND_ACTIONS]
