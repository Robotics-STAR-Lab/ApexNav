class HABITAT_STATE:
    READY = 0
    ACTION_EXEC = 1
    ACTION_FINISH = 2
    EPISODE_FINISH = 3


class ROS_STATE:
    INIT = 0
    WAIT_TRIGGER = 1
    PLAN_ACTION = 2
    WAIT_ACTION_FINISH = 3
    PUB_ACTION = 4
    FINISH = 5


class ACTION:
    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    TURN_DOWN = 4
    TURN_UP = 5


RESULT_TYPES = [
    "success",
    "infeasible",
    "no frontier",
    "false positive",
    "stepout true negative",
    "stepout feasible",
    "stucking",
    "[no frontier] false negative",
    "[stucking] false negative",
    "[stepout] false negative",
]


class FINAL_RESULT:
    EXPLORE = 0
    SEARCH_OBJECT = 1
    STUCKING = 2
    NO_FRONTIER = 3
    REACH_OBJECT = 4


class EXPL_RESULT:
    EXPLORATION = 0
    SEARCH_BEST_OBJECT = 1
    SEARCH_OVER_DEPTH_OBJECT = 2
    SEARCH_SUSPICIOUS_OBJECT = 3
    NO_PASSABLE_FRONTIER = 4
    NO_COVERABLE_FRONTIER = 5
    SEARCH_EXTREME = 6
