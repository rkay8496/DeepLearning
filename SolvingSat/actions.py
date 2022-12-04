from enum import IntEnum

''' simple helper class to enumerate actions in the grid levels '''
class Actions(IntEnum):
    # NotRandNotS = 0
    # RandNotS = 1
    # NotRandS = 2
    # RansS = 3
    NotS0andNotS1 = 0
    NotS0andS1 = 1
    S0andNotS1 = 2

    # get the enum name without the class
    def __str__(self): return self.name