from enum import IntEnum

''' simple helper class to enumerate actions in the grid levels '''
class Actions_v10(IntEnum):
    r0 = 0
    r1 = 1
    r2 = 2
    r3 = 3

    # get the enum name without the class
    def __str__(self): return self.name