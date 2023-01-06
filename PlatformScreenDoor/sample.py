import stl

spec1 = stl.parse("G((standby & {currentspeed > minimumspeed}) -> F[1, 1](resume & accelerate))")
data = {"p": [(0, False), (1, True), (2, True), (3, True)], "q": [(0, False), (1, True), (2, True), (3, True)], "r": [(0, False), (1, False), (2, False)], "s": [(0, False), (1, False), (2, False)]}
print(spec1)
print(spec1(data))
