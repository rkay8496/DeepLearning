import mtl

spec1 = mtl.parse("(G((p & ~r) -> Xp) & G((q & ~s) -> Xq) & G((p & r & X~r) -> X~p) & G((q & s & X~s) -> X~q))")
spec2 = mtl.parse('(G(~(r & s)) & G((p -> Fr)) & G((q -> Fs)))')
data = {"p": [(0, False), (1, True), (2, True), (3, True)], "q": [(0, False), (1, True), (2, True), (3, True)], "r": [(0, False), (1, False), (2, False)], "s": [(0, False), (1, False), (2, False)]}
print(spec1)
print(spec1(data, quantitative=False))
print(spec2)
print(spec2(data, quantitative=False))
# print(True if spec2(data) > 0 else False)
