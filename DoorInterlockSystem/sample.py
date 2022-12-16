import mtl

spec1 = mtl.parse("(G((p & ~r) -> Xp) & G((q & ~s) -> Xq) & G((p & r & X~r) -> X~p) & G((q & s & X~s) -> X~q))")
spec2 = mtl.parse('(G(~(r & s)) & G((Fr -> p)) & G((Fs -> q)))')
data = {'p': [(0, True), (1, True)], 'q': [(0, False), (1, False)], 'r': [(0, False), (1, True)], 's': [(0, False), (1, False)]}
print(spec1)
print(spec1(data, quantitative=False))
print(spec2)
print(spec2(data, quantitative=False))
# print(True if spec2(data) > 0 else False)

spec1 = mtl.parse("(((!p && !q) && G((p && !r) -> X(p)) && G((q && !s) -> X(q)) && G((p && r && X!r) -> X(!p)) && G((q && s && X(!s)) -> X(!q))) -> ((!r && !s) && G(!(r && s)) && G(F(p -> r)) && G(F(q -> s))))")
spec2 = mtl.parse('(G(!(r && s)) && G(F(p -> r)) && G(F(q -> s)))')