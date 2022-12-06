import stl

spec = stl.parse("(F[0, 2]{x > 4})")
data = {
  'x': [(0, 10), (2, 4)],
  'y': [(0, 2), (2, 0)],
}
print(spec(data))