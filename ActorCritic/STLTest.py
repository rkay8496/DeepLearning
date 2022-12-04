import numpy as np

import signal_tl as stl

p = stl.Predicate("p") > 10
q = stl.Predicate('q') < 5

phi1 = p | q
# phi2 = stl.And([stl.Not(p), q])
# phi = stl.Always(phi)
# phi = stl.Eventually(p)
# phi = stl.Or(p, q)

print(phi1)
# print(phi2)

# trace = {
#     "p": stl.Signal([10, 20, 30, 5],  [0, 1, 2, 3]),
#     'q': stl.Signal([4, 5, 6, 7], [0, 1, 2, 3])
# }
#
# rob = stl.compute_robustness(phi, trace)
# print(rob.at(0))
