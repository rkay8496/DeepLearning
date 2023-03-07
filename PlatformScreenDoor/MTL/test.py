from flloat.parser.ltlf import LTLfParser

# parse the formula
parser = LTLfParser()
formula = "G (b -> X a)"
parsed_formula = parser(formula)

# evaluate over finite traces
t1 = [
    {"A": False, "B": True},
    {"A": True},
]
print(parsed_formula.truth(t1, 0))
t2 = [
    {"A": False, "B": False},
    {"A": True, "B": True},
    {"A": False, "B": True},
]
assert not parsed_formula.truth(t2, 0)

# from LTLf formula to DFA
dfa = parsed_formula.to_automaton()
assert dfa.accepts(t1)
assert not dfa.accepts(t2)