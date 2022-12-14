env_properties = [
    {
        'category': 'safety',
        'property': '(G((closed & none) -> Xclosed) & ' # door
                    'G((opened & none) -> Xopened) & '
                    'G((closed & open) -> Xpartially) & '
                    'G((opened & close) -> Xpartially) & '
                    'G((partially & open) -> Xopened) & '
                    'G((partially & close) -> Xclosed) & '
                    'G((closed & ~partially & ~opened) | (~closed & partially & ~opened) | '
                    '(~closed & ~partially & opened)) & '
                    'G(closed -> (Xopen | Xnone)) & ' # request
                    'G(opened -> (Xclose | Xnone)) & '
                    'G((open & X~opened) -> Xopen) & '
                    'G((close & X~closed) -> Xclose) & '
                    'G((none & ~close & ~open) | (~none & close & ~open) | (~none & ~close & open)) & '
                    'G(off -> X~power) & '  # power
                    'G(on -> Xpower) & '
                    'G(nothing -> (Xpower <-> power)))',
        'quantitative': False
    },
    {
        'category': 'liveness',
        'property': '',
        'quantitative': False
    },
]

env_specification = ''
results = list(filter(lambda item: len(item['property']) > 0, env_properties))
if len(results) > 0:
    env_specification += '('
    for x in results:
        env_specification += x['property'] + ' & '
    env_specification = env_specification[:-3]
    env_specification += ')'
print(env_specification)