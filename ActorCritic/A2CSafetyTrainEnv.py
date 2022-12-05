import numpy as np
from gym.spaces import Discrete
import mtl

class ActorCritic():
    metadata = {'render_modes': ['human']}

    def __init__(self, **kwargs):
        super().__init__()

        self.env_properties = [
            {
                'category': 'safety',
                'property': '(G(p -> (Xp | Xq)) & G(q -> (Xq | Xp | Xr)) & G(r -> (Xr | Xq)))',
                'quantitative': False
            },
            {
                'category': 'liveness',
                'property': '',
                'quantitative': False
            },
        ]

        self.env_specification = ''
        results = list(filter(lambda item: len(item['property']) > 0, self.env_properties))
        if len(results) > 0:
            self.env_specification += '('
            for x in results:
                self.env_specification += x['property'] + ' & '
            self.env_specification = self.env_specification[:-3]
            self.env_specification += ')'

        self.sys_properties = [
            {
                'category': 'safety',
                'property': '(G(Xr -> Xu))',
                'quantitative': False
            },
            {
                'category': 'liveness',
                'property': '(GF(p -> t) & GF(q -> u))',
                'quantitative': False
            },
        ]

        self.sys_specification = ''
        results = list(filter(lambda item: len(item['property']) > 0, self.sys_properties))
        if len(results) > 0:
            self.sys_specification += '('
            for x in results:
                self.sys_specification += x['property'] + ' & '
            self.sys_specification = self.sys_specification[:-3]
            self.sys_specification += ')'

        self.specification = '(' + self.env_specification + ' -> ' + self.sys_specification + ')'

        self.width = kwargs.get('width', 3)
        self.max_x = self.width - 1
        self.action_space = Discrete(2)
        self.observation_space = Discrete(self.width)
        self.x = 0
        self.start = kwargs.get('start', self.x)
        self.initial_pos = kwargs.get('initial_pos', self.start)
        self.x = self.initial_pos

    def take_env(self):
        def compute_inputs():
            possible_inputs = []
            for v in range(self.width):
                if v == 0:
                    self.traces['p'].append((len(self.traces['p']), True))
                    self.traces['q'].append((len(self.traces['q']), False))
                    self.traces['r'].append((len(self.traces['r']), False))
                elif v == 1:
                    self.traces['p'].append((len(self.traces['p']), False))
                    self.traces['q'].append((len(self.traces['q']), True))
                    self.traces['r'].append((len(self.traces['r']), False))
                elif v == 2:
                    self.traces['p'].append((len(self.traces['p']), False))
                    self.traces['q'].append((len(self.traces['q']), False))
                    self.traces['r'].append((len(self.traces['r']), True))
                safety_eval = True
                if len(self.env_properties[0]['property']) > 0:
                    phi = mtl.parse(self.env_properties[0]['property'])
                    safety_eval = phi(self.traces, quantitative=self.env_properties[0]['quantitative'])
                liveness_eval = True
                if len(self.env_properties[1]['property']) > 0:
                    phi = mtl.parse(self.env_properties[1]['property'])
                    liveness_eval = phi(self.traces, quantitative=self.env_properties[1]['quantitative'])
                if safety_eval and liveness_eval:
                    possible_inputs.append(v)
                self.traces['p'].pop(len(self.traces['p']) - 1)
                self.traces['q'].pop(len(self.traces['q']) - 1)
                self.traces['r'].pop(len(self.traces['r']) - 1)
            return possible_inputs

        possible_inputs = compute_inputs()
        selected_input = np.random.choice(possible_inputs)
        self.x = selected_input
        if selected_input == 0:
            self.traces['p'].append((len(self.traces['p']), True))
            self.traces['q'].append((len(self.traces['q']), False))
            self.traces['r'].append((len(self.traces['r']), False))
        elif selected_input == 1:
            self.traces['p'].append((len(self.traces['p']), False))
            self.traces['q'].append((len(self.traces['q']), True))
            self.traces['r'].append((len(self.traces['r']), False))
        elif selected_input == 2:
            self.traces['p'].append((len(self.traces['p']), False))
            self.traces['q'].append((len(self.traces['q']), False))
            self.traces['r'].append((len(self.traces['r']), True))

    def step(self, action):
        self.action = action
        if self.action == 0:
            self.traces['t'].append((len(self.traces['t']), True))
            self.traces['u'].append((len(self.traces['u']), False))
        elif self.action == 1:
            self.traces['t'].append((len(self.traces['t']), False))
            self.traces['u'].append((len(self.traces['u']), True))
        obs = np.array(self.x)

        info = {
            'satisfiable': False
        }
        reward = 0
        done = False
        safety_eval = True
        if len(self.sys_properties[0]['property']) > 0:
            phi = mtl.parse(self.sys_properties[0]['property'])
            safety_eval = phi(self.traces, quantitative=self.sys_properties[0]['quantitative'])
        liveness_eval = True
        if len(self.sys_properties[1]['property']) > 0:
            phi = mtl.parse(self.sys_properties[1]['property'])
            liveness_eval = phi(self.traces, quantitative=self.sys_properties[1]['quantitative'])
        if safety_eval and liveness_eval:
            reward += 100
            done = True
            info['satisfiable'] = True
        elif safety_eval and not liveness_eval:
            reward += 50
            done = False
            info['satisfiable'] = False
        # elif not safety_eval and liveness_eval:
        #     reward += -500
        #     done = False
        #     info['satisfiable'] = False
        else:
            reward += -100
            done = True
            info['satisfiable'] = False
        return obs, reward, done, info

    def reset(self):
        self.traces = {
            # safety distance
            'p': [(0, True)],
            # warning distance
            'q': [(0, False)],
            # danger distance
            'r': [(0, False)],
            # acceleration
            't': [(0, True)],
            # deceleration
            'u': [(0, False)],
        }
        self.take_env()
        return np.array(self.x)

    def render(self, mode='human', e=0, rewards=0):
        if mode == 'human':
            print("(input: %d , output: %d , epi_num: %d , rewards: %d )" % (self.x, self.action, e, rewards))
        else:
            super().render(mode=mode)  # just raise an exception