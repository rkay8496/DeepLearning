import numpy as np
from gym.spaces import Discrete
import mtl

class ActorCritic():
    metadata = {'render_modes': ['human']}

    def __init__(self, **kwargs):
        super().__init__()

        self.env_safety_properties = {
            'G((p & ~r) -> Xp)': False,
            'G((q & ~s) -> Xq)': False,
            'G((p & r & X~r) -> X~p)': False,
            'G((q & s & X~s) -> X~q)': False,
        }
        self.env_liveness_properties = {

        }

        self.env_safety_spec = ''
        if len(self.env_safety_properties.keys()) > 0:
            self.env_safety_spec += '('
            for key in self.env_safety_properties.keys():
                self.env_safety_spec += key + ' & '
            self.env_safety_spec = self.env_safety_spec[:-3]
            self.env_safety_spec += ')'

        self.env_liveness_spec = ''
        if len(self.env_liveness_properties.keys()) > 0:
            self.env_liveness_spec += '('
            for key in self.env_liveness_properties.keys():
                self.env_liveness_spec += key + ' & '
            self.env_liveness_spec = self.env_liveness_spec[:-3]
            self.env_liveness_spec += ')'

        self.env_spec = ''
        if len(self.env_safety_properties.keys()) > 0 and len(self.env_liveness_properties.keys()) > 0:
            self.env_spec += '(' + self.env_safety_spec + ' & ' + self.env_liveness_spec + ')'
        elif len(self.env_safety_properties.keys()) > 0:
            self.env_spec += '(' + self.env_safety_spec + ')'
        elif len(self.env_liveness_properties.keys()) > 0:
            self.env_spec += '(' + self.env_liveness_spec + ')'

        self.sys_safety_properties = {
            'G(~(r & s))': False,
        }
        self.sys_liveness_properties = {
            'GF(p -> r)': False,
            'GF(q -> s)': False,
        }

        self.sys_safety_spec = ''
        if len(self.sys_safety_properties.keys()) > 0:
            self.sys_safety_spec += '('
            for key in self.sys_safety_properties.keys():
                self.sys_safety_spec += key + ' & '
            self.sys_safety_spec = self.sys_safety_spec[:-3]
            self.sys_safety_spec += ')'

        self.sys_liveness_spec = ''
        if len(self.sys_liveness_properties.keys()) > 0:
            self.sys_liveness_spec += '('
            for key in self.sys_liveness_properties.keys():
                self.sys_liveness_spec += key + ' & '
            self.sys_liveness_spec = self.sys_liveness_spec[:-3]
            self.sys_liveness_spec += ')'

        self.sys_spec = ''
        if len(self.sys_safety_properties.keys()) > 0 and len(self.sys_liveness_properties.keys()) > 0:
            self.sys_spec += '(' + self.sys_safety_spec + ' & ' + self.sys_liveness_spec + ')'
        elif len(self.sys_safety_properties.keys()) > 0:
            self.sys_spec += '(' + self.sys_safety_spec + ')'
        elif len(self.sys_liveness_properties.keys()) > 0:
            self.sys_spec += '(' + self.sys_liveness_spec + ')'

        self.spec = '(' + self.env_spec + ' -> ' + self.sys_spec + ')'

        self.width = kwargs.get('width', 4)
        self.max_x = self.width - 1
        self.action_space = Discrete(4)
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
                    self.traces['p'].append((len(self.traces['p']), False))
                    self.traces['q'].append((len(self.traces['q']), False))
                elif v == 1:
                    self.traces['p'].append((len(self.traces['p']), False))
                    self.traces['q'].append((len(self.traces['q']), True))
                elif v == 2:
                    self.traces['p'].append((len(self.traces['p']), True))
                    self.traces['q'].append((len(self.traces['q']), False))
                elif v == 3:
                    self.traces['p'].append((len(self.traces['p']), True))
                    self.traces['q'].append((len(self.traces['q']), True))
                safety_eval = True
                if len(self.env_safety_properties.keys()) > 0:
                    phi = mtl.parse(self.env_safety_spec)
                    safety_eval = phi(self.traces, quantitative=False)
                liveness_eval = True
                if len(self.env_liveness_properties.keys()) > 0:
                    phi = mtl.parse(self.env_liveness_spec)
                    liveness_eval = phi(self.traces, quantitative=False)
                if safety_eval and liveness_eval:
                    possible_inputs.append(v)
                self.traces['p'].pop(len(self.traces['p']) - 1)
                self.traces['q'].pop(len(self.traces['q']) - 1)
            return possible_inputs

        possible_inputs = compute_inputs()
        selected_input = np.random.choice(possible_inputs)
        self.x = selected_input
        if selected_input == 0:
            self.traces['p'].append((len(self.traces['p']), False))
            self.traces['q'].append((len(self.traces['q']), False))
        elif selected_input == 1:
            self.traces['p'].append((len(self.traces['p']), False))
            self.traces['q'].append((len(self.traces['q']), True))
        elif selected_input == 2:
            self.traces['p'].append((len(self.traces['p']), True))
            self.traces['q'].append((len(self.traces['q']), False))
        elif selected_input == 3:
            self.traces['p'].append((len(self.traces['p']), True))
            self.traces['q'].append((len(self.traces['q']), True))

    def step(self, action):
        self.action = action
        if self.action == 0:
            self.traces['r'].append((len(self.traces['r']), False))
            self.traces['s'].append((len(self.traces['s']), False))
        if self.action == 1:
            self.traces['r'].append((len(self.traces['r']), False))
            self.traces['s'].append((len(self.traces['s']), True))
        if self.action == 2:
            self.traces['r'].append((len(self.traces['r']), True))
            self.traces['s'].append((len(self.traces['s']), False))
        if self.action == 3:
            self.traces['r'].append((len(self.traces['r']), True))
            self.traces['s'].append((len(self.traces['s']), True))
        obs = np.array(self.x)

        info = {
            'satisfiable': False
        }
        reward = 0
        done = False
        safety_eval = True
        if len(self.sys_safety_properties.keys()) > 0:
            phi = mtl.parse(self.sys_safety_spec)
            safety_eval = phi(self.traces, quantitative=False)
        liveness_eval = True
        if len(self.sys_liveness_properties.keys()) > 0:
            phi = mtl.parse(self.sys_liveness_spec)
            liveness_eval = phi(self.traces, quantitative=False)
        if safety_eval and liveness_eval:
            reward += 500
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
            reward += -5000
            done = True
            info['satisfiable'] = False
        return obs, reward, done, info

    def reset(self):
        self.traces = {
            'p': [(0, False)],
            'q': [(0, False)],
            'r': [(0, False)],
            's': [(0, False)],
        }
        self.take_env()
        return np.array(self.x)

    def render(self, mode='human', e=0, rewards=0):
        if mode == 'human':
            print("(input: %d , output: %d , epi_num: %d , rewards: %d )" % (self.x, self.action, e, rewards))
        else:
            super().render(mode=mode)  # just raise an exception