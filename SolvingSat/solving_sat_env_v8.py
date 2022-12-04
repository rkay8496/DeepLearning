import random

import pygame

from SolvingSat.actions import Actions
import numpy as np
from gym.spaces import Discrete, MultiDiscrete
import mtl

class SolvingSatEnv_v8():
    metadata = {'render_modes': ['human']}

    def __init__(self, **kwargs):
        super().__init__()

        self.traces = {
            # door
            'p0': [(0, False)],
            'p1': [(0, False)],
            # request
            'q0': [(0, False)],
            'q1': [(0, False)],
            # power
            'r0': [(0, True)],
            # action
            's0': [(0, False)],
            's1': [(0, False)]
        }

        self.env_bits = []
        for l in ['00', '01', '10']:
            for k in ['0', '1']:
                for j in ['00', '01', '10']:
                    for i in ['00', '01', '10']:
                        self.env_bits.append(l + k + j + i)

        self.env_formula = {
            '(G((((~p0 & ~p1) & (~q0 & ~q1)) -> (X~p0 & X~p1))))': False,
            '(G((((p0 & ~p1) & (~q0 & ~q1)) -> (Xp0 & X~p1))))': False,
            '(G((((~p0 & ~p1) & (q0 & ~q1)) -> (X~p0 & Xp1))))': False,
            '(G((((p0 & ~p1) & (~q0 & q1)) -> (X~p0 & Xp1))))': False,
            '(G((((~p0 & p1) & (q0 & ~q1)) -> (Xp0 & X~p1))))': False,
            '(G((((~p0 & p1) & (~q0 & q1)) -> (X~p0 & X~p1))))': False,
            '(G((~p0 & ~p1) -> (~(X~q0 & Xq1))))': False,
            '(G((p0 & ~p1) -> (~(Xq0 & X~q1))))': False,
            '(G(((q0 & ~q1) & (~(Xp0 & X~p1))) -> (Xq0 & X~q1)))': False,
            '(G(((~q0 & q1) & (~(X~p0 & X~p1))) -> (X~q0 & Xq1)))': False,
            '(G((~s0 & s1) -> (X~r0)))': False,
            '(G((s0 & ~s1) -> (Xr0)))': False,
            '(G((~s0 & ~s1) -> (Xr0 <-> r0)))': False,
        }

        self.env_spec = '('
        for key in self.env_formula.keys():
            self.env_spec += key + ' & '
        self.env_spec = self.env_spec[:-3]
        self.env_spec += ')'

        self.sys_formula = {
            '(G((Xq0 & X~q1) -> (X~s0 & Xs1)))': False,
            '(G(((X~p0 & X~p1) & (X~q0 & Xq1)) -> (Xs0 & X~s1)))': False,
        }

        self.sys_spec = '('
        for key in self.sys_formula.keys():
            self.sys_spec += key + ' & '
        self.sys_spec = self.sys_spec[:-3]
        self.sys_spec += ')'

        self.spec = '(' + self.env_spec + ' -> ' + self.sys_spec + ')'

        # dimensions of the grid
        self.width = kwargs.get('width', 18)
        self.height = kwargs.get('height', 1)

        # define the maximum x and y values
        self.max_x = self.width - 1
        self.max_y = self.height - 1

        self.action_space = Discrete(3)

        self.observation_space = MultiDiscrete([self.width, self.height])

        self.x = 0
        self.y = 0

        # the start and end positions in the grid
        # - by default these are the top-left and bottom-right respectively
        self.start = kwargs.get('start', [np.random.choice(np.arange(self.width)), 0])
        self.end = kwargs.get('end', [self.max_x, self.max_y])

        # Baby Robot's initial position
        # - by default this is the grid start
        self.initial_pos = kwargs.get('initial_pos', self.start)

        # Baby Robot's position in the grid
        self.x = self.initial_pos[0]
        self.y = self.initial_pos[1]

    def take_env(self, action):
        self.y = 0

        def compute_inputs():
            possible_inputs = []
            for bits in self.env_bits:
                if bits[4] == '1':
                    self.traces['p0'].append((len(self.traces['p0']), True))
                else:
                    self.traces['p0'].append((len(self.traces['p0']), False))
                if bits[3] == '1':
                    self.traces['p1'].append((len(self.traces['p1']), True))
                else:
                    self.traces['p1'].append((len(self.traces['p1']), False))
                if bits[2] == '1':
                    self.traces['q0'].append((len(self.traces['q0']), True))
                else:
                    self.traces['q0'].append((len(self.traces['q0']), False))
                if bits[1] == '1':
                    self.traces['q1'].append((len(self.traces['q1']), True))
                else:
                    self.traces['q1'].append((len(self.traces['q1']), False))
                if bits[0] == '1':
                    self.traces['r0'].append((len(self.traces['r0']), True))
                else:
                    self.traces['r0'].append((len(self.traces['r0']), False))
                phi = mtl.parse(self.spec)
                evaluation = phi(self.traces, quantitative=False)
                if evaluation:
                    possible_inputs.append(bits)
                self.traces['p0'].pop(len(self.traces['p0']) - 1)
                self.traces['p1'].pop(len(self.traces['p1']) - 1)
                self.traces['q0'].pop(len(self.traces['q0']) - 1)
                self.traces['q1'].pop(len(self.traces['q1']) - 1)
                self.traces['r0'].pop(len(self.traces['r0']) - 1)
            return possible_inputs

        possible_inputs = compute_inputs()
        selected_inputs = np.random.choice(possible_inputs)
        self.x = self.env_bits.index(selected_inputs)
        if selected_inputs[4] == '1':
            self.traces['p0'].append((len(self.traces['p0']), True))
        else:
            self.traces['p0'].append((len(self.traces['p0']), False))
        if selected_inputs[3] == '1':
            self.traces['p1'].append((len(self.traces['p1']), True))
        else:
            self.traces['p1'].append((len(self.traces['p1']), False))
        if selected_inputs[2] == '1':
            self.traces['q0'].append((len(self.traces['q0']), True))
        else:
            self.traces['q0'].append((len(self.traces['q0']), False))
        if selected_inputs[1] == '1':
            self.traces['q1'].append((len(self.traces['q1']), True))
        else:
            self.traces['q1'].append((len(self.traces['q1']), False))
        if selected_inputs[0] == '1':
            self.traces['r0'].append((len(self.traces['r0']), True))
        else:
            self.traces['r0'].append((len(self.traces['r0']), False))
        print(self.traces)

    def step(self, action):
        self.action = action

        if action == 0:
            self.traces['s0'].append((len(self.traces['s0']), False))
            self.traces['s1'].append((len(self.traces['s1']), False))
        elif action == 1:
            self.traces['s0'].append((len(self.traces['s0']), False))
            self.traces['s1'].append((len(self.traces['s1']), True))
        elif action == 2:
            self.traces['s0'].append((len(self.traces['s0']), True))
            self.traces['s1'].append((len(self.traces['s1']), False))
        obs = np.array([self.x, self.y])

        for key in self.sys_formula.keys():
            self.sys_formula[key] = False

        reward = 0
        done = False
        phi = mtl.parse(self.spec)
        evaluation = phi(self.traces, quantitative=False)
        if evaluation:
            reward += 50
            done = True
        else:
            reward += -500
            done = False

        self.take_env(action=action)
        info = {}
        return obs, reward, done, info

    def reset(self):
        self.x = 9
        self.y = 0
        self.traces = {
            'p0': [(0, False)],
            'p1': [(0, False)],
            'q0': [(0, False)],
            'q1': [(0, False)],
            'r0': [(0, True)],
            's0': [(0, False)],
            's1': [(0, False)]
        }
        self.take_env(action=None)
        return np.array([self.x, self.y])

    def render(self, mode='human', e=0, rewards=0):
        if mode == 'human':
            print("(input: %d , output: %d , epi_num: %d , rewards: %d )" % (self.x, self.action, e, rewards))
        else:
            super().render(mode=mode)  # just raise an exception