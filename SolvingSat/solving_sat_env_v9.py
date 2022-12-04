import random

import pygame

from SolvingSat.actions import Actions
import numpy as np
from gym.spaces import Discrete, MultiDiscrete
import mtl

class SolvingSatEnv_v9():
    metadata = {'render_modes': ['human']}

    def __init__(self, **kwargs):
        super().__init__()

        self.env_formula = {
            '(G(n0 -> Xn0))': False,
            '(G((~(~r3 & ~r2 & ~r1 & ~r0) & ~(~r3 & ~r2 & r1 & ~r0) & ~(~r3 & r2 & ~r1 & ~r0) & ~(~r3 & r2 & r1 & r0)) '
            '-> (n0 <-> Xn0)))': False,
        }

        self.env_spec = '('
        for key in self.env_formula.keys():
            self.env_spec += key + ' & '
        self.env_spec = self.env_spec[:-3]
        self.env_spec += ')'

        self.sys_formula = {
            '(G((Xn0) <-> (Xb0)))': False,
            # 0 -> 0 | 8
            '(G((~r3 & ~r2 & ~r1 & ~r0) -> (X(~r3 & ~r2 & ~r1 & ~r0) | X(r3 & ~r2 & ~r1 & ~r0))))': False,
            # 1 -> 1 | 11
            '(G((~r3 & ~r2 & ~r1 & r0) -> (X(~r3 & ~r2 & ~r1 & r0) | X(r3 & ~r2 & r1 & r0))))': False,
            # 2 -> 2 | 10
            '(G((~r3 & ~r2 & r1 & ~r0) -> (X(~r3 & ~r2 & r1 & ~r0) | X(r3 & ~r2 & r1 & ~r0))))': False,
            # 3 -> 3 | 10
            '(G((~r3 & ~r2 & r1 & r0) -> (X(~r3 & ~r2 & r1 & r0) | X(r3 & ~r2 & r1 & ~r0))))': False,
            # 4 -> 4 | 9
            '(G((~r3 & r2 & ~r1 & ~r0) -> (X(~r3 & r2 & ~r1 & ~r0) | X(r3 & ~r2 & ~r1 & r0))))': False,
            # 5 -> 5 | 9
            '(G((~r3 & r2 & ~r1 & r0) -> (X(~r3 & r2 & ~r1 & r0) | X(r3 & ~r2 & ~r1 & r0))))': False,
            # 6 -> 6 | 9
            '(G((~r3 & r2 & r1 & ~r0) -> (X(~r3 & r2 & r1 & ~r0) | X(r3 & ~r2 & ~r1 & r0))))': False,
            # 7 -> 7 | 8
            '(G((~r3 & r2 & r1 & r0) -> (X(~r3 & r2 & r1 & r0) | X(r3 & ~r2 & ~r1 & ~r0))))': False,
            # 8 -> 8 | 9 | 11
            '(G((r3 & ~r2 & ~r1 & ~r0) -> (X(r3 & ~r2 & ~r1 & ~r0) | X(r3 & ~r2 & ~r1 & r0) | X(r3 & ~r2 & r1 & r0))))': False,
            # 9 -> 9 | 8 | 10
            '(G((r3 & ~r2 & ~r1 & r0) -> (X(r3 & ~r2 & ~r1 & r0) | X(r3 & ~r2 & ~r1 & ~r0) | X(r3 & ~r2 & r1 & ~r0))))': False,
            # 10 -> 10 | 9 | 11
            '(G((r3 & ~r2 & r1 & ~r0) -> (X(r3 & ~r2 & r1 & ~r0) | X(r3 & ~r2 & ~r1 & r0) | X(r3 & ~r2 & r1 & r0))))': False,
            # 11 -> 11 | 8 | 10
            '(G((r3 & ~r2 & r1 & r0) -> (X(r3 & ~r2 & r1 & r0) | X(r3 & ~r2 & ~r1 & ~r0) | X(r3 & ~r2 & r1 & ~r0))))': False,
            '(G(Xn0 <-> Xb0))': False,
            '(G(((~r3 & ~r2 & ~r1 & ~r0) & Xn0) -> X(~r3 & ~r2 & ~r1 & ~r0)))': False,
            '(G(((~r3 & ~r2 & r1 & ~r0) & Xn0) -> X(~r3 & ~r2 & r1 & ~r0)))': False,
            '(G(((~r3 & r2 & ~r1 & ~r0) & Xn0) -> X(~r3 & r2 & ~r1 & ~r0)))': False,
            '(G(((~r3 & r2 & r1 & r0) & Xn0) -> X(~r3 & r2 & r1 & r0)))': False,
            '(G(F((~r3 & ~r2 & ~r1 & ~r0) | n0)))': False,
            '(G(F((~r3 & ~r2 & ~r1 & r0) | n0)))': False,
            '(G(F((~r3 & ~r2 & r1 & ~r0) | n0)))': False,
            '(G(F((~r3 & ~r2 & r1 & r0) | n0)))': False,
            '(G(F((~r3 & r2 & ~r1 & ~r0) | n0)))': False,
            '(G(F((~r3 & r2 & ~r1 & r0) | n0)))': False,
            '(G(F((~r3 & r2 & r1 & ~r0) | n0)))': False,
            '(G(F((~r3 & r2 & r1 & r0) | n0)))': False,
            '(G(F((r3 & ~r2 & ~r1 & ~r0) | n0)))': False,
            '(G(F((r3 & ~r2 & ~r1 & r0) | n0)))': False,
            '(G(F((r3 & ~r2 & r1 & ~r0) | n0)))': False,
            '(G(F((r3 & ~r2 & r1 & r0) | n0)))': False
        }

        self.sys_spec = '('
        for key in self.sys_formula.keys():
            self.sys_spec += key + ' & '
        self.sys_spec = self.sys_spec[:-3]
        self.sys_spec += ')'

        self.spec = '(' + self.env_spec + ' -> ' + self.sys_spec + ')'

        # dimensions of the grid
        self.width = kwargs.get('width', 2)
        self.height = kwargs.get('height', 12)
        self.depth = kwargs.get('depth', 2)

        # define the maximum x and y values
        self.max_x = self.width - 1
        self.max_y = self.height - 1
        self.max_z = self.depth - 1

        self.action_space = MultiDiscrete([self.height, self.depth])

        self.observation_space = MultiDiscrete([self.width, 1])

        self.x = 0
        self.y = 0
        self.z = 0

        self.start = kwargs.get('start', [0, 0])
        self.end = kwargs.get('end', [self.max_x, 0])

        self.initial_pos = kwargs.get('initial_pos', self.start)

        self.x = self.initial_pos[0]
        self.y = 11
        self.z = 0

    def take_env(self, action):
        def compute_inputs():
            possible_inputs = []
            for v in range(self.width):
                if v == 0:
                    self.traces['n0'].append((len(self.traces['n0']), False))
                elif v == 1:
                    self.traces['n0'].append((len(self.traces['n0']), True))
                phi = mtl.parse(self.spec)
                evaluation = phi(self.traces, quantitative=False)
                if evaluation:
                    possible_inputs.append(v)
                self.traces['n0'].pop(len(self.traces['n0']) - 1)
            return possible_inputs

        possible_inputs = compute_inputs()
        selected_inputs = np.random.choice(possible_inputs)
        self.x = selected_inputs
        if selected_inputs == 0:
            self.traces['n0'].append((len(self.traces['n0']), False))
        elif selected_inputs == 1:
            self.traces['n0'].append((len(self.traces['n0']), True))

    def step(self, action):
        self.action = action

        if action == 0:
            self.traces['r0'].append((len(self.traces['r0']), False))
            self.traces['r1'].append((len(self.traces['r1']), False))
            self.traces['r2'].append((len(self.traces['r2']), False))
            self.traces['r3'].append((len(self.traces['r3']), False))
            self.traces['b0'].append((len(self.traces['b0']), False))
        if action == 1:
            self.traces['r0'].append((len(self.traces['r0']), False))
            self.traces['r1'].append((len(self.traces['r1']), False))
            self.traces['r2'].append((len(self.traces['r2']), False))
            self.traces['r3'].append((len(self.traces['r3']), False))
            self.traces['b0'].append((len(self.traces['b0']), True))
        if action == 2:
            self.traces['r0'].append((len(self.traces['r0']), True))
            self.traces['r1'].append((len(self.traces['r1']), False))
            self.traces['r2'].append((len(self.traces['r2']), False))
            self.traces['r3'].append((len(self.traces['r3']), False))
            self.traces['b0'].append((len(self.traces['b0']), False))
        if action == 3:
            self.traces['r0'].append((len(self.traces['r0']), True))
            self.traces['r1'].append((len(self.traces['r1']), False))
            self.traces['r2'].append((len(self.traces['r2']), False))
            self.traces['r3'].append((len(self.traces['r3']), False))
            self.traces['b0'].append((len(self.traces['b0']), True))
        if action == 4:
            self.traces['r0'].append((len(self.traces['r0']), False))
            self.traces['r1'].append((len(self.traces['r1']), True))
            self.traces['r2'].append((len(self.traces['r2']), False))
            self.traces['r3'].append((len(self.traces['r3']), False))
            self.traces['b0'].append((len(self.traces['b0']), False))
        if action == 5:
            self.traces['r0'].append((len(self.traces['r0']), False))
            self.traces['r1'].append((len(self.traces['r1']), True))
            self.traces['r2'].append((len(self.traces['r2']), False))
            self.traces['r3'].append((len(self.traces['r3']), False))
            self.traces['b0'].append((len(self.traces['b0']), True))
        if action == 6:
            self.traces['r0'].append((len(self.traces['r0']), True))
            self.traces['r1'].append((len(self.traces['r1']), True))
            self.traces['r2'].append((len(self.traces['r2']), False))
            self.traces['r3'].append((len(self.traces['r3']), False))
            self.traces['b0'].append((len(self.traces['b0']), False))
        if action == 7:
            self.traces['r0'].append((len(self.traces['r0']), True))
            self.traces['r1'].append((len(self.traces['r1']), True))
            self.traces['r2'].append((len(self.traces['r2']), False))
            self.traces['r3'].append((len(self.traces['r3']), False))
            self.traces['b0'].append((len(self.traces['b0']), True))
        if action == 8:
            self.traces['r0'].append((len(self.traces['r0']), False))
            self.traces['r1'].append((len(self.traces['r1']), False))
            self.traces['r2'].append((len(self.traces['r2']), True))
            self.traces['r3'].append((len(self.traces['r3']), False))
            self.traces['b0'].append((len(self.traces['b0']), False))
        if action == 9:
            self.traces['r0'].append((len(self.traces['r0']), False))
            self.traces['r1'].append((len(self.traces['r1']), False))
            self.traces['r2'].append((len(self.traces['r2']), True))
            self.traces['r3'].append((len(self.traces['r3']), False))
            self.traces['b0'].append((len(self.traces['b0']), True))
        if action == 10:
            self.traces['r0'].append((len(self.traces['r0']), True))
            self.traces['r1'].append((len(self.traces['r1']), False))
            self.traces['r2'].append((len(self.traces['r2']), True))
            self.traces['r3'].append((len(self.traces['r3']), False))
            self.traces['b0'].append((len(self.traces['b0']), False))
        if action == 11:
            self.traces['r0'].append((len(self.traces['r0']), True))
            self.traces['r1'].append((len(self.traces['r1']), False))
            self.traces['r2'].append((len(self.traces['r2']), True))
            self.traces['r3'].append((len(self.traces['r3']), False))
            self.traces['b0'].append((len(self.traces['b0']), True))
        if action == 12:
            self.traces['r0'].append((len(self.traces['r0']), False))
            self.traces['r1'].append((len(self.traces['r1']), True))
            self.traces['r2'].append((len(self.traces['r2']), True))
            self.traces['r3'].append((len(self.traces['r3']), False))
            self.traces['b0'].append((len(self.traces['b0']), False))
        if action == 13:
            self.traces['r0'].append((len(self.traces['r0']), False))
            self.traces['r1'].append((len(self.traces['r1']), True))
            self.traces['r2'].append((len(self.traces['r2']), True))
            self.traces['r3'].append((len(self.traces['r3']), False))
            self.traces['b0'].append((len(self.traces['b0']), True))
        if action == 14:
            self.traces['r0'].append((len(self.traces['r0']), True))
            self.traces['r1'].append((len(self.traces['r1']), True))
            self.traces['r2'].append((len(self.traces['r2']), True))
            self.traces['r3'].append((len(self.traces['r3']), False))
            self.traces['b0'].append((len(self.traces['b0']), False))
        if action == 15:
            self.traces['r0'].append((len(self.traces['r0']), True))
            self.traces['r1'].append((len(self.traces['r1']), True))
            self.traces['r2'].append((len(self.traces['r2']), True))
            self.traces['r3'].append((len(self.traces['r3']), False))
            self.traces['b0'].append((len(self.traces['b0']), True))
        if action == 16:
            self.traces['r0'].append((len(self.traces['r0']), False))
            self.traces['r1'].append((len(self.traces['r1']), False))
            self.traces['r2'].append((len(self.traces['r2']), False))
            self.traces['r3'].append((len(self.traces['r3']), True))
            self.traces['b0'].append((len(self.traces['b0']), False))
        if action == 17:
            self.traces['r0'].append((len(self.traces['r0']), False))
            self.traces['r1'].append((len(self.traces['r1']), False))
            self.traces['r2'].append((len(self.traces['r2']), False))
            self.traces['r3'].append((len(self.traces['r3']), True))
            self.traces['b0'].append((len(self.traces['b0']), True))
        if action == 18:
            self.traces['r0'].append((len(self.traces['r0']), True))
            self.traces['r1'].append((len(self.traces['r1']), False))
            self.traces['r2'].append((len(self.traces['r2']), False))
            self.traces['r3'].append((len(self.traces['r3']), True))
            self.traces['b0'].append((len(self.traces['b0']), False))
        if action == 19:
            self.traces['r0'].append((len(self.traces['r0']), True))
            self.traces['r1'].append((len(self.traces['r1']), False))
            self.traces['r2'].append((len(self.traces['r2']), False))
            self.traces['r3'].append((len(self.traces['r3']), True))
            self.traces['b0'].append((len(self.traces['b0']), True))
        if action == 20:
            self.traces['r0'].append((len(self.traces['r0']), False))
            self.traces['r1'].append((len(self.traces['r1']), True))
            self.traces['r2'].append((len(self.traces['r2']), False))
            self.traces['r3'].append((len(self.traces['r3']), True))
            self.traces['b0'].append((len(self.traces['b0']), False))
        if action == 21:
            self.traces['r0'].append((len(self.traces['r0']), False))
            self.traces['r1'].append((len(self.traces['r1']), True))
            self.traces['r2'].append((len(self.traces['r2']), False))
            self.traces['r3'].append((len(self.traces['r3']), True))
            self.traces['b0'].append((len(self.traces['b0']), True))
        if action == 22:
            self.traces['r0'].append((len(self.traces['r0']), True))
            self.traces['r1'].append((len(self.traces['r1']), True))
            self.traces['r2'].append((len(self.traces['r2']), False))
            self.traces['r3'].append((len(self.traces['r3']), True))
            self.traces['b0'].append((len(self.traces['b0']), False))
        if action == 23:
            self.traces['r0'].append((len(self.traces['r0']), True))
            self.traces['r1'].append((len(self.traces['r1']), True))
            self.traces['r2'].append((len(self.traces['r2']), False))
            self.traces['r3'].append((len(self.traces['r3']), True))
            self.traces['b0'].append((len(self.traces['b0']), True))
        obs = np.array([self.x, 0])

        for key in self.sys_formula.keys():
            self.sys_formula[key] = False

        reward = 0
        done = False
        phi = mtl.parse(self.spec)
        evaluation = phi(self.traces, quantitative=False)
        if evaluation:
            reward += 50
            done = False
        else:
            reward += -500
            done = True

        info = {}
        return obs, reward, done, info

    def reset(self):
        self.traces = {
            # 1 bits for nemo
            'n0': [(0, False)],
            # 4 bits for regions
            'r0': [(0, True)],
            'r1': [(0, True)],
            'r2': [(0, False)],
            'r3': [(0, True)],
            # 1 bits for beep
            'b0': [(0, False)]
        }
        self.take_env(action=None)
        return np.array([self.x, 0])

    def render(self, mode='human', e=0, rewards=0):
        if mode == 'human':
            print("(input: %d , output: %d , epi_num: %d , rewards: %d )" % (self.x, self.action, e, rewards))
        else:
            super().render(mode=mode)  # just raise an exception