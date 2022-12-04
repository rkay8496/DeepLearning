import random

import pygame

import numpy as np
from gym.spaces import Discrete
import mtl

class SolvingSatEnv_v10():
    metadata = {'render_modes': ['human']}

    def __init__(self, **kwargs):
        super().__init__()

        self.env_formula = {
            'G((~r2 & ~r4) -> (Xw <-> w))': False,
            'G(w -> Xw)': False,
        }

        self.env_spec = '('
        for key in self.env_formula.keys():
            self.env_spec += key + ' & '
        self.env_spec = self.env_spec[:-3]
        self.env_spec += ')'

        self.sys_formula = {
            'G(r1 -> (Xr1 | Xr2 | Xr4))': False,
            'G(r2 -> (Xr1 | Xr2 | Xr3))': False,
            'G(r3 -> (Xr2 | Xr3 | Xr4))': False,
            'G(r4 -> (Xr1 | Xr3 | Xr4))': False,
            'G((Xr1 & ~Xr2 & ~Xr3 & ~Xr4) | (~Xr1 & Xr2 & ~Xr3 & ~Xr4) | (~Xr1 & ~Xr2 & Xr3 & ~Xr4) | (~Xr1 & ~Xr2 & ~Xr3 & Xr4))': False,
            'G((r2 & Xw) -> Xr2)': False,
            'G((r4 & Xw) -> Xr4)': False,
            # 'GF(r2 | w)': False,
            # 'GF(r4 | w)': False,
        }

        self.sys_spec = '('
        for key in self.sys_formula.keys():
            self.sys_spec += key + ' & '
        self.sys_spec = self.sys_spec[:-3]
        self.sys_spec += ')'

        self.spec = '(' + self.env_spec + ' -> ' + self.sys_spec + ')'

        self.width = kwargs.get('width', 2)

        # define the maximum x and y values
        self.max_x = self.width - 1

        self.action_space = Discrete(4, start=1)
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
                    self.traces['w'].append((len(self.traces['w']), False))
                elif v == 1:
                    self.traces['w'].append((len(self.traces['w']), True))
                phi = mtl.parse(self.env_spec)
                evaluation = phi(self.traces, quantitative=False)
                if evaluation:
                    possible_inputs.append(v)
                self.traces['w'].pop(len(self.traces['w']) - 1)
            return possible_inputs

        possible_inputs = compute_inputs()
        selected_input = np.random.choice(possible_inputs)
        self.x = selected_input
        if selected_input == 0:
            self.traces['w'].append((len(self.traces['w']), False))
        elif selected_input == 1:
            self.traces['w'].append((len(self.traces['w']), True))

    def step(self, action):
        self.action = action + 1
        if self.action == 1:
            self.traces['r1'].append((len(self.traces['r1']), True))
            self.traces['r2'].append((len(self.traces['r2']), False))
            self.traces['r3'].append((len(self.traces['r3']), False))
            self.traces['r4'].append((len(self.traces['r4']), False))
        if self.action == 2:
            self.traces['r1'].append((len(self.traces['r1']), False))
            self.traces['r2'].append((len(self.traces['r2']), True))
            self.traces['r3'].append((len(self.traces['r3']), False))
            self.traces['r4'].append((len(self.traces['r4']), False))
        if self.action == 3:
            self.traces['r1'].append((len(self.traces['r1']), False))
            self.traces['r2'].append((len(self.traces['r2']), False))
            self.traces['r3'].append((len(self.traces['r3']), True))
            self.traces['r4'].append((len(self.traces['r4']), False))
        if self.action == 4:
            self.traces['r1'].append((len(self.traces['r1']), False))
            self.traces['r2'].append((len(self.traces['r2']), False))
            self.traces['r3'].append((len(self.traces['r3']), False))
            self.traces['r4'].append((len(self.traces['r4']), True))
        obs = np.array(self.x)

        reward = 0
        done = False
        phi = mtl.parse(self.sys_spec)
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
            # 1 bits for waldo
            'w': [(0, False)],
            # 4 bits for regions
            'r1': [(0, True)],
            'r2': [(0, False)],
            'r3': [(0, False)],
            'r4': [(0, False)],
        }
        self.take_env()
        return np.array(self.x)

    def render(self, mode='human', e=0, rewards=0):
        if mode == 'human':
            print("(input: %d , output: %d , epi_num: %d , rewards: %d )" % (self.x, self.action, e, rewards))
        else:
            super().render(mode=mode)  # just raise an exception