import random

import pygame

from SolvingSat.actions import Actions
import numpy as np
from gym.spaces import Discrete, MultiDiscrete
import mtl
import copy


class SolvingSatEnv_v7():
    metadata = {'render_modes': ['human']}

    def __init__(self, **kwargs):
        super().__init__()

        self.env_spec = "(G((p & ~r) -> Xp) & G((q & ~s) -> Xq))"
        self.sys_spec = "(G(~(r & s)) & GF(p -> r) & GF(q -> s))"
        self.spec = "((G((p & ~r) -> Xp) & G((q & ~s) -> Xq)) -> (G(~(r & s)) & GF(p -> r) & GF(q -> s)))"

        self.env_formula = {
            'G((p & ~r) -> Xp)': False,
            'G((q & ~s) -> Xq)': False
        }
        self.sys_formula = {
            'G(~(r & s))': False,
            'GF(p -> r)': False,
            'GF(q -> s)': False
        }

        self.traces = {
            'p': [(0, False)],
            'q': [(0, False)],
            'r': [(0, False)],
            's': [(0, False)]
        }

        # dimensions of the grid
        self.width = kwargs.get('width', 4)
        self.height = kwargs.get('height', 1)

        # define the maximum x and y values
        self.max_x = self.width - 1
        self.max_y = self.height - 1

        # there are 5 possible actions: move N,E,S,W or stay in same state
        self.action_space = Discrete(4)

        # the observation will be the coordinates of Baby Robot
        # 0: !p & !q
        # 1: p & !q
        # 2: !p & q
        # 3: p & q
        self.observation_space = MultiDiscrete([self.width, self.height])

        # Baby Robot's position in the grid
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
        self.x = np.random.choice(np.arange(self.width))
        self.y = 0

        possible_inputs = []
        for idx in range(self.width):
            if idx == 0:
                self.traces['p'].append((len(self.traces['p']), False))
                self.traces['q'].append((len(self.traces['q']), False))
            elif idx == 1:
                self.traces['p'].append((len(self.traces['p']), True))
                self.traces['q'].append((len(self.traces['q']), False))
            elif idx == 2:
                self.traces['p'].append((len(self.traces['p']), False))
                self.traces['q'].append((len(self.traces['q']), True))
            elif idx == 3:
                self.traces['p'].append((len(self.traces['p']), True))
                self.traces['q'].append((len(self.traces['q']), True))
            phi = mtl.parse(self.env_spec)
            evaluation = phi(self.traces, quantitative=False)
            if evaluation:
                possible_inputs.append(idx)
            self.traces['p'].pop(len(self.traces['p']) - 1)
            self.traces['q'].pop(len(self.traces['q']) - 1)

        self.x = random.choice(possible_inputs)

        if self.x == 0:
            self.traces['p'].append((len(self.traces['p']), False))
            self.traces['q'].append((len(self.traces['q']), False))
        elif self.x == 1:
            self.traces['p'].append((len(self.traces['p']), True))
            self.traces['q'].append((len(self.traces['q']), False))
        elif self.x == 2:
            self.traces['p'].append((len(self.traces['p']), False))
            self.traces['q'].append((len(self.traces['q']), True))
        elif self.x == 3:
            self.traces['p'].append((len(self.traces['p']), True))
            self.traces['q'].append((len(self.traces['q']), True))

    def step(self, action):
        self.action = action

        if action == 0:
            self.traces['r'].append((len(self.traces['r']), False))
            self.traces['s'].append((len(self.traces['s']), False))
        elif action == 1:
            self.traces['r'].append((len(self.traces['r']), True))
            self.traces['s'].append((len(self.traces['s']), False))
        elif action == 2:
            self.traces['r'].append((len(self.traces['r']), False))
            self.traces['s'].append((len(self.traces['s']), True))
        elif action == 3:
            self.traces['r'].append((len(self.traces['r']), True))
            self.traces['s'].append((len(self.traces['s']), True))

        obs = np.array([self.x, self.y])

        reward = 0
        done = False
        for key in self.sys_formula.keys():
            phi = mtl.parse(key)
            evaluation = phi(self.traces, quantitative=False)
            if evaluation:
                reward += 1
            else:
                reward += -10
            self.sys_formula[key] = evaluation
        count = len(dict(filter(lambda elem: elem[1] == True, self.sys_formula.items())).keys())
        if count == len(self.sys_formula.keys()):
            reward += 50
            done = True
        else:
            reward += -500
            done = False
        self.take_env(action=action)
        info = {}
        return obs, reward, done, info

    def reset(self):
        self.x = 0
        self.y = 0
        self.traces = {
            'p': [(0, False)],
            'q': [(0, False)],
            'r': [(0, False)],
            's': [(0, False)]
        }
        self.take_env(action=None)
        return np.array([self.x, self.y])

    def render(self, mode='human', e=0, rewards=0):
        if mode == 'human':
            print("(input: %d , output: %d , epi_num: %d , rewards: %d )" % (self.x, self.action, e, rewards))
        else:
            super().render(mode=mode)  # just raise an exception