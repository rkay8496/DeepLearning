import pygame

from SolvingSat.actions import Actions
import numpy as np
from gym.spaces import Discrete, MultiDiscrete
import mtl


class SolvingSatEnv_v5():
    metadata = {'render_modes': ['human']}

    def __init__(self, **kwargs):
        super().__init__()

        # self.spec = "(G(X(p | q) -> X(r)) & G(F(p)) & G(F(q)))"
        # self.spec = "G((p & q) -> X(r))"
        self.spec = "(G(p -> Xr) & G(q -> Xs) & G(~(r & s)))"
        # self.spec = "(GF(p -> r) & GF(q -> s) & G(~(r & s)))"

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

    def take_action(self, action):
        self.x = np.random.choice(np.arange(self.width))
        self.y = 0

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

        done = False
        phi = mtl.parse(self.spec)
        evaluation = phi(self.traces, quantitative=False)
        if evaluation:
            reward = 50
            done = True
        else:
            reward = 0
            done = False

        self.take_action(action=action)

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

        return np.array([self.x, self.y])

    def render(self, mode='human', e=0, rewards=0):
        if mode == 'human':
            print("(input: %d , output: %d , epi_num: %d , rewards: %d )" % (self.x, self.action, e, rewards))
        else:
            super().render(mode=mode)  # just raise an exception