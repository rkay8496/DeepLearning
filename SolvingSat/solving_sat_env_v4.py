import pygame

from SolvingSat.actions import Actions
import numpy as np
from gym.spaces import Discrete, MultiDiscrete


class SolvingSatEnv_v4():
    metadata = {'render_modes': ['human']}

    def __init__(self, **kwargs):
        super().__init__()

        # dimensions of the grid
        self.width = kwargs.get('width', 4)
        self.height = kwargs.get('height', 1)

        # define the maximum x and y values
        self.max_x = self.width - 1
        self.max_y = self.height - 1

        # there are 5 possible actions: move N,E,S,W or stay in same state
        self.action_space = Discrete(2)

        # the observation will be the coordinates of Baby Robot
        # GF(p & q => r)
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
        # ''' apply the supplied action '''
        #
        # # move in the direction of the specified action
        # if action == Actions.North:
        #     self.y -= 1
        # elif action == Actions.South:
        #     self.y += 1
        # elif action == Actions.West:
        #     self.x -= 1
        # elif action == Actions.East:
        #     self.x += 1
        #
        # # make sure the move stays on the grid
        # if self.x < 0: self.x = 0
        # if self.y < 0: self.y = 0
        # if self.x > self.max_x: self.x = self.max_x
        # if self.y > self.max_y: self.y = self.max_y
        pass

    def step(self, action):
        self.action = action

        # take the action and update the position
        # self.take_action(action)
        obs = np.array([self.x, self.y])

        if self.x == 0 or self.x == 1 or self.x == 2:
            done = True
        elif self.x == 3 and action:
            done = True
        else:
            done = False

        reward = 1 if done else 0

        info = {}
        return obs, reward, done, info

    def reset(self):
        # reset Baby Robot's position in the grid
        self.x = np.random.choice(np.arange(self.width))
        self.y = 0
        return np.array([self.x, self.y])

    def render(self, mode='human', e=0, rewards=0):
        if mode == 'human':
            print("(input: %d , output: %d , epi_num: %d , rewards: %d )" % (self.x, self.action, e, rewards))
        else:
            super().render(mode=mode)  # just raise an exception