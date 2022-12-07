import numpy as np
from gym.spaces import Discrete, Box
import gym
import pygame
import stl

class ActorCritic(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        super().__init__()

        self.size = size  # The size of the square grid
        self.window_size = 512 # The size of the PyGame window

        self.env_properties = [
            {
                'category': 'safety',
                'property': '(G(({d > -1} & {d < 1}) -> (F[1, 1]({d > -1} & {d < 1}) | F[1, 1]({d > 0} & {d < 2}))) & '
                            'G(({d > 0} & {d < 2}) -> (F[1, 1]({d > 0} & {d < 2}) | F[1, 1]({d > -1} & {d < 1}) | F[1, 1]({d > 1} & {d < 3}))) & '
                            'G(({d > 1} & {d < 3}) -> (F[1, 1]({d > 1} & {d < 3}) | F[1, 1]({d > 0} & {d < 2}))))',
                'quantitative': True
            },
            {
                'category': 'liveness',
                'property': '',
                'quantitative': True
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
                'property': '(G(F[1, 1]({d < 1} & {d > -1}) -> F[1, 1]{v < 10}))',
                'quantitative': True
            },
            {
                'category': 'liveness',
                'property': '(G(({d < 3} & {d > 1}) -> F[0, 2]{v > 120}) & G(({d < 2} & {d > 0}) -> F[0, 2]({v < 50} & {v > 30})))',
                'quantitative': True
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

        self.width = 3
        self.observation_space = Box(low=-1, high=3, shape=(1,), dtype=np.int32)
        self.action_space = Box(low=-1, high=150, shape=(1,), dtype=np.int32)

        self.x = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.action = None
        self.window = None
        self.clock = None

    def take_env(self):
        def compute_inputs():
            possible_inputs = []
            for v in range(self.width):
                if v == 0:
                    self.traces['d'].append((len(self.traces['d']), 0))
                elif v == 1:
                    self.traces['d'].append((len(self.traces['d']), 1))
                elif v == 2:
                    self.traces['d'].append((len(self.traces['d']), 2))
                safety_eval = True
                if len(self.env_properties[0]['property']) > 0:
                    phi = stl.parse(self.env_properties[0]['property'])
                    safety_eval = phi(self.traces) > 0 if True else False
                liveness_eval = True
                if len(self.env_properties[1]['property']) > 0:
                    phi = stl.parse(self.env_properties[1]['property'])
                    liveness_eval = phi(self.traces) > 0 if True else False
                if safety_eval and liveness_eval:
                    possible_inputs.append(v)
                self.traces['d'].pop(len(self.traces['d']) - 1)
            return possible_inputs

        possible_inputs = compute_inputs()
        selected_input = np.random.choice(possible_inputs)
        self.x = selected_input
        if selected_input == 0:
            self.traces['d'].append((len(self.traces['d']), 0))
        elif selected_input == 1:
            self.traces['d'].append((len(self.traces['d']), 1))
        elif selected_input == 2:
            self.traces['d'].append((len(self.traces['d']), 2))

    def step(self, action):
        self.action = action
        self.traces['v'].append((len(self.traces['v']), self.action))
        obs = np.array(self.x)

        info = {
            'satisfiable': False
        }

        reward = 0
        done = False
        safety_eval = True
        if len(self.sys_properties[0]['property']) > 0:
            phi = stl.parse(self.sys_properties[0]['property'])
            safety_eval = phi(self.traces) > 0 if True else False
        liveness_eval = True
        if len(self.sys_properties[1]['property']) > 0:
            phi = stl.parse(self.sys_properties[1]['property'])
            liveness_eval = phi(self.traces) > 0 if True else False
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
        self.render()
        return obs, reward, done, info

    def reset(self):
        self.traces = {
            # distance
            'd': [(0, 2)],
            # velocity
            'v': [(0, 130)],
        }

        self.take_env()

        if self.render_mode == "human":
            self._render_frame()

        return np.array(self.x)

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size / self.size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 0, 0))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * np.array([4, 0]),
                (pix_square_size, pix_square_size),
            ),
        )

        if self.x == 0:
            train = np.array([2, 0])
        elif self.x == 1:
            train = np.array([1, 0])
        elif self.x == 2:
            train = np.array([0, 0])

        color = 764 if self.action == None else self.action * 3
        color = divmod(color, 255)
        if color[0] == 0:
            color = (color[1], 0, 0)
        elif color[0] == 1:
            color = (0, color[1], 0)
        elif color[0] == 2:
            color = (0, 0, color[1])

        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            # (0, 0, 255),
            color,
            (train + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                (255, 255, 255),
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                (255, 255, 255),
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )