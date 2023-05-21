import numpy as np
from gym.spaces import Discrete, MultiDiscrete
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
                'property': '(G(off -> X~power) & '
                            'G(on -> Xpower))',
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
                'property': '',
                'quantitative': False
            },
            {
                'category': 'liveness',
                'property': '(G(closed -> Fon) & '
                            'G(~closed -> F[0, 1]off))',
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

        self.observation_space = MultiDiscrete([3, 2])
        self.action_space = Discrete(2)
        self.observation = [0, 1]
        self.action = 1

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def take_env(self):
        def compute_observation():
            obs = self.observation_space.sample()
            self.traces['closed'].append((len(self.traces['closed']), True if obs[0] == 0 else False))
            self.traces['partially'].append((len(self.traces['partially']), True if obs[0] == 1 else False))
            self.traces['open'].append((len(self.traces['open']), True if obs[0] == 2 else False))
            self.traces['power'].append((len(self.traces['power']), True if obs[1] == 1 else False))

            safety_eval = True
            if len(self.env_properties[0]['property']) > 0:
                phi = stl.parse(self.env_properties[0]['property'])
                safety_eval = phi(self.traces, quantitative=self.env_properties[0]['quantitative'])
            liveness_eval = True
            if len(self.env_properties[1]['property']) > 0:
                phi = stl.parse(self.env_properties[1]['property'])
                liveness_eval = phi(self.traces, quantitative=self.env_properties[1]['quantitative'])
            if safety_eval and liveness_eval:
                self.observation = obs
                return True
            else:
                self.traces['closed'].pop(len(self.traces['closed']) - 1)
                self.traces['open'].pop(len(self.traces['open']) - 1)
                self.traces['partially'].pop(len(self.traces['partially']) - 1)
                self.traces['power'].pop(len(self.traces['power']) - 1)
                return False

        cnt = 1
        computed = compute_observation()
        while not computed:
            computed = compute_observation()
            cnt += 1
            if cnt == 100 and not computed:
                break
        self.traces['aux0'].append((len(self.traces['aux0']), computed))
        return computed

    def step(self, action):
        self.action = action
        self.traces['off'].append((len(self.traces['off']), True if self.action == 0 else False))
        self.traces['on'].append((len(self.traces['on']), True if self.action == 1 else False))

        obs = np.array(self.observation)

        done = False
        info = {
            'satisfiable': False
        }
        reward = 0
        safety_eval = True
        if len(self.sys_properties[0]['property']) > 0:
            phi = stl.parse(self.sys_properties[0]['property'])
            safety_eval = phi(self.traces, quantitative=self.sys_properties[0]['quantitative'])
        liveness_eval = True
        if len(self.sys_properties[1]['property']) > 0:
            phi = stl.parse(self.sys_properties[1]['property'])
            liveness_eval = phi(self.traces, quantitative=self.sys_properties[1]['quantitative'])
        if safety_eval and liveness_eval:
            reward += 1
            done = False
            info['satisfiable'] = True
        elif safety_eval and not liveness_eval:
            reward += 1
            done = False
            info['satisfiable'] = False
        elif not safety_eval and liveness_eval:
            done = True
            info['satisfiable'] = False
        elif not safety_eval and not liveness_eval:
            done = True
            info['satisfiable'] = False
        return obs, reward, done, info

    def reset(self):
        self.traces = {
            'closed': [(0, True)],
            'partially': [(0, False)],
            'open': [(0, False)],
            'power': [(0, True)],
            # action
            'off': [(0, False)],
            'on': [(0, True)],
            'aux0': [(0, True)],
        }

        return np.array(self.observation)

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
        canvas.fill((255, 255, 255))
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

        diff = 0 if self.action == None else self.action

        if self.x == 0:
            train = np.array([2, 0])
            color = (255 - (diff * 1.3), 255 - (diff * 1.3), 255 - (diff * 1.3))
        elif self.x == 1:
            train = np.array([1, 0])
            color = (255 - (diff * 1.3), 255 - (diff * 1.3), 255 - (diff * 1.3))
        elif self.x == 2:
            train = np.array([0, 0])
            color = (255 - (diff * 1.3), 255 - (diff * 1.3), 255 - (diff * 1.3))

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
                (0, 0, 0),
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                (0, 0, 0),
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

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()