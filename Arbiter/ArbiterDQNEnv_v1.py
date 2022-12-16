import pygame
import numpy as np
from gym.spaces import Discrete, MultiBinary, MultiDiscrete, Dict
import mtl
import gym
import stl

class ArbiterEnv():
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        super().__init__()

        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.observation_value = [['-', '-'], ['-', '+'], ['+', '-'], ['+', '+']]

        # p0, p1
        # closed, partially_open, open
        # for i in ['00', '01', '10']:
            # q0, q1
            # nothing, close, open
            # for j in ['00', '01', '10']:
                    # r0
                    # off, on
                    # for k in ['0', '1']:
                    #     value = []
                    #     value.append(i[0])
                    #     value.append(i[1])
                    #     value.append(j[0])
                    #     value.append(j[1])
                    #     value.append(k)
                    #     self.observation_value.append(value)

        # s0, s1
        # nothing, turn_off, turn_on
        self.action_value = [['-', '-'], ['-', '+'], ['+', '-'], ['+', '+']]

        self.env_properties = [
            {

                'category': 'safety',
                'property': '(G((p & ~r) -> F[1, 1] p) & '  # door
                            'G((q & ~s) -> F[1, 1] q) & '
                            'G((p & r & F[1, 1] ~r) -> F[1, 1] ~p) & '
                            'G((q & s & F[1, 1] ~s) -> F[1, 1] ~q))',
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
                'property': '(G(~(r & s)) & '
                            'G(F(p -> r)) & '
                            'G(F(q -> s)))',
                'quantitative': True
            },
            {
                'category': 'liveness',
                'property': '',
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

        self.observation_space = Discrete(4)
        self.action_space = Discrete(4)
        self.observation = 0
        self.action = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.action = None
        self.window = None
        self.clock = None

    def take_env(self):
        def compute_observation():
            obs = self.observation_space.sample()
            value = self.observation_value[obs]
            self.traces['p'].append((len(self.traces['p']), 1 if value[0] == '+' else 0))
            self.traces['q'].append((len(self.traces['q']), 1 if value[0] == '+' else 0))
            safety_eval = True
            if len(self.sys_properties[0]['property']) > 0:
                phi = stl.parse(self.sys_properties[0]['property'])
                safety_eval = True if phi(self.traces) >= 0 else False
            liveness_eval = True
            if len(self.sys_properties[1]['property']) > 0:
                phi = stl.parse(self.sys_properties[1]['property'])
                liveness_eval = True if phi(self.traces) >= 0 else False
            if safety_eval and liveness_eval:
                self.observation = obs
                return True
            else:
                self.traces['p'].pop(len(self.traces['p']) - 1)
                self.traces['q'].pop(len(self.traces['q']) - 1)
                return False

        cnt = 1
        computed = compute_observation()
        while not computed:
            computed = compute_observation()
            cnt += 1
            if cnt == 10 and not computed:
                break
        return computed

    def step(self, action):
        self.action = action
        value = self.action_value[self.action]
        self.traces['r'].append((len(self.traces['r']), 1 if value[0] == '+' else 0))
        self.traces['s'].append((len(self.traces['s']), 1 if value[0] == '+' else 0))

        obs = np.array(self.observation)
        info = {
            'satisfiable': False
        }
        reward = 0
        safety_eval = True
        if len(self.sys_properties[0]['property']) > 0:
            phi = stl.parse(self.sys_properties[0]['property'])
            safety_eval = True if phi(self.traces) >= 0 else False
        liveness_eval = True
        if len(self.sys_properties[1]['property']) > 0:
            phi = stl.parse(self.sys_properties[1]['property'])
            liveness_eval = True if phi(self.traces) >= 0 else False
        if safety_eval and liveness_eval:
            reward += 10
            done = False
            info['satisfiable'] = True
        elif safety_eval and not liveness_eval:
            reward += 1
            done = False
            info['satisfiable'] = False
        # elif not safety_eval and liveness_eval:
        #     reward += -500
        #     done = False
        #     info['satisfiable'] = False
        else:
            reward += -10
            done = True
            info['satisfiable'] = False
        return obs, reward, done, info

    def reset(self):
        self.traces = {
            'p': [],
            'q': [],
            'r': [],
            's': [],
        }
        self.observation = self.observation_space.sample()
        value = self.observation_value[self.observation]
        self.traces['p'].append((len(self.traces['p']), 1 if value[0] == '+' else 0))
        self.traces['q'].append((len(self.traces['q']), 1 if value[0] == '+' else 0))
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