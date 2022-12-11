import pygame
import numpy as np
from gym.spaces import Discrete, MultiBinary, MultiDiscrete, Dict, Text
import mtl
import gym

class DoorInterlockSystemEnv():
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        super().__init__()

        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.env_bits = []
        for l in ['00', '01', '10']:
            for k in ['0', '1']:
                for j in ['00', '01', '10']:
                    for i in ['00', '01', '10']:
                        self.env_bits.append(l + k + j + i)

        self.env_properties = [
            {
                'category': 'safety',
                'property': '((G((((~p0 & ~p1) & (~q0 & ~q1)) -> (X~p0 & X~p1)))) & '
                            '(G((((p0 & ~p1) & (~q0 & ~q1)) -> (Xp0 & X~p1)))) & '
                            '(G((((~p0 & ~p1) & (q0 & ~q1)) -> (X~p0 & Xp1)))) & '
                            '(G((((p0 & ~p1) & (~q0 & q1)) -> (X~p0 & Xp1)))) & '
                            '(G((((~p0 & p1) & (q0 & ~q1)) -> (Xp0 & X~p1)))) & '
                            '(G((((~p0 & p1) & (~q0 & q1)) -> (X~p0 & X~p1)))) & '
                            '(G((~p0 & ~p1) -> (~(X~q0 & Xq1)))) & '
                            '(G((p0 & ~p1) -> (~(Xq0 & X~q1)))) & '
                            '(G(((q0 & ~q1) & (~(Xp0 & X~p1))) -> (Xq0 & X~q1))) & '
                            '(G(((~q0 & q1) & (~(X~p0 & X~p1))) -> (X~q0 & Xq1))) & '
                            '(G((~s0 & s1) -> (X~r0))) & '
                            '(G((s0 & ~s1) -> (Xr0))) & '
                            '(G((~s0 & ~s1) -> (Xr0 <-> r0))))',
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
                'property': '((G((Xq0 & X~q1) -> (X~s0 & Xs1))) & '
                            '(G(((X~p0 & X~p1) & (X~q0 & Xq1)) -> (Xs0 & X~s1))))',
                'quantitative': False
            },
            {
                'category': 'liveness',
                'property': '',
                'quantitative': False
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

        # [p0, p1, q0, q1, r0]
        self.observation_space = Discrete(18)
        # [s0, s1]
        self.action_space = Discrete(3)
        self.observation = 9
        self.action = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.action = None
        self.window = None
        self.clock = None

    def take_env(self):
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
                safety_eval = True
                if len(self.sys_properties[0]['property']) > 0:
                    phi = mtl.parse(self.sys_properties[0]['property'])
                    safety_eval = phi(self.traces, quantitative=False)
                liveness_eval = True
                if len(self.sys_properties[1]['property']) > 0:
                    phi = mtl.parse(self.sys_properties[1]['property'])
                    liveness_eval = phi(self.traces, quantitative=False)
                if safety_eval and liveness_eval:
                    possible_inputs.append(bits)
                self.traces['p0'].pop(len(self.traces['p0']) - 1)
                self.traces['p1'].pop(len(self.traces['p1']) - 1)
                self.traces['q0'].pop(len(self.traces['q0']) - 1)
                self.traces['q1'].pop(len(self.traces['q1']) - 1)
                self.traces['r0'].pop(len(self.traces['r0']) - 1)
            return possible_inputs

        possible_inputs = compute_inputs()
        selected_input = np.random.choice(possible_inputs)
        self.observation = self.env_bits.index(selected_input)
        if selected_input[4] == '1':
            self.traces['p0'].append((len(self.traces['p0']), True))
        else:
            self.traces['p0'].append((len(self.traces['p0']), False))
        if selected_input[3] == '1':
            self.traces['p1'].append((len(self.traces['p1']), True))
        else:
            self.traces['p1'].append((len(self.traces['p1']), False))
        if selected_input[2] == '1':
            self.traces['q0'].append((len(self.traces['q0']), True))
        else:
            self.traces['q0'].append((len(self.traces['q0']), False))
        if selected_input[1] == '1':
            self.traces['q1'].append((len(self.traces['q1']), True))
        else:
            self.traces['q1'].append((len(self.traces['q1']), False))
        if selected_input[0] == '1':
            self.traces['r0'].append((len(self.traces['r0']), True))
        else:
            self.traces['r0'].append((len(self.traces['r0']), False))

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
        obs = np.array(self.observation)
        info = {
            'satisfiable': False
        }
        reward = 0
        safety_eval = True
        if len(self.sys_properties[0]['property']) > 0:
            phi = mtl.parse(self.sys_properties[0]['property'])
            safety_eval = phi(self.traces, quantitative=False)
        liveness_eval = True
        if len(self.sys_properties[1]['property']) > 0:
            phi = mtl.parse(self.sys_properties[1]['property'])
            liveness_eval = phi(self.traces, quantitative=False)
        if safety_eval and liveness_eval:
            reward += 100
            done = True
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
            reward += -1
            done = True
            info['satisfiable'] = False
        return obs, reward, done, info

    def reset(self):
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
        self.take_env()
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