import numpy as np
from gym.spaces import MultiDiscrete, Discrete
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
                'property': '(G(((r2 | r4) & ~ckbby) -> ~Xckbby) & '
                            'G(((r6 | r7 | r8) & ckbby) -> Xckbby) & '
                            'G(~(r2 | r4 | r6 | r7 | r8) -> (Xckbby <-> ckbby)))',
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
                'property': '(G(r1 -> X(r1 | r2 | r4)) & '
                            'G(r2 -> X(r1 | r2 | r3 | r5)) & '
                            'G(r3 -> X(r2 | r3 | r4)) & '
                            'G(r4 -> X(r1 | r3 | r4)) & '
                            'G(r5 -> X(r2 | r5 | r6)) & '
                            'G(r6 -> X(r5 | r6 | r9)) & '
                            'G(r7 -> X(r7 | r9)) & '
                            'G(r8 -> X(r8 | r10)) & '
                            'G(r9 -> X(r6 | r7 | r9 | r10)) & '
                            'G(r10 -> X(r8 | r9 | r10)))',
                'quantitative': False
            },
            {
                'category': 'liveness',
                'property': '(GF(r2 | ~ckbby) & '
                            'GF(r4 | ~ckbby) & '
                            'GF(r6 | ckbby) & '
                            'GF(r7 | ckbby) & '
                            'GF(r8 | ckbby))',
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

        self.observation_space = Discrete(2)
        self.action_space = Discrete(10)
        self.observation = 1
        self.action = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def take_env(self):
        def compute_observation():
            obs = self.observation_space.sample()
            self.traces['ckbby'].append((len(self.traces['ckbby']), True if obs == 1 else False))

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
                self.traces['ckbby'].pop(len(self.traces['ckbby']) - 1)
                return False

        cnt = 1
        computed = compute_observation()
        while not computed:
            computed = compute_observation()
            cnt += 1
            if cnt == 10 and not computed:
                break
        self.traces['aux0'].append((len(self.traces['aux0']), computed))
        return computed

    def step(self, action):
        self.action = action
        self.traces['r1'].append((len(self.traces['r1']), True if action == 0 else False))
        self.traces['r2'].append((len(self.traces['r2']), True if action == 1 else False))
        self.traces['r3'].append((len(self.traces['r3']), True if action == 2 else False))
        self.traces['r4'].append((len(self.traces['r4']), True if action == 3 else False))
        self.traces['r5'].append((len(self.traces['r5']), True if action == 4 else False))
        self.traces['r6'].append((len(self.traces['r6']), True if action == 5 else False))
        self.traces['r7'].append((len(self.traces['r7']), True if action == 6 else False))
        self.traces['r8'].append((len(self.traces['r8']), True if action == 7 else False))
        self.traces['r9'].append((len(self.traces['r9']), True if action == 8 else False))
        self.traces['r10'].append((len(self.traces['r10']), True if action == 9 else False))

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
            'ckbby': [(0, True)],
            'r1': [(0, True)],
            'r2': [(0, False)],
            'r3': [(0, False)],
            'r4': [(0, False)],
            'r5': [(0, False)],
            'r6': [(0, False)],
            'r7': [(0, False)],
            'r8': [(0, False)],
            'r9': [(0, False)],
            'r10': [(0, False)],
            'aux0': [(0, True)]
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