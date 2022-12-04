import sys
import gym
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import clear_output
from io import StringIO
from contextlib import closing
from tqdm import tqdm
from gym import utils
from gym.envs.toy_text import discrete

np.set_printoptions(precision=2, suppress=True)

RIGHT = 0
DIAGONAL = 1

MAP = {
    "8x2": ["STRRTRTR","RRTTRTRG"]
}

class GlassBridgeEnv(discrete.DiscreteEnv):
    """
    Squid Game episode 7: crossing the glass bridge.

    --> _|_|_|_|_|_|_|_|
        _|_|_|_|_|_|_|_| -->

    The surface is described using a grid like the following
        STRRTRTRRTRRR
        RRTTRTRTTRTTG
    S : starting point, safe
    T : tempered glass, safe
    R : regular glass, fall to your doom
    G : goal, where you will be ending your game
    The episode ends when you reach the goal or step on a regular glass tile.
    You receive a reward of 1 if you reach the goal, and zero otherwise.
    """

    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, desc=None, map_name="8x2", is_slippery=True):
        # if desc is None and map_name is None:
        #     desc = generate_random_map()
        # elif desc is None:
        #     desc = MAP[map_name]
        desc = MAP[map_name]
        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        nA = 2 # Number of possible actions.
        nS = nrow * ncol # Number of possible states.

        # Initial state distribution.
        isd = np.array(desc == b"S").astype("float64").ravel()
        isd /= isd.sum()

        P = {s: {a: [] for a in range(nA)} for s in range(nS)} # Transitions.

        def to_s(row: int, col: int) -> int:
            """
            to state (row,col).

            Returns the new state.
            """
            return row * ncol + col

        def inc(row: int, col: int, action: int) -> tuple:
            """
            Increment.

            Returns the new (row, col), after an action is taken.
            """
            if action == RIGHT:
                col = min(col + 1, ncol - 1)
            elif action == DIAGONAL:
                col = min(col + 1, ncol - 1)
                if row == 0:
                    row = 1
                elif row == 1:
                    row = 0
            return (row, col)

        def update_probability_matrix(row: int, col: int, action: int) -> list:
            """
            Returns newstate, reward and a flag indicating whether episode is finished, following action.
            """
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]
            done = bytes(newletter) in b"GR"
            reward = float(newletter == b"G")
            return newstate, reward, done

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(2):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b"GR":
                        li.append((1.0, s, 0, True)) # (probability, nextstate, reward, done)
                    else:
                        if is_slippery:
                            li.append((0.8, *update_probability_matrix(row, col, a)))
                            # Small proabability that the environemnt returns the other action
                            li.append((0.2, *update_probability_matrix(row, col, (a+3)%2)))
                        else:
                            li.append((1.0, *update_probability_matrix(row, col, a)))
        print(f"li: {li}")
        super(GlassBridgeEnv, self).__init__(nS, nA, P, isd) # Calls the parent class discrete.DiscreteEnv

    def render(self, mode="human"):
        outfile = StringIO() if mode == "ansi" else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(
                "  ({})\n".format(["Right", "Diagonal"][self.lastaction])
            )
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        if mode != "human":
            with closing(outfile):
                return outfile.getvalue()

env = GlassBridgeEnv()
for i_episode in range(20):
    newstate = env.reset()
    for t in range(10):
        env.render()
        print(newstate)
        action = env.action_space.sample()
        newstate, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break

env.close()