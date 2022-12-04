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

def bellman_update(Q: np.ndarray, learning_rate: float, discount_factor: float,
                   reward: float, state: int, newstate: int, action: int) -> np.ndarray:
    """
    Returns the updated Q table using the Bellman equation.
    """
    Q[state, action] = Q[state, action] + learning_rate*(
                                   reward + \
                                   discount_factor * np.max(Q[newstate,:]) - \
                                   Q[state, action]
                                 )
    return Q

def run_update_rule(env: GlassBridgeEnv(), Q: np.ndarray, learning_rate: float, discount_factor: float) -> np.ndarray:
    """
    Runs the Bellman update with random policy until the episode terminates.
    Returns the updated Q table at the end of the episode.
    """
    state = env.reset()
    done = False
    while (not done):
        action = env.action_space.sample() # random policy.
        newstate, reward, done, _ = env.step(action)
        Q = bellman_update(Q, learning_rate, discount_factor, reward, state, newstate, action)
        state = newstate
    return Q

num_states = env.observation_space.n
num_actions = env.action_space.n

def train_agent(env: GlassBridgeEnv(), episodes: int, learning_rate: float, discount_factor: float) -> np.ndarray:
    Q = np.zeros([num_states, num_actions])
    for episode in range(episodes):
        Q = run_update_rule(env, Q, learning_rate, discount_factor)
        print(Q)
        clear_output(wait = True)
    return Q

discount_factor = 0.95
learning_rate = 0.1
episodes = 2000

def normalize_Q(Q: np.ndarray) -> np.ndarray:
    Q_max = np.max(Q)
    if Q_max > 0.0: # if agent never succeeds, then Q_max = 0
        Q = (Q/Q_max)*1
    return Q

Q = train_agent(env, episodes, learning_rate, discount_factor)
print(normalize_Q(Q))

policy = 'random'   #@param ['greedy', 'random']
episodes = 1000

eps_decay = 0.965
episodes = 100

epsilon = 1.0
eps_values = np.zeros(episodes)

for episode in range(episodes):
    eps_values[episode] = epsilon
    epsilon *= eps_decay


def policy_eps_greedy(Q: np.ndarray, state: int, epsilon: float) -> int:
    """
    Returns the action calculated using epsilon greedy policy.
    """
    if np.random.random() < epsilon:
        action = env.action_space.sample()  # random policy.
    else:
        action = np.argmax(Q[state, :])  # greedy policy.
    return action


def run_epsilon_greedy_episode(env: GlassBridgeEnv(), Q: np.ndarray, epsilon: float,
                               learning_rate: float, discount_factor: float) -> tuple:
    """
    Returns tuple containing episode's return and Q table.
    """
    state = env.reset()
    done = False
    episode_return = 0
    while (not done):
        action = policy_eps_greedy(Q, state, epsilon)
        newstate, reward, done, _ = env.step(action)
        episode_return += reward
        Q = bellman_update(Q, learning_rate, discount_factor, reward, state, newstate, action)
        state = newstate
    return (episode_return, Q)


def train_agent(env: GlassBridgeEnv, epsiodes: int, learning_rate: float,
                discount_factor: float, eps_decay: float) -> tuple:
    """
    Trains agent using epsilon greedy policy. Returns array of rewards for each episode and Q table
    """
    reward_history = np.array([])
    Q = np.zeros([num_states, num_actions])
    epsilon = 1.0
    for episode in range(episodes):
        reward, Q = run_epsilon_greedy_episode(env, Q, epsilon, learning_rate, discount_factor)
        reward_history = np.append(reward_history, reward)
        if (epsilon > EPS_MIN):
            epsilon *= eps_decay
    return (reward_history, Q)


def check_success(env: GlassBridgeEnv, Q: np.ndarray):
    """
    Check success rate using learned Q table.
    """
    success = 0
    for episode in range(100):
        state = env.reset()
        done = False
        reward = 0
        while not done:
            action = np.argmax(Q[state, :])
            state, reward, done, _ = env.step(action)
        success += reward
    print(f"\nSuccess rate: {success} %")


def visualize_training(reward_history: np.array):
    """
    Plots reward and success % over episodes.
    """
    plt.subplot(2, 1, 1)
    plt.plot(range(len(reward_history)), reward_history)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Reward during Training')
    num_bins = episodes / 1000
    plt.subplot(2, 1, 2)
    plt.hist(np.nonzero(reward_history)[0], bins=int(num_bins), range=(0, episodes), rwidth=0.4)
    plt.xlabel('Episodes')
    plt.ylabel('# Success')

# Set parameters
eps_decay = 0.98
episodes = 20000
discount_factor = 0.95
learning_rate = 0.02

EPS_MIN = 0.05

# Run agent, print q-values, and plot reward history.
reward_history, Q = train_agent(env, episodes, learning_rate, discount_factor, eps_decay)
print(Q)
visualize_training(reward_history)
check_success(env, Q)

env.close()