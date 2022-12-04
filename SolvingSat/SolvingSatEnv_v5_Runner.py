import numpy as np
import gym
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import mean_squared_error
from matplotlib import pyplot as plt
from SolvingSat.solving_sat_env_v5 import SolvingSatEnv_v5

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.n_actions = action_size
        # we define some parameters and hyperparameters:
        # "lr" : learning rate
        # "gamma": discounted factor
        # "exploration_proba_decay": decay of the exploration probability
        # "batch_size": size of experiences we sample to train the DNN
        self.lr = 0.001
        self.gamma = 0.99
        self.exploration_proba = 1.0
        self.exploration_proba_decay = 0.001
        self.batch_size = 32

        # We define our memory buffer where we will store our experiences
        # We stores only the 2000 last time steps
        self.memory_buffer = list()
        self.max_memory_buffer = 2000

        # We creaate our model having to hidden layers of 24 units (neurones)
        # The first layer has the same size as a state size
        # The last layer has the size of actions space
        self.model = Sequential([
            Dense(units=24, input_dim=state_size, activation='relu'),
            Dense(units=24, activation='relu'),
            Dense(units=action_size, activation='linear')
        ])
        self.model.compile(loss="mse",
                           optimizer=Adam(lr=self.lr))

    # The agent computes the action to perform given a state
    def compute_action(self, current_state):
        # We sample a variable uniformly over [0,1]
        # if the variable is less than the exploration probability
        #     we choose an action randomly
        # else
        #     we forward the state through the DNN and choose the action
        #     with the highest Q-value.
        if np.random.uniform(0, 1) < self.exploration_proba:
            return np.random.choice(range(self.n_actions))
        q_values = self.model.predict(current_state)[0]
        return np.argmax(q_values)

    # when an episode is finished, we update the exploration probability using
    # espilon greedy algorithm
    def update_exploration_probability(self):
        self.exploration_proba = self.exploration_proba * np.exp(-self.exploration_proba_decay)
        print(self.exploration_proba)

    # At each time step, we store the corresponding experience
    def store_episode(self, current_state, action, reward, next_state, done):
        # We use a dictionnary to store them
        self.memory_buffer.append({
            "current_state": current_state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done
        })
        # If the size of memory buffer exceeds its maximum, we remove the oldest experience
        if len(self.memory_buffer) > self.max_memory_buffer:
            self.memory_buffer.pop(0)

    # At the end of each episode, we train our model
    def train(self):
        # We shuffle the memory buffer and select a batch size of experiences
        np.random.shuffle(self.memory_buffer)
        batch_sample = self.memory_buffer[0:self.batch_size]

        # We iterate over the selected experiences
        for experience in batch_sample:
            # We compute the Q-values of S_t
            q_current_state = self.model.predict(experience["current_state"])
            # We compute the Q-target using Bellman optimality equation
            q_target = experience["reward"]
            if not experience["done"]:
                q_target = q_target + self.gamma * np.max(self.model.predict(experience["next_state"])[0])
            q_current_state[0][experience["action"]] = q_target
            # train the model
            self.model.fit(experience["current_state"], q_current_state, verbose=0)

env = SolvingSatEnv_v5()

# We get the shape of a state and the actions space size
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
# Number of episodes to run
n_episodes = 1000
# Max iterations per epiode
max_iteration_ep = 30
# We define our agent
agent = DQNAgent(state_size, action_size)
total_steps = 0

batch_size = 32

# We iterate over episodes
for e in range(n_episodes):
    rewards = 0
    current_state = env.reset()
    current_state = np.array([current_state])
    for step in range(max_iteration_ep):
        total_steps = total_steps + 1
        action = agent.compute_action(current_state)
        next_state, reward, done, _ = env.step(action)
        rewards += reward
        next_state = np.array([next_state])
        agent.store_episode(current_state, action, reward, next_state, done)
        # env.render(e=e, rewards=rewards)
        if done:
            agent.update_exploration_probability()
            break
        if step == max_iteration_ep - 1:
            agent.update_exploration_probability()
            break
        current_state = next_state
    if total_steps >= batch_size:
        agent.train()

def test():
    env = SolvingSatEnv_v5()
    solve = 0
    total_steps = 0
    max_episode = 100
    max_iteration_ep = 30
    for i in range(max_episode):
        rewards = 0
        steps = 0
        done = False
        state = env.reset()
        for j in range(max_iteration_ep):
            state = np.array([state])
            action = agent.compute_action(state)
            state, reward, done, _ = env.step(action)
            steps += 1
            rewards += reward
            env.render(e=steps, rewards=rewards)
            if done:
                total_steps += steps
                solve += 1
                break
    print("(total_steps: %d, avg_steps: %f)" % (total_steps, total_steps / solve))
    print("(solve: %d, total: %d, accuracy: %f )" % (solve, max_episode, solve / max_episode))
test()