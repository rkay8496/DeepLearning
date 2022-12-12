import numpy as np
import gym
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import mean_squared_error
from matplotlib import pyplot as plt

from DoorInterlockSystem.DoorInterlockSystemDQNEnv_v1 import DoorInterlockSystemEnv


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.n_actions = action_size
        # we define some parameters and hyperparameters:
        # "lr" : learning rate
        # "gamma": discounted factor
        # "exploration_proba_decay": decay of the exploration probability
        # "batch_size": size of experiences we sample to train the DNN
        self.lr = 0.0000000001
        self.gamma = 0.99
        self.exploration_proba = 1.0
        self.exploration_proba_decay = 0.001
        self.batch_size = 32

        # We define our memory buffer where we will store our experiences
        # We stores only the 2000 last time steps
        self.memory_buffer = list()
        self.max_memory_buffer = 2000

        # We create our model having to hidden layers of 24 units (neurones)
        # The first layer has the same size as a state size
        # The last layer has the size of actions space
        self.model = Sequential([
            Dense(units=state_size, input_dim=1, activation='relu', name='inputs'),
            Dense(units=1024, activation='relu'),
            Dense(units=action_size, activation='linear', name='outputs')
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
        q_values = self.model.predict(current_state)
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

env = DoorInterlockSystemEnv()
state_size = env.observation_space.n
action_size = env.action_space.n
episodes = 3000
iterations = 10
agent = DQNAgent(state_size, action_size)
total_steps = 0
batch_size = 32
ep_reward = []
total_avgr = []
solved = 0

for e in range(episodes):
    total_reward = 0
    current_state = env.reset()
    current_state = np.array([current_state])
    rewards = []
    for step in range(1, iterations):
        total_steps += 1
        action = agent.compute_action(current_state)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        rewards.append(reward)
        next_state = np.array([next_state])
        agent.store_episode(current_state, action, reward, next_state, done)
        current_state = next_state
        if done and info['satisfiable']:
            solved += 1
            ep_reward.append(total_reward)
            avg_reward = np.mean(ep_reward[-100:])
            total_avgr.append(avg_reward)
            agent.update_exploration_probability()
            print("total reward after {} episodes is {} and avg reward is {} and number of solved is {}".format(e,
                                                                                                                total_reward,
                                                                                                                avg_reward,
                                                                                                                solved))
            break
        elif done and not info['satisfiable']:
            ep_reward.append(total_reward)
            avg_reward = np.mean(ep_reward[-100:])
            total_avgr.append(avg_reward)
            agent.update_exploration_probability()
            print("total reward after {} episodes is {} and avg reward is {} and number of solved is {}".format(e,
                                                                                                                total_reward,
                                                                                                                avg_reward,
                                                                                                                solved))
            break
        else:
            env.take_env()
    if total_steps >= batch_size:
        agent.train()

agent.model.save('DoorInterlockSystem_v1.h5')

model = tf.keras.models.load_model('./DoorInterlockSystem_v1.h5')
print(model.predict([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                     10, 11, 12, 13, 14, 15, 16, 17]))

ep = [i for i in range(episodes)]
plt.plot(ep, total_avgr, 'b')
plt.title("avg reward Vs episodes")
plt.xlabel("episodes")
plt.ylabel("average reward per 100 episodes")
plt.grid(True)
plt.show()

ep_reward = []
total_avgr = []
solved = 0

for e in range(episodes):
    total_reward = 0
    current_state = env.reset()
    current_state = np.array([current_state])
    rewards = []
    for step in range(1, iterations):
        action = agent.compute_action(current_state)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        rewards.append(reward)
        next_state = np.array([next_state])
        current_state = next_state
        if done and info['satisfiable']:
            solved += 1
            ep_reward.append(total_reward)
            avg_reward = np.mean(ep_reward[-100:])
            total_avgr.append(avg_reward)
            print("total reward after {} episodes is {} and avg reward is {} and number of solved is {}".format(e,
                                                                                                                total_reward,
                                                                                                                avg_reward,
                                                                                                                solved))
            break
        elif done and not info['satisfiable']:
            ep_reward.append(total_reward)
            avg_reward = np.mean(ep_reward[-100:])
            total_avgr.append(avg_reward)
            print("total reward after {} episodes is {} and avg reward is {} and number of solved is {}".format(e,
                                                                                                                total_reward,
                                                                                                                avg_reward,
                                                                                                                solved))
            break
        else:
            env.take_env()

ep = [i for i in range(episodes)]
plt.plot(ep, total_avgr, 'b')
plt.title("avg reward Vs episodes")
plt.xlabel("episodes")
plt.ylabel("average reward per 100 episodes")
plt.grid(True)
plt.show()