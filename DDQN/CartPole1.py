import numpy as np
import gym
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import mean_squared_error
from matplotlib import pyplot as plt
from keras.models import load_model


class DDQNAgent:
    # constructor: we need both the state size and the number of actions that the agent can take
    #              to initialize the DNN
    def __init__(self, state_size, action_size):
        # we define some hyperparameters
        self.n_actions = action_size
        self.lr = 0.001
        self.gamma = 0.99
        self.exploration_proba = 1.0
        self.exploration_proba_decay = 0.005
        self.memory_buffer = list()
        self.max_memory_buffer = 2000
        self.q_model = self.build_model(state_size, action_size)
        self.q_target_model = self.build_model(state_size, action_size)

    # building a model of 2 hidden later of 24 units each
    def build_model(self, state_size, action_size):
        model = Sequential([
            Dense(units=24, input_dim=state_size, activation='relu'),
            Dense(units=24, activation='relu'),
            Dense(units=action_size, activation='linear')
        ])
        model.compile(loss="mse", optimizer=Adam(lr=self.lr))
        return model

    # the agent computes the action to take given a state
    def compute_action(self, current_state):
        if np.random.uniform(0, 1) < self.exploration_proba:
            return np.random.choice(range(self.n_actions))
        q_values = self.q_model.predict(current_state)[0]
        return np.argmax(q_values)

    # we sotre all experiences
    def store_episode(self, current_state, action, reward, next_state, done):
        self.memory_buffer.append({
            "current_state": current_state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done
        })
        if len(self.memory_buffer) > self.max_memory_buffer:
            self.memory_buffer.pop(0)

    # when an episode is finished, we update the exploration probability
    def update_exploration_probability(self):
        self.exploration_proba = self.exploration_proba * np.exp(-self.exploration_proba_decay)
        print(self.exploration_proba)

    # train the model using the replayed memory
    def train(self, batch_size):
        np.random.shuffle(self.memory_buffer)
        batch_sample = self.memory_buffer[0:batch_size]

        for experience in batch_sample:
            q_current_state = self.q_model.predict(experience["current_state"])
            q_target = experience["reward"]
            if not experience["done"]:
                q_target = q_target + self.gamma * np.max(self.q_target_model.predict(experience["next_state"])[0])
            q_current_state[0][experience["action"]] = q_target
            self.q_model.fit(experience["current_state"], q_current_state, verbose=0)

    # we update the weights of the Q-target DNN
    def update_q_target_network(self):
        self.q_target_model.set_weights(self.q_model.get_weights())

env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
n_episodes = 500
max_iteration_ep = 500
batch_size = 64
q_target_update_freq = 10

agent = DDQNAgent(state_size, action_size)
total_steps = 0
n_training = 0
for e in range(n_episodes):
    current_state = env.reset()
    current_state = np.array([current_state])
    rewards = 0
    for step in range(max_iteration_ep):
        total_steps = total_steps + 1
        action = agent.compute_action(current_state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.array([next_state])
        rewards = rewards + reward
        agent.store_episode(current_state, action, reward, next_state, done)

        if done:
            agent.update_exploration_probability()
            break
        current_state = next_state

    print("episode ", e + 1, " rewards: ", rewards)
    if total_steps >= batch_size:
        agent.train(batch_size=batch_size)
        n_training = n_training + 1
        if n_training % q_target_update_freq:
            agent.update_q_target_network()