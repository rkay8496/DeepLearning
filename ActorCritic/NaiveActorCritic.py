import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from ActorCritic.NaiveActorCriticEnv_v1 import ActorCritic

env = ActorCritic()
state_size = env.observation_space.n
action_size = env.action_space.n

class critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(2048, activation='relu')
        self.d2 = tf.keras.layers.Dense(1536, activation='relu')
        self.v = tf.keras.layers.Dense(1, activation=None)

    def call(self, input_data):
        x = self.d1(input_data)
        x = self.d2(x)
        v = self.v(x)
        return v


class actor(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(state_size, input_shape=(1, ), activation='relu')
        self.d2 = tf.keras.layers.Dense(1536, activation='relu')
        self.a = tf.keras.layers.Dense(action_size, activation='softmax')
        self.model = tf.keras.models.Sequential([
            self.d1,
            self.d2,
            self.a
        ])

    def call(self, input_data):
        x = self.d1(input_data)
        x = self.d2(x)
        a = self.a(x)
        return a


class agent():
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.a_opt = tf.keras.optimizers.Adam(learning_rate=0.00000001)
        self.c_opt = tf.keras.optimizers.Adam(learning_rate=0.00000001)
        self.actor = actor()
        self.critic = critic()
        self.log_prob = None

    def act(self, state):
        prob = self.actor(np.array([state]))
        # print(prob)
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])
        # action = np.random.choice([i for i in range(env.action_space.n)], 1, p=prob[0])
        # log_prob = tf.math.log(prob[0][action]).numpy()
        # self.log_prob = log_prob[0]
        # #print(self.log_prob)
        # return action[0]

    def actor_loss(self, prob, action, td):
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob * td
        return loss

    def learn(self, state, action, reward, next_state, done):
        state = np.array([state])
        next_state = np.array([next_state])
        # self.gamma = tf.convert_to_tensor(0.99, dtype=tf.double)
        # d = 1 - done
        # d = tf.convert_to_tensor(d, dtype=tf.double)
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(state, training=True)

            # p = self.actor(state, training=True).numpy()[0][action]
            # p = tf.convert_to_tensor([[p]], dtype=tf.float32)
            # print(p)
            v = self.critic(state, training=True)
            # v = tf.dtypes.cast(v, tf.double)

            vn = self.critic(next_state, training=True)
            # vn = tf.dtypes.cast(vn, tf.double)
            td = reward + self.gamma * vn * (1 - int(done)) - v
            # print(td)
            # td = tf.math.subtract(tf.math.add(reward, tf.math.multiply(tf.math.multiply(self.gamma, vn), d)), v)
            # a_loss = -self.log_prob*td
            a_loss = self.actor_loss(p, action, td)
            # a_loss = -tf.math.multiply(tf.math.log(p),td)
            # a_loss = tf.keras.losses.categorical_crossentropy(td, p)
            # a_loss = -tf.math.multiply(self.log_prob,td)
            c_loss = td ** 2
            # c_loss = tf.math.pow(td,2)
        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return a_loss, c_loss


agentoo7 = agent()
n_episodes = 10000
n_iteration_ep = 10
total_steps = 0

for e in range(n_episodes):
    print('epi %d =======================' % (e + 1))
    rewards = 0
    state = env.reset()
    state = np.array([state])
    all_aloss = []
    all_closs = []
    for step in range(1, n_iteration_ep):
        total_steps = total_steps + 1
        action = agentoo7.act(state)
        next_state, reward, done, info = env.step(action)
        rewards += reward
        next_state = np.array([next_state])
        aloss, closs = agentoo7.learn(state, action, reward, next_state, done)
        if step < n_iteration_ep - 1:
            if not done and not info['satisfiable']:
                env.take_env()
            else:
                break
        else:
            break
        state = next_state

agentoo7.actor.model.save('NaiveActorCritic.h5')
