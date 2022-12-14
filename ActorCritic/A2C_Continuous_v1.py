import numpy as np
import tensorflow as tf
import gym
import tensorflow_probability as tfp
import keras.losses as kls
import matplotlib.pyplot as plt
from ActorCritic.A2CSafetyTrainEnv_Continuous_v1 import ActorCritic

env = ActorCritic(render_mode='human')
# state_size = env.observation_space.n
# action_size = env.action_space.n

class critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
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
        self.d1 = tf.keras.layers.Dense(5, input_shape=(1,), activation='relu')
        self.d2 = tf.keras.layers.Dense(1536, activation='relu')
        self.a = tf.keras.layers.Dense(152, activation='softmax')
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
        self.a_opt = tf.keras.optimizers.RMSprop(learning_rate=0.000001)
        self.c_opt = tf.keras.optimizers.RMSprop(learning_rate=0.000001)
        self.actor = actor()
        self.critic = critic()

    def act(self, state):
        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def actor_loss(self, probs, actions, td):

        probability = []
        log_probability = []
        for pb, a in zip(probs, actions):
            dist = tfp.distributions.Categorical(probs=pb, dtype=tf.float32)
            log_prob = dist.log_prob(a)
            prob = dist.prob(a)
            probability.append(prob)
            log_probability.append(log_prob)

        # print(probability)
        # print(log_probability)

        p_loss = []
        e_loss = []
        td = td.numpy()
        # print(td)
        for pb, t, lpb in zip(probability, td, log_probability):
            t = tf.constant(t)
            policy_loss = tf.math.multiply(lpb, t)
            entropy_loss = tf.math.negative(tf.math.multiply(pb, lpb))
            p_loss.append(policy_loss)
            e_loss.append(entropy_loss)
        p_loss = tf.stack(p_loss)
        e_loss = tf.stack(e_loss)
        p_loss = tf.reduce_mean(p_loss)
        e_loss = tf.reduce_mean(e_loss)
        # print(p_loss)
        # print(e_loss)
        loss = -p_loss - 0.0001 * e_loss
        # print(loss)
        return loss

    def learn(self, states, actions, discnt_rewards):
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(states, training=True)
            v = self.critic(states, training=True)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discnt_rewards, v)
            # print(discnt_rewards)
            # print(v)
            # print(td.numpy())
            a_loss = self.actor_loss(p, actions, td)
            c_loss = 0.5 * kls.mean_squared_error(discnt_rewards, v)
        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return a_loss, c_loss


def preprocess1(states, actions, rewards, gamma):
    discnt_rewards = []
    sum_reward = 0
    rewards.reverse()
    for r in rewards:
        sum_reward = r + gamma * sum_reward
        discnt_rewards.append(sum_reward)
    discnt_rewards.reverse()
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int32)
    discnt_rewards = np.array(discnt_rewards, dtype=np.float32)

    return states, actions, discnt_rewards


tf.random.set_seed(336699)
agentoo7 = agent()
episodes = 10000
ep_reward = []
total_avgr = []
solved = 0
for epi in range(episodes):
    done = False
    info = {
        'satisfiable': False
    }
    state = env.reset()
    state = np.array([state])
    total_reward = 0
    all_aloss = []
    all_closs = []
    rewards = []
    states = []
    actions = []

    while not done and not info['satisfiable']:
        action = agentoo7.act(state)
        next_state, reward, done, info = env.step(action)
        rewards.append(reward)
        states.append(state)
        # actions.append(tf.one_hot(action, 2, dtype=tf.int32).numpy().tolist())
        actions.append(action)
        next_state = np.array([next_state])
        state = next_state
        total_reward += reward

        if done and info['satisfiable']:
            solved += 1
            ep_reward.append(total_reward)
            avg_reward = np.mean(ep_reward[-100:])
            total_avgr.append(avg_reward)
            print("total reward after {} episodes is {} and avg reward is {} and number of solved is {}".format(epi, total_reward, avg_reward, solved))
            states, actions, discnt_rewards = preprocess1(states, actions, rewards, 1)
            al, cl = agentoo7.learn(states, actions, discnt_rewards)
            print(f"al{al}")
            print(f"cl{cl}")
            break
        elif done and not info['satisfiable']:
            ep_reward.append(total_reward)
            avg_reward = np.mean(ep_reward[-100:])
            total_avgr.append(avg_reward)
            print("total reward after {} episodes is {} and avg reward is {} and number of solved is {}".format(epi, total_reward, avg_reward, solved))
            break
        else:
            env.take_env()

agentoo7.actor.model.save('./A2C.h5')

model = tf.keras.models.load_model('./A2C.h5')
model.load_weights('./A2C.h5')
print(model.predict([0, 1, 2]))

ep = [i for i in range(episodes)]
plt.plot(ep, total_avgr, 'b')
plt.title("avg reward Vs episodes")
plt.xlabel("episodes")
plt.ylabel("average reward per 100 episodes")
plt.grid(True)
plt.show()

def test():
    env = ActorCritic(render_mode='human')
    ep_reward = []
    total_avgr = []
    solved = 0
    for epi in range(episodes):
        done = False
        info = {
            'satisfiable': False
        }
        state = env.reset()
        state = np.array([state])
        total_reward = 0
        rewards = []
        states = []
        actions = []
        env.render()

        while not done and not info['satisfiable']:
            state = np.array([state])
            action = agentoo7.act(state)
            next_state, reward, done, info = env.step(action)
            rewards.append(reward)
            states.append(state)
            actions.append(action)
            next_state = np.array([next_state])
            state = next_state
            total_reward += reward
            env.render()

            if done and info['satisfiable']:
                solved += 1
                ep_reward.append(total_reward)
                avg_reward = np.mean(ep_reward[-100:])
                total_avgr.append(avg_reward)
                print("total reward after {} episodes is {} and avg reward is {} and number of solved is {}".format(epi, total_reward, avg_reward, solved))
                break
            elif done and not info['satisfiable']:
                ep_reward.append(total_reward)
                avg_reward = np.mean(ep_reward[-100:])
                total_avgr.append(avg_reward)
                print("total reward after {} episodes is {} and avg reward is {} and number of solved is {}".format(epi, total_reward, avg_reward, solved))
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

    env.close()

test()

