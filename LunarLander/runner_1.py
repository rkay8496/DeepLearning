import gym
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.optimizers import Adam

# 하이퍼파라미터 설정
LEARNING_RATE = 0.0005
GAMMA = 0.98
N_STEP_RETURN = 8
N_EPISODES = 5000
N_MAX_STEP = 1000
BATCH_SIZE = 32
ENTROPY_BETA = 0.001

# A2C 모델 클래스 정의
class A2C:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.actor_lr = LEARNING_RATE
        self.critic_lr = LEARNING_RATE * 2
        self.gamma = GAMMA
        self.n_step_return = N_STEP_RETURN
        self.batch_size = BATCH_SIZE
        self.entropy_beta = ENTROPY_BETA

        # Actor Network 정의
        inputs = Input(shape=(self.state_dim,))
        fc1 = Dense(256, activation='relu')(inputs)
        fc2 = Dense(256, activation='relu')(fc1)
        mu = Dense(self.action_dim, activation='softmax')(fc2)
        sigma = Dense(self.action_dim, activation='softplus')(fc2)
        value = Dense(1)(fc2)

        def sample(args):
            mu, sigma = args
            return mu + tf.random.normal(tf.shape(mu)) * sigma

        actions = Lambda(sample)([mu, sigma])
        self.actor = Model(inputs=inputs, outputs=actions)
        actor_optimizer = Adam(lr=self.actor_lr)
        self.actor.compile(optimizer=actor_optimizer, loss='categorical_crossentropy')

        # Critic Network 정의
        self.critic = Model(inputs=inputs, outputs=value)
        critic_optimizer = Adam(lr=self.critic_lr)
        self.critic.compile(optimizer=critic_optimizer, loss='mse')

    # Advantage 계산 함수
    def get_advantages(self, rewards, values, next_values, dones):
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_advantage = 0
        last_return = 0

        for t in reversed(range(len(rewards))):
            if t + self.n_step_return < len(rewards):
                returns = rewards[t:t+self.n_step_return] + self.gamma * next_values[t+self.n_step_return] * (1-dones[t+self.n_step_return])
                last_return = returns[0]
                advantages[t] = last_advantage = last_return - values[t]
            else:
                returns = last_return * (1-dones[t+self.n_step_return]) + rewards[-1]
                last_advantage = last_return - values[-1]
                advantages[t] = last_advantage

            for r in reversed(returns[:-1]):
                last_advantage = last_advantage * self.gamma + r - values[t]
                advantages[t] = last_advantage
        return advantages

    # A2C 학습 함수
    def train(self):
        for episode in range(N_EPISODES):
            done = False
            state = self.env.reset()
            states, actions, rewards, values, dones = [], [], [], [], []
            total_reward = 0

            while not done:
                # Actor로부터 Action 샘플링
                action_probs = self.actor.predict(np.reshape(state[0], [1, self.state_dim]))
                action = np.random.choice(self.action_dim, p=np.squeeze(action_probs))

                # Environment에서 Action 실행
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward

                # Trajectory 저장
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)

                # Critic으로부터 Value 예측
                value = tf.nn.softmax(self.actor.predict(np.reshape(state, (self.state_dim,))))
                values.append(value)

                state = next_state

                if len(states) == self.batch_size or done:
                    # 마지막 상태의 Value 예측
                    if done:
                        next_value = 0
                    else:
                        next_value = self.critic.predict(np.reshape(next_state, [1, self.state_dim]))[0, 0]

                    # Advantages 계산
                    advantages = self.get_advantages(rewards, values,
                                                     np.append(next_value, np.zeros(self.n_step_return)), dones)

                    # 타겟값 계산
                    targets = np.zeros((len(states), self.action_dim))
                    for i, (advantage, action) in enumerate(zip(advantages, actions)):
                        targets[i, action] = advantage

                    # Actor와 Critic 업데이트
                    self.actor.fit(np.array(states), targets, batch_size=self.batch_size, verbose=0)
                    self.critic.fit(np.array(states), np.reshape(np.array(values), [len(values), 1]),
                                    batch_size=self.batch_size, verbose=0)

                    # Trajectory 초기화
                    states, actions, rewards, values, dones = [], [], [], [], []

            # 에피소드 정보 출력
            if episode % 10 == 0:
                print("Episode: {}, Score: {}".format(episode, total_reward))

# LunarLander-v2 환경 생성
env = gym.make("LunarLander-v2")

# A2C 모델 생성
model = A2C(env)

# A2C 학습
model.train()


