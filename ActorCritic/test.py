import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

model = tf.keras.models.load_model('./A2C.h5')
model.load_weights('./A2C.h5')

episodes = 250
iterations = 20
for epi in range(episodes):
    state = np.random.choice([0, 1, 2])
    state = np.array([state])
    for step in range(1, iterations):
        prob = model.predict(state)
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        print(int(action.numpy()[0]))
