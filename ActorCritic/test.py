import tensorflow as tf

model = tf.keras.models.load_model('./A2C.h5')
model.load_weights('./A2C.h5')
print(model.predict([0, 1, 2]))

