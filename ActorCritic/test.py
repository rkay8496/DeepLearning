import tensorflow as tf

pb_path = '/Users/ryeonggukwon/PycharmProjects/DeepLearning/ActorCritic/NaiveActorCritic.h5'
model = tf.keras.models.load_model(pb_path)
model.load_weights(pb_path)
print(model.predict([0, 1, 2, 3]))
