import tensorflow as tf

pb_path = '/Users/ryeonggukwon/PycharmProjects/DeepLearning/SafetyTrain/SafetyTrainModel_v1.h5'
model = tf.keras.models.load_model(pb_path)
model.load_weights(pb_path)
print(model.predict([0, 1, 2]))

# tf.saved_model.save(model, "PATH TO SAVE MODEL")