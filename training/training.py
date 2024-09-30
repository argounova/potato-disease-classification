import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 15

# Load the potato data into a tensorflow dataset
potato_ds = tf.keras.preprocessing.image_dataset_from_directory(
  'training/potato-data',
  shuffle=True,
  image_size=IMAGE_SIZE,
  batch_size=BATCH_SIZE
)

# Get the class names
class_names = potato_ds.class_names

# Split the dataset into training, validation, and test sets
def get_potato_ds_partitions(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
  ds_size = len(ds)
  if shuffle:
    ds = ds.shuffle(shuffle_size, seed=12)
  train_size = int(train_split * ds_size)
  val_size = int(val_split * ds_size)
  test_size = int(test_split * ds_size)
  return ds.take(train_size), ds.skip(train_size).take(val_size), ds.skip(train_size + val_size).take(test_size)

# Get the training, validation, and test sets
train_ds, val_ds, test_ds = get_potato_ds_partitions(potato_ds)

# Optimizing the dataset performance
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# Preprocess layers for the dataset
resize_and_rescale = keras.Sequential([
  layers.Resizing(256, 256, input_shape=(256, 256, 3)),
  layers.Rescaling(1.0 / 255)
])

data_augmentation = keras.Sequential([
  layers.RandomFlip('horizontal_and_vertical'),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.1),
  layers.RandomContrast(0.1)
])

# Create the model
model = keras.Sequential([
  resize_and_rescale,
  data_augmentation,
  layers.Conv2D(32, (3,3), activation='relu'),
  layers.MaxPooling2D((2,2)),
  layers.Conv2D(64, (3,3), activation='relu'),
  layers.MaxPooling2D((2,2)),
  layers.Conv2D(64, (3,3), activation='relu'),
  layers.MaxPooling2D((2,2)),
  layers.Conv2D(64, (3,3), activation='relu'),
  layers.MaxPooling2D((2,2)),
  layers.Flatten(),
  layers.Dense(64, activation='relu'),
  layers.Dense(3, activation='softmax')
])

# print(model.summary())

# Compile the model
model.compile(
  optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy']
)

# Train the model
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=EPOCHS,
  verbose=1,
  batch_size=BATCH_SIZE,
)

# Evaluate the model
# evaluation = model.evaluate(test_ds)
# print(evaluation)

# Plot the training and validation accuracy
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(range(EPOCHS), acc, label='Training Accuracy')
# plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
# plt.show()

# # Plot the training and validation loss
# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 2)
# plt.plot(range(EPOCHS), loss, label='Training Loss')
# plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()

# Batch prediction
# def predict(model, img):
#   img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
#   img_array = tf.expand_dims(img_array, 0)
#   predictions = model.predict(img_array)
#   predicted_class = class_names[np.argmax(predictions[0])]
#   confidence = round(100 * (np.max(predictions[0])), 2)
#   return predicted_class, confidence

# plt.figure(figsize=(15, 15))
# plt.tight_layout()

# for images, labels in test_ds.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"))
#     predicted_class, confidence = predict(model, images[i].numpy())
#     actual_class = class_names[labels[i]] 
#     plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")
#     plt.axis("off")
# plt.show()

# Save the model to the models directory
model.save('models/potato-model_15-epochs.keras') 