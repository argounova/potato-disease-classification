import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 20

# Load the potato data into a tensorflow dataset
potato_ds = tf.keras.preprocessing.image_dataset_from_directory(
  'training/potato-data',
  shuffle=True,
  image_size=IMAGE_SIZE,
  batch_size=BATCH_SIZE
)

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
resize_and_rescale = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(IMAGE_SIZE[0], IMAGE_SIZE[1]),
  layers.experimental.preprocessing.Rescaling(1.0 / 255)
])

data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
  layers.experimental.preprocessing.RandomRotation(0.2),
])