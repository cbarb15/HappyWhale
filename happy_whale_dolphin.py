import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
import pathlib

batch_size = 32
image_height = 180
image_width = 180

data_dir = pathlib.Path('data/train_images')
image_list = list(data_dir.glob('*.jpg'))
image_count = len(image_list)
list_ds = tf.data.Dataset.list_files(str(data_dir/'*.jpg'), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

val_size = int(image_count * 0.2)
train_dataset = list_ds.skip(val_size)
val_dataset = list_ds.take(val_size)

train_labels_dataframe = pd.read_csv("data/train.csv")
labels_list = []

# def write_labels_to_csv():
#     with open('data/train_labels.csv', 'w') as f:
#         for i, file in enumerate(image_list[:4]):
#             row_label = train_labels_dataframe.loc[train_labels_dataframe['image'] == file.name]
#             species_and_id = row_label[['image', 'species', 'individual_id']].values
#             labels_list.append(species_and_id)
#             arr = species_and_id.flatten()
#             print(f'writing label {i} to file {arr[0], arr[1], arr[2]}')
#             f.write(f'{arr[0]}, {arr[1], arr[2]}\n')

def get_label(file_path):
    # jpg_name = tf.strings.split(file_path, '/')[-1].numpy().decode('utf-8')
    parts = tf.strings.split(file_path, '/')
    jpg_name = parts[-1]
    row_label = train_labels_dataframe.loc[train_labels_dataframe['image'] == jpg_name]
    return row_label[['individual_id']].to_numpy().flatten()


def decode_img(img):
    img = tf.io.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [image_height, image_width])

def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)

    return img, label

labels = []
images = []

counter = 0
for file in train_dataset.take(10):
    image, label = process_path(file)
    labels.append(label)
    images.append(image)

val_labels = []
val_images = []

for file in val_dataset.take(10):
    image, label = process_path(file)
    val_labels.append(label)
    val_images.append(image)

train_dataset_with_labels = tf.data.Dataset.from_tensor_slices((images, labels))
train_dataset_with_labels = train_dataset_with_labels.batch(batch_size)
val_dataset_withLabels = tf.data.Dataset.from_tensor_slices((val_images, val_labels))


AUTOTUNE = tf.data.AUTOTUNE
# train_dataset_with_labels = train_dataset_with_labels.cache(1000).prefetch(buffer_size=AUTOTUNE)
# val_dataset_withLabels = val_dataset_withLabels.cache(1000).prefetch(buffer_size=AUTOTUNE)

data_augmentation = Sequential(
  [
    # layers.RandomFlip("horizontal",
    #                   input_shape=(image_height,
    #                               image_width,
    #                               3)),
layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

model = Sequential([
  keras.Input(shape=(180, 180, 3)),
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(1, name="outputs")
])

model.summary()

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(train_dataset_with_labels, validation_data=val_dataset_withLabels, epochs=15)
#
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
#
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs_range = range(15)
#
# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
#
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()