import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pathlib

batch_size = 32
image_height = 180
image_width = 180

data_dir = pathlib.Path('data/train_images')
image_count = len(list(data_dir.glob('*.jpg')))
list_ds = tf.data.Dataset.list_files(str(data_dir/'*.jpg'), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

val_size = int(image_count * 0.2)
train_dataset = list_ds.skip(val_size)
val_dataset = list_ds.take(val_size)

# def get_label(file_path):


def decode_img(img):
    img = tf.io.decode_jpeg(img, channels=3)
    return tf.image.resize(img, [image_height, image_width])

def process_path(file_path):
    print(file_path)
    # label = get_label from train.csv
    img = tf.io.read_file(file_path)
    img = decode_img(img)

    return img

# for path in train_dataset.take(1):
#     process_path(path)

train_dataset = train_dataset.take(1).map(process_path, num_parallel_calls =tf.data.AUTOTUNE)
