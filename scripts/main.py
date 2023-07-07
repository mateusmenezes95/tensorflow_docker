from __future__ import absolute_import, division, print_function, unicode_literals

import sys

import cv2
import os
from cv2 import norm
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import matplotlib.pyplot as plt

from IPython.display import clear_output

#include path to pyton path
# sys.path.append("/home/tensorflow/python_ws/marine-debris-fls-datasets/md_fls_dataset")

# import models.resnet as resnet

dataset_basepath = "/home/tensorflow/python_ws/marine-debris-fls-datasets/md_fls_dataset/data/watertank-segmentation/"
watertank_original_images_path = dataset_basepath + "Images"
watertank_mask_images_path = dataset_basepath + "Masks"

TRAIN_LENGTH = 1000
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
OUTPUT_CHANNELS = 12
SPLIT_SEED = 42

def normalize(input_image):
  input_image = input_image / 255.0
  return input_image

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

def get_images(base_folder_path, image_type="original"):
    if not os.path.exists(base_folder_path):
        print(f"The folder '{base_folder_path}' does not exist.")
        return

    if image_type == "original":
        image_folder_path = watertank_original_images_path
    elif image_type == "mask":
        image_folder_path = watertank_mask_images_path
    else:
        raise Exception(f"Invalid image type '{image_type}'.")

    images_vec = []
    # Get a list of all files in the folder
    files = os.listdir(image_folder_path)

    # Filter and print the filenames of image files
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
    image_files = [file for file in files if os.path.splitext(file)[1].lower() in image_extensions]

    if len(image_files) == 0:
        print(f"No image files found in the folder '{base_folder_path}'.")
    else:
        print(str(len(image_files)) +  " images found:")
        for image_file in image_files:
            image_file_path = os.path.join(image_folder_path, image_file)
            if image_type == "original":
                image = cv2.imread(image_file_path)
                image = cv2.resize(image, dsize=(128, 128))
            if image_type == "mask":
                image = cv2.imread(image_file_path, flags=cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
            images_vec.append(image)

    return np.array(images_vec)

def plot_images(images):
  plt.figure(figsize=(128,128))
  for i in range(25):
      plt.subplot(5,5,i+1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)
      plt.imshow(images[i])
  plt.show()

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(display_list[i])
    plt.axis('off')
  plt.show()

def unet_model(output_channels, down_stack):
    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]

    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2,
        padding='same', activation='softmax')  #64x64 -> 128x128

    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    x = inputs

    # Downsampling através do modelo
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling e estabelecimento das conexões de salto
    for up, skip in zip(up_stack, skips):
      x = up(x)
      concat = tf.keras.layers.Concatenate()
      x = concat([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(image, mask, model):
    display([image, mask,
                create_mask(model.predict(image[tf.newaxis, ...]))])

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

def main():
    original_images = get_images(dataset_basepath, "original")
    mask_images = get_images(dataset_basepath, "mask")

    x_train, x_test, y_train, y_test = train_test_split(original_images, mask_images,
                                                        shuffle=True,
                                                        test_size=0.3,
                                                        random_state=SPLIT_SEED)
    x_train = normalize(x_train)
    x_test = normalize(x_test)

    base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

    # Use as ativações dessas camadas
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]

    layers = [base_model.get_layer(name).output for name in layer_names]
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
    down_stack.trainable = False

    model = unet_model(OUTPUT_CHANNELS, down_stack)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    tf.keras.utils.plot_model(model, show_shapes=True)

    EPOCHS = 20
    # model = None
    # model = resnet.resnet20_model(input_shape=(128, 128, 3), num_classes=11)
    # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model_history = model.fit(
        x=x_train,
        y=y_train,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_split=0.2
    )

    show_predictions(x_test[0], y_test[0], model)

    loss = model_history.history['loss']

if __name__ == "__main__":
    print("TensorFlow version: {}".format(tf.__version__))
    main()
