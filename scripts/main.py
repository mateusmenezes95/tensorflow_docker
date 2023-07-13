from __future__ import absolute_import, division, print_function, unicode_literals
from re import T
import re

import sys

import cv2
import os
from cv2 import norm
from cv2 import mean
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from tensorflow_examples.models.pix2pix import pix2pix
import matplotlib.pyplot as plt

from IPython.display import clear_output

dataset_basepath = "/home/tensorflow/python_ws/marine-debris-fls-datasets/md_fls_dataset/data/watertank-segmentation/"
watertank_original_images_path = dataset_basepath + "Images"
watertank_mask_images_path = dataset_basepath + "Masks"

SAVE_MODEL = True
TRAIN_LENGTH = 1307
BATCH_SIZE = 32
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
OUTPUT_CHANNELS = 12
SPLIT_SEED = 0
CLASS_NAMES = [
    "background",
    "bottle",
    "can",
    "chain",
    "drink-carton",
    "hook",
    "propeller",
    "shampoo-bottle",
    "standing-bottle",
    "tire",
    "valve",
    "wall"
]

def normalize(input_image, by_mean_approach=False):
    if by_mean_approach:
        # Approach used in the paper
        input_image = input_image - 84.5
    else:
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
        print(str(len(image_files)) + " " + image_type + " images found!")
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

def create_single_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def create_all_masks(predictions):
    # Extract the predicted classes with the highest probability for each pixel
    predicted_classes = np.argmax(predictions, axis=-1)
    return predicted_classes

def show_predictions(image, mask, model):
    display([image, mask,
                create_single_mask(model.predict(image[tf.newaxis, ...]))])

def get_confusion_matrix(predictions, y_test):
    predictions_masks = create_all_masks(predictions)
    iou = keras.metrics.MeanIoU(num_classes=OUTPUT_CHANNELS)
    iou.update_state(y_test, predictions_masks)
    return np.array(iou.get_weights()).reshape(OUTPUT_CHANNELS, OUTPUT_CHANNELS), iou.result().numpy()

def get_row_sum(matrix, row_index):
    return np.sum(matrix[row_index, :])

def get_column_sum(matrix, column_index):
    return np.sum(matrix[:, column_index])

def get_class_iou(confusion_matrix, class_index):
    true_positives = confusion_matrix[class_index, class_index]
    false_negatives = get_row_sum(confusion_matrix, class_index) - true_positives
    false_positives = get_column_sum(confusion_matrix, class_index) - true_positives
    iou = true_positives / (true_positives + false_negatives + false_positives)
    return iou

def plot_confusion_matrix(cm, class_labels):
    from matplotlib.colors import Normalize

    plt.figure(figsize=(15, 15))

    norm = Normalize(vmin=np.min(cm), vmax=np.max(cm))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, norm=norm)
    plt.colorbar()
    # plt.title('Confusion Matrix')

    # Add labels to the plot
    tick_marks = np.arange(len(class_labels))
    plt.tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)

    thresh = np.max(cm) / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')

    # Set axis labels and show the plot
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    # plt.tight_layout()
    plt.show()

def print_model_fit_metrics(model_history):
    #plot the training and validation accuracy and loss at each epoch
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    epochs = range(1, int(len(loss) + 1))
    plt.plot(epochs, loss, 'y', label='Loss do treinamento')
    plt.plot(epochs, val_loss, 'r', label='Loss da validação')
    # plt.title('Loss')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, max(epochs)+1, 5))
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    plt.show()

    acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']
    plt.plot(epochs, acc, 'y', label='Precisão do treinamento')
    plt.plot(epochs, val_acc, 'r', label='Precisão da validação')
    # plt.title('Training and validation accuracy')
    plt.xlabel('Época')
    plt.ylabel('Precisão')
    plt.xticks(np.arange(0, max(epochs)+1, 5))
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    plt.show()

def print_iou_per_class_table(confusion_matrix, class_nameS):
    # Create a dictionary with class names and their corresponding indices
    class_indices = {class_name: i for i, class_name in enumerate(class_nameS)}

    # Create a table header
    table_header = "| Class Name | Class IoU |"
    table_line = "-" * len(table_header)

    # Print the table header
    print(table_line)
    print(table_header)
    print(table_line)
    # Iterate over the class names and calculate the IoU for each class
    for class_name in class_nameS:
        class_index = class_indices[class_name]
        class_iou = get_class_iou(confusion_matrix, class_index)
        # Format the class name and IoU values
        class_name_formatted = f"| {class_name:<14}"
        class_iou_formatted = f"| {class_iou:.4f}     "

        # Print the table row
        print(f"{class_name_formatted}{class_iou_formatted}|")

    # Print the table bottom line
    print(table_line)

def get_images_per_class_indices(test_set, class_label):
    selected_images = []
    np.random.seed(SPLIT_SEED)
    indices = np.where(test_set == class_label)[0]
    selected_index = np.random.choice(indices)
    selected_images.append(selected_index)
    
    return selected_images

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

if __name__ == "__main__":
    print("TensorFlow version: {}".format(tf.__version__))
    original_images = get_images(dataset_basepath, "original")
    mask_images = get_images(dataset_basepath, "mask")

    x_train, x_test, y_train, y_test = train_test_split(original_images, mask_images,
                                                        shuffle=True,
                                                        test_size=0.3,
                                                        random_state=SPLIT_SEED)
    x_train = normalize(x_train)
    x_test = normalize(x_test)

    script_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_path, "mobile_net_v2_to_sonar.h5")

    if (SAVE_MODEL):
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=[128, 128, 3],
            include_top=False)

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
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        tf.keras.utils.plot_model(model, show_shapes=True)

        EPOCHS = 20

        model_history = model.fit(
            x=x_train,
            y=y_train,
            epochs=EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_split=0.25
        )
        model.save(model_path)
        print_model_fit_metrics(model_history)
        print("Evaluate on test data")
    else:
        model = tf.keras.models.load_model(model_path)

    for i in range(len(CLASS_NAMES)):
        index = get_images_per_class_indices(y_test, i)[0]
        print("Class:", CLASS_NAMES[i])
        show_predictions(x_test[index], y_test[index], model)

    results = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
    print("test loss, test acc:", results)
    
    predictions = model.predict(x_test)
    
    cm, mean_iou = get_confusion_matrix(predictions, y_test)
    plot_confusion_matrix(cm, CLASS_NAMES)
    plot_confusion_matrix(cm[1:, 1:], CLASS_NAMES[1:])
    print("Mean IoU:", mean_iou)
    print_iou_per_class_table(cm, CLASS_NAMES)
