#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Semi-supervised classifier with multiple models
    Models: Xception, VGG16, VGG19, ResNet50, InceptionV3, MobileNet

    Name: main.py
    Author: Gabriel Kirsten Menezes (gabriel.kirsten@hotmail.com)

"""

import time
import os
import argparse
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from PseudoLabel import PseudoLabel
from ExperimentUtils import ExperimentUtils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings
START_TIME = time.time()

# =========================================================
# Parameters
#
# =========================================================

IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS = 128, 128, 3
TRAIN_DATA_DIR = "../data/train"
VALIDATION_DATA_DIR = "../data/validation"
TEST_DATA_DIR = "../data/test"
NO_LABEL_DATA_DIR = "../data/no_label"
BATCH_SIZE = 32
PSEUDO_LABEL_BATCH_SIZE = 64
EPOCHS = 1
CLASS_NAMES = ['ferrugemAsiatica',
               'folhaSaudavel',
               'fundo',
               'manchaAlvo',
               'mildio',
               'oidio']


def get_args():
    """Read the arguments of the program."""
    arg_parse = argparse.ArgumentParser()

    arg_parse.add_argument("-a", "--architecture",
                           required=False,
                           help="Select architecture(Xception, VGG16, VGG19, ResNet50" +
                           ", InceptionV3, MobileNet)",
                           default="VGG16",
                           type=str)

    arg_parse.add_argument("-f", "--fineTuningRate",
                           required=False,
                           help="Fine tuning rate",
                           default=None,
                           type=int)

    arg_parse.add_argument("-d", "--datasetPath",
                           required=True,
                           help="Dataset location",
                           default=None,
                           type=str)

    return vars(arg_parse.parse_args())


def plot_confusion_matrix(confusion_matrix_to_print, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
        This function prints applicationsand plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
    """

    plt.imshow(confusion_matrix_to_print,
               interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = confusion_matrix_to_print.max() / 2.
    for i, j in itertools.product(range(confusion_matrix_to_print.shape[0]),
                                  range(confusion_matrix_to_print.shape[1])):
        plt.text(j, i, format(confusion_matrix_to_print[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if confusion_matrix_to_print[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def make_confusion_matrix_and_plot(validation_generator, file_name, model_final):
    """Predict and plot confusion matrix"""

    validation_features = model_final.predict_generator(validation_generator,
                                                        validation_generator.samples,
                                                        verbose=0)

    plt.figure()

    plot_confusion_matrix(confusion_matrix(np.argmax(validation_features, axis=1),
                                           validation_generator.classes),
                          classes=CLASS_NAMES,
                          title='Confusion matrix - ' + file_name)

    plt.savefig('../output_images/' + file_name + '_doenca.png')

    print("Total time after generate confusion matrix: %s" %
          (time.time() - START_TIME))


def main():
    args = get_args()  # read arguments
    experiment_utils = ExperimentUtils()
    experiment_utils.create_experiment_dataset(args["datasetPath"])

    pseudo_label = PseudoLabel(image_width=IMG_WIDTH,
                               image_height=IMG_HEIGHT,
                               image_channels=IMG_CHANNELS,
                               class_labels=CLASS_NAMES,
                               train_data_directory=experiment_utils.train_dataset_folder,
                               validation_data_directory=experiment_utils.validation_dataset_folder,
                               no_label_data_directory=experiment_utils.no_label_dataset_folder,
                               epochs=EPOCHS,
                               batch_size=BATCH_SIZE,
                               pseudo_label_batch_size=PSEUDO_LABEL_BATCH_SIZE,
                               transfer_learning={
                                   'use_transfer_learning': True,
                                   'fine_tuning': (80 if args["fineTuningRate"] == None else args["fineTuningRate"])
                               },
                               architecture=args["architecture"])
    pseudo_label.fit_with_pseudo_label(use_checkpoints=False,
                                       steps_per_epoch=pseudo_label.train_generator.samples // pseudo_label.batch_size,
                                       validation_steps=pseudo_label.validation_generator.samples // pseudo_label.batch_size)


if __name__ == '__main__':
    main()
