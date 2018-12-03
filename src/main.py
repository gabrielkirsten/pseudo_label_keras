#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Semi-supervised classifier with multiple models
    Models: Xception, VGG16, VGG19, ResNet50, InceptionV3, MobileNet

    Name: main.py
    Author: Gabriel Kirsten Menezes (gabriel.kirsten@hotmail.com)

"""

import argparse
import datetime
import os
import time

import numpy as np

from classification.PseudoLabel import PseudoLabel
from metrics.ConfusionMatrix import ConfusionMatrix
from metrics.LearningCurve import LearningCurve
from utils.DatasetUtils import DatasetUtils
from sklearn.metrics import accuracy_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings
START_TIME = time.time()

# =========================================================
# Parameters
#
# =========================================================

IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS = 256, 256, 3
BATCH_SIZE = 32
PSEUDO_LABEL_BATCH_SIZE = 64
EPOCHS = 15
CLASS_NAMES = ['AMENDOIM_BRAVO',
               'CAPIM_AMARGOSO',
               'MILHO',
               'SOJA',
               'SOLO']


def get_args():
    """Read the arguments of the program."""
    arg_parse = argparse.ArgumentParser()

    arg_parse.add_argument("-a", "--architecture",
                           required=False,
                           nargs='+',
                           help="Select architecture(Xception, VGG16, VGG19, ResNet50" +
                           ", InceptionV3, MobileNet)",
                           default=["VGG16"],
                           type=str)

    arg_parse.add_argument("-f", "--fineTuningRate",
                           required=False,
                           help="Fine tuning rate",
                           default=50,
                           type=int)

    arg_parse.add_argument("-d", "--datasetPath",
                           required=True,
                           help="Dataset location",
                           default=None,
                           type=str)

    arg_parse.add_argument("-n", "--noLabelPercent",
                           required=False,
                           nargs='+',
                           help="Percent of no label dataset",
                           default=[80],
                           type=int)

    return vars(arg_parse.parse_args())


def main():
    print "--------------------"
    print "Experiment begin at:"
    print datetime.datetime.now()
    print "--------------------"

    args = get_args()  # read arguments

    fine_tuning_percent = (
        80 if args["fineTuningRate"] == None else args["fineTuningRate"])

    dataset_utils = DatasetUtils()
    dataset_utils.create_experiment_dataset_list(args["datasetPath"],
                                                 percent_of_no_label_dataset=args['noLabelPercent'])

    for i in args['noLabelPercent']:
        dataset_utils.normalize(i)

    exit(0)

    for architecture in args["architecture"]:

        data_points_to_learning_curve = []
        for no_label_percent in args['noLabelPercent']:

            dataset_utils.get_dataset(no_label_percent)

            print "--------------------"
            print "Testing architecture: " + architecture
            print "With no label percent: " + no_label_percent
            print "--------------------"
            print ""
            print "--------------------"
            print "SUPERVISED"
            print "--------------------"
            pseudo_label_supervised_test = PseudoLabel(image_width=IMG_WIDTH,
                                                       image_height=IMG_HEIGHT,
                                                       image_channels=IMG_CHANNELS,
                                                       class_labels=CLASS_NAMES,
                                                       train_data_directory=os.path.join(dataset_utils.get_dataset(0),"train"),
                                                       validation_data_directory=os.path.join(dataset_utils.get_dataset(0),"validation"),
                                                       test_data_directory=os.path.join(dataset_utils.get_dataset(0),"test"),
                                                       no_label_data_directory=os.path.join(dataset_utils.v(0),"no_label"),
                                                       epochs=EPOCHS,
                                                       batch_size=BATCH_SIZE,
                                                       pseudo_label_batch_size=PSEUDO_LABEL_BATCH_SIZE,
                                                       transfer_learning={
                                                           'use_transfer_learning': True,
                                                           'fine_tuning': fine_tuning_percent
                                                       },
                                                       architecture=architecture,
                                                       disconsider_no_label=True)

            pseudo_label_supervised_test.fit_with_pseudo_label(use_checkpoints=True,
                                                               steps_per_epoch=pseudo_label_supervised_test.train_generator.samples // pseudo_label.batch_size,
                                                               validation_steps=pseudo_label_supervised_test.validation_generator.samples // pseudo_label.batch_size)
            print "--------------------"
            print "SEMI-SUPERVISED"
            print "--------------------"

            pseudo_label = PseudoLabel(image_width=IMG_WIDTH,
                                       image_height=IMG_HEIGHT,
                                       image_channels=IMG_CHANNELS,
                                       class_labels=CLASS_NAMES,
                                       train_data_directory=os.path.join(dataset_utils.get_dataset(no_label_percent),"train"),
                                       validation_data_directory=os.path.join(dataset_utils.get_dataset(no_label_percent),"validation"),
                                       test_data_directory=os.path.join(dataset_utils.get_dataset(no_label_percent),"test"),
                                       no_label_data_directory=os.path.join(dataset_utils.get_dataset(no_label_percent),"no_label"),
                                       epochs=EPOCHS,
                                       batch_size=BATCH_SIZE,
                                       pseudo_label_batch_size=PSEUDO_LABEL_BATCH_SIZE,
                                       transfer_learning={
                                           'use_transfer_learning': True,
                                           'fine_tuning': fine_tuning_percent
                                       },
                                       architecture=architecture)

            pseudo_label.fit_with_pseudo_label(use_checkpoints=True,
                                               steps_per_epoch=pseudo_label.train_generator.samples // pseudo_label.batch_size,
                                               validation_steps=pseudo_label.validation_generator.samples // pseudo_label.batch_size)

            print "Total time to train: %s" % (time.time() - START_TIME)

            pseudo_label.model.load_weights(
                "../models_checkpoints/" + pseudo_label.h5_filename + ".h5")

            output_predict = pseudo_label.model.predict_generator(pseudo_label.test_generator,
                                                                  pseudo_label.test_generator.samples,
                                                                  verbose=0)

            output_predict = np.argmax(output_predict, axis=1)

            output_real = pseudo_label.test_generator.classes

            ConfusionMatrix.obtain(output_predict,
                                   output_real,
                                   str(fine_tuning_percent) +
                                   '_'+str(no_label_percent),
                                   CLASS_NAMES)

            data_points_to_learning_curve.append(
                {'qtd_examples': pseudo_label.train_generator.samples, 'output_predict': output_predict, 'output_real': output_real})
            print("Accuracy: %f" % accuracy_score(output_real, output_predict))

            del pseudo_label
            del dataset_utils

        LearningCurve.obtain(data_points_to_learning_curve)

    print "--------------------"
    print "Experiment end at:"
    print datetime.datetime.now()
    print "--------------------"


if __name__ == '__main__':
    main()
