#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from metrics.Metric import Metric


class ConfusionMatrix(Metric):

    @staticmethod
    def plot_confusion_matrix(confusion_matrix_to_print,
                              classes,
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

    @staticmethod
    def obtain(output_predict,
               classes,
               file_name,
               class_names,
               generate_graphic_matrix=False):
        """print confusion matrix"""

        if generate_graphic_matrix:
            plt.figure()

            ConfusionMatrix.plot_confusion_matrix(confusion_matrix(output_predict,classes),
                                                  classes=class_names,
                                                  title='Confusion matrix - ' + file_name)

            plt.savefig('../output_images/' + file_name + '.png')
        else:
            print confusion_matrix(
                output_predict,
                classes)
