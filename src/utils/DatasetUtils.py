#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import shutil

"""
Class to generate dataset of files by creating a reference symbolic link

# Atributes: 
    experiment_folder (str): Folder that contains dataset folders (eg. train/test/validation)
    train_dataset_folder (str): Folder that contains train files
    test_dataset_folder (str): Folder that contains test files
    validation_dataset_folder (str): Folder that contains validation files
    no_label_dataset_folder (str): Folder that contains no_label files (to semi-supervised trainning only)

# Notes:
    Name: DatasetUtils.py
    Author: Author: Gabriel Kirsten Menezes (https://github.com/gabrielkirsten)
    GitHub: https://github.com/gabrielkirsten/pseudo_label_keras
"""

class DatasetUtils:


    def __init__(self):
        self.experiment_folder = None
        self.train_dataset_folder = None
        self.test_dataset_folder = None
        self.validation_dataset_folder = None
        self.no_label_dataset_folder = None

    def create_experiment_dataset(self,
                                  dataset_path,
                                  folder_name_experiment='.experiment',
                                  percent_of_train_dataset=60,
                                  percent_of_test_dataset=20,
                                  percent_of_validation_dataset=20,
                                  percent_of_no_label_dataset=80.0):
        """
        Create the file structure and symbolic links to your dataset.

        # Arguments:
            dataset_path (str): 
                Path to your dataset folders. 
            folder_name_experiment (str):
                Name of folder that contains your prepared dataset (default '.experiment').
            percent_of_train_dataset (str): 
                Percentage of files that will be used to train (default = 60)
            percent_of_test_dataset (str): 
                Percentage of files that will be used to test (default = 20) 
            percent_of_validation_dataset (str): 
                Percentage of files that will be used to validation (default = 20) 
            percent_of_no_label_dataset (str): 
                Percentage of files that will be used to remove label (default = 80) 
        """

        if percent_of_train_dataset+percent_of_test_dataset+percent_of_validation_dataset != 100.0:
            raise ValueError(
                "The sum of train, test and validation dataset must be 100.00!")

        if not (100 > percent_of_no_label_dataset > 0):
            raise ValueError(
                "The value of percent_of_no_label_dataset, must be between 0.00 and 100.00!")

        self.experiment_folder = dataset_path+'/'+folder_name_experiment
        self.train_dataset_folder = self.experiment_folder+'/train'
        self.test_dataset_folder = self.experiment_folder+'/test'
        self.validation_dataset_folder = self.experiment_folder+'/validation'
        self.no_label_dataset_folder = self.experiment_folder+'/no_label'

        self._create_dataset_folders()

        self._process_classes(dataset_path,
                              folder_name_experiment,
                              percent_of_no_label_dataset,
                              percent_of_train_dataset,
                              percent_of_test_dataset,
                              percent_of_validation_dataset)

    def _create_dataset_folders(self):
        # Create dataset folder
        if not os.path.exists(self.experiment_folder):
            os.makedirs(self.experiment_folder)

        # Create train dataset folder
        if not os.path.exists(self.train_dataset_folder):
            os.makedirs(self.train_dataset_folder)

        # Create test dataset folder
        if not os.path.exists(self.test_dataset_folder):
            os.makedirs(self.test_dataset_folder)

        # Create validation dataset folder
        if not os.path.exists(self.validation_dataset_folder):
            os.makedirs(self.validation_dataset_folder)

        # Create unlabel dataset folder
        no_label_folder = os.path.join(self.no_label_dataset_folder,
                                       'no_label')
        if not os.path.exists(no_label_folder):
            os.makedirs(no_label_folder)
        else:
            shutil.rmtree(no_label_folder)
            os.makedirs(no_label_folder)

    def _process_classes(self,
                         dataset_path,
                         folder_name_experiment,
                         percent_of_no_label_dataset,
                         percent_of_train_dataset,
                         percent_of_test_dataset,
                         percent_of_validation_dataset):

        for class_name in os.listdir(dataset_path):
            if (class_name != folder_name_experiment):
                class_folder = os.path.join(dataset_path, class_name)

                self._create_all_folders_to_class(class_name)

                for (class_path, _, filenames) in os.walk(class_folder):
                    self._process_class(class_path,
                                        filenames,
                                        class_name,
                                        percent_of_no_label_dataset,
                                        percent_of_train_dataset,
                                        percent_of_test_dataset,
                                        percent_of_validation_dataset)

    def _process_class(self,
                       class_path,
                       filenames,
                       class_name,
                       percent_of_no_label_dataset,
                       percent_of_train_dataset,
                       percent_of_test_dataset,
                       percent_of_validation_dataset):

        self._randomize_files(class_path, filenames)

        size_of_no_label_dataset, size_of_train_dataset, size_of_test_dataset, size_of_validation_dataset = self._calculate_dataset_sizes(filenames,
                                                                                                                                          percent_of_no_label_dataset,
                                                                                                                                          percent_of_train_dataset,
                                                                                                                                          percent_of_test_dataset,
                                                                                                                                          percent_of_validation_dataset)

        self._create_symbolic_links(class_path,
                                    filenames,
                                    class_name,
                                    size_of_no_label_dataset,
                                    size_of_train_dataset,
                                    size_of_test_dataset,
                                    size_of_validation_dataset)

    def _randomize_files(self, class_path, filenames):
        class_path = os.path.abspath(class_path)
        random.shuffle(filenames)

    def _create_symbolic_links(self,
                               class_path,
                               class_name,
                               filenames,
                               size_of_no_label_dataset,
                               size_of_train_dataset,
                               size_of_test_dataset,
                               size_of_validation_dataset):

        for file_to_create_symbolic_link in filenames[0:size_of_no_label_dataset]:
            os.symlink(os.path.join(class_path, file_to_create_symbolic_link),
                       os.path.join(self.experiment_folder,
                                    'no_label',
                                    'no_label',
                                    file_to_create_symbolic_link))
        current_index = size_of_no_label_dataset

        # Train
        for file_to_create_symbolic_link in filenames[current_index:current_index+size_of_train_dataset]:
            self._create_symbolic_link(
                class_path,
                class_name,
                file_to_create_symbolic_link,
                self.train_dataset_folder)
        current_index = size_of_no_label_dataset

        # Test
        for file_to_create_symbolic_link in filenames[current_index:current_index+size_of_test_dataset]:
            self._create_symbolic_link(class_path,
                                       class_name,
                                       file_to_create_symbolic_link,
                                       self.test_dataset_folder)
        current_index = current_index+size_of_test_dataset

        # Validation
        for file_to_create_symbolic_link in filenames[current_index:-1]:
            self._create_symbolic_link(class_path,
                                       class_name,
                                       file_to_create_symbolic_link,
                                       self.validation_dataset_folder)
        current_index = size_of_no_label_dataset

    def _create_symbolic_link(self, class_path, class_name, file_to_create_symbolic_link, dataset_folder):
        os.symlink(os.path.join(class_path,
                                file_to_create_symbolic_link),
                   os.path.join(dataset_folder,
                                class_name,
                                file_to_create_symbolic_link))

    def _create_all_folders_to_class(self, class_name):
        self._create_train_folder(class_name)
        self._create_test_folder(class_name)
        self._create_validation_folder(class_name)

    def _calculate_dataset_sizes(self,
                                 filenames,
                                 percent_of_no_label_dataset,
                                 percent_of_train_dataset,
                                 percent_of_test_dataset,
                                 percent_of_validation_dataset):

        size_of_no_label_dataset = int(
            (len(filenames)*percent_of_no_label_dataset)//100)

        size_of_train_dataset = int(
            ((len(filenames)-size_of_no_label_dataset)*percent_of_train_dataset)//100)

        size_of_test_dataset = int(
            ((len(filenames)-size_of_no_label_dataset)*percent_of_test_dataset)//100)

        size_of_validation_dataset = int(
            ((len(filenames)-size_of_no_label_dataset)*percent_of_validation_dataset)//100)

        return size_of_no_label_dataset, size_of_train_dataset, size_of_test_dataset, size_of_validation_dataset

    def _create_test_folder(self, class_name):
        test_folder = os.path.join(self.experiment_folder,
                                   'test',
                                   class_name)
        self._create_or_delete_old_dataset(test_folder)

    def _create_train_folder(self, class_name):
        train_folder = os.path.join(self.experiment_folder,
                                    'train',
                                    class_name)
        self._create_or_delete_old_dataset(train_folder)

    def _create_validation_folder(self, class_name):
        validation_folder = os.path.join(self.experiment_folder,
                                         'validation',
                                         class_name)
        self._create_or_delete_old_dataset(validation_folder)

    def _create_or_delete_old_dataset(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            shutil.rmtree(folder)
            os.makedirs(folder)
