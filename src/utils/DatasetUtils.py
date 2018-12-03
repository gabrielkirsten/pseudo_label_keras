#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import shutil
import sys

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
                                  dataset_is_sub_folder=False,
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
                "The sum of train, test and validation dataset must be 100.00!"
            )

        if not (100 >= percent_of_no_label_dataset >= 0):
            raise ValueError(
                "The value of percent_of_no_label_dataset, must be between 0.00 and 100.00!"
            )

        self.experiment_folder = os.path.join(
            dataset_path if not dataset_is_sub_folder else os.path.abspath(os.path.join(
                os.path.abspath(os.path.join(dataset_path, os.pardir)), os.pardir)),
            folder_name_experiment
        )
        self.train_dataset_folder = os.path.join(
            self.experiment_folder,
            'train'
        )
        self.test_dataset_folder = os.path.join(
            self.experiment_folder,
            'test'
        )
        self.validation_dataset_folder = os.path.join(
            self.experiment_folder,
            'validation'
        )
        self.no_label_dataset_folder = os.path.join(
            self.experiment_folder,
            'no_label'
        )

        self._create_dataset_folders()

        self._process_classes(dataset_path,
                              folder_name_experiment,
                              percent_of_no_label_dataset,
                              percent_of_train_dataset,
                              percent_of_test_dataset,
                              percent_of_validation_dataset)

    def create_experiment_dataset_list(self,
                                       dataset_path,
                                       percent_of_no_label_dataset,
                                       use_old_dataset,
                                       folder_name_experiment='.experiment',
                                       percent_of_train_dataset=60,
                                       percent_of_test_dataset=20,
                                       percent_of_validation_dataset=20):

        if not use_old_dataset:
            try:
                shutil.rmtree(os.path.join(dataset_path, folder_name_experiment))
            except:
                pass

        if 0 not in percent_of_no_label_dataset:
            raise ValueError(
                "The percent of no label images must contains zero!"
            )

        for index, percent in enumerate(percent_of_no_label_dataset):
            self.create_experiment_dataset(
                dataset_is_sub_folder=percent != 0,
                dataset_path=dataset_path if percent == 0 else os.path.join(
                    dataset_path, folder_name_experiment, str(
                        percent_of_no_label_dataset[index-1])
                ),
                folder_name_experiment=os.path.join(
                    folder_name_experiment, str(percent)
                ),
                percent_of_train_dataset=percent_of_train_dataset,
                percent_of_test_dataset=percent_of_test_dataset,
                percent_of_validation_dataset=percent_of_validation_dataset,
                percent_of_no_label_dataset=percent
            )

    def normalize(self, no_label_percent):
        min_examples_in_class = sys.maxsize
        for class_name in os.listdir(self.get_dataset(no_label_percent)):
            if class_name == "train":
                class_folder = os.path.join(self.get_dataset(no_label_percent), class_name)
                for (class_path, _, filenames) in os.walk(class_folder):
                    if (min_examples_in_class > len(filenames) and len(filenames) > 0):
                        min_examples_in_class = len(filenames)
        for class_name in os.listdir(self.get_dataset(no_label_percent)):
            if class_name == "train":
                class_folder = os.path.join(self.get_dataset(no_label_percent), class_name)
                for class_name in os.listdir(class_folder):
                    if class_name != "no_label":
                        class_folder = os.path.join(self.get_dataset(no_label_percent), class_name)
                        for (class_path, _, filenames) in os.walk(class_folder):
                            print class_path
                            if len(filenames) > min_examples_in_class:   
                                self._randomize_files(class_path, filenames)
                                qtd_files_to_remove = len(filenames) - min_examples_in_class
                                while qtd_files_to_remove:
                                    qtd_files_to_remove = qtd_files_to_remove -1
                                    os.remove(os.path.join(class_path, filenames[qtd_files_to_remove]))
                                    print "removed"
                                    print os.path.join(class_path, filenames[qtd_files_to_remove])


    def get_dataset(self, no_label_percent):
        return os.path.join(os.path.abspath(os.path.join(self.experiment_folder, os.pardir)), str(no_label_percent))

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
                         percent_of_validation_dataset,
                         is_dataset_class=False,
                         dataset_class_name=""):

        for class_name in os.listdir(dataset_path):
            if (class_name != folder_name_experiment) and class_name not in ".experiment" and class_name not in ['train', 'test', 'validation']:
                class_folder = os.path.join(dataset_path, class_name)

                self._create_all_folders_to_class(class_name)

                for (class_path, _, filenames) in os.walk(class_folder):
                    self._process_class(class_path,
                                        filenames,
                                        class_name,
                                        percent_of_no_label_dataset,
                                        percent_of_train_dataset,
                                        percent_of_test_dataset,
                                        percent_of_validation_dataset,
                                        is_dataset_class=is_dataset_class,
                                        dataset_class_name=dataset_class_name)

            elif class_name in ['train', 'test', 'validation', 'no_label']:
                self._process_classes(dataset_path=os.path.join(dataset_path, class_name),
                                      folder_name_experiment=folder_name_experiment,
                                      percent_of_no_label_dataset=percent_of_no_label_dataset,
                                      percent_of_train_dataset=percent_of_train_dataset,
                                      percent_of_test_dataset=percent_of_test_dataset,
                                      percent_of_validation_dataset=percent_of_validation_dataset,
                                      is_dataset_class=True,
                                      dataset_class_name=class_name)
            else:
                pass

    def _process_class(self,
                       class_path,
                       filenames,
                       class_name,
                       percent_of_no_label_dataset,
                       percent_of_train_dataset,
                       percent_of_test_dataset,
                       percent_of_validation_dataset,
                       is_dataset_class=False,
                       dataset_class_name=""):

        self._randomize_files(class_path, filenames)

        size_of_no_label_dataset, size_of_train_dataset, size_of_test_dataset, size_of_validation_dataset = self._calculate_dataset_sizes(filenames,
                                                                                                                                          percent_of_no_label_dataset,
                                                                                                                                          percent_of_train_dataset,
                                                                                                                                          percent_of_test_dataset,
                                                                                                                                          percent_of_validation_dataset)

        self._create_symbolic_links(class_path,
                                    class_name,
                                    filenames,
                                    size_of_no_label_dataset,
                                    size_of_train_dataset,
                                    size_of_test_dataset,
                                    size_of_validation_dataset,
                                    is_dataset_class=is_dataset_class,
                                    dataset_class_name=dataset_class_name)
        

    def _randomize_files(self, class_path, filenames):
        class_path = os.path.abspath(class_path)
        random.shuffle(filenames)

    def _create_symbolic_links_to_dataset_folder(self,
                                                 class_path,
                                                 class_name,
                                                 dataset_folder,
                                                 filenames,
                                                 size_of_no_label_dataset,
                                                 size_of_train_dataset,
                                                 size_of_test_dataset,
                                                 size_of_validation_dataset):

        for file_to_create_symbolic_link in filenames[0:size_of_no_label_dataset]:
            os.symlink(os.path.join(class_path,
                                    file_to_create_symbolic_link),
                       os.path.join(self.experiment_folder,
                                    'no_label',
                                    'no_label',
                                    file_to_create_symbolic_link))

        current_index = size_of_no_label_dataset

        for file_to_create_symbolic_link in filenames[current_index:]:

            path1 = os.readlink(os.path.join(class_path, file_to_create_symbolic_link))
            path2 = os.path.join(dataset_folder, self._get_folder_by_dataset_folder(dataset_folder), class_name, file_to_create_symbolic_link) if dataset_folder is not 'no_label' else os.path.join(self.experiment_folder, 'no_label', 'no_label', file_to_create_symbolic_link)
            
            os.symlink(path1, path2)

    def _get_folder_by_dataset_folder(self, dataset_folder):
        switcher = {
            'train': self.train_dataset_folder,
            'test': self.test_dataset_folder,
            'validation': self.validation_dataset_folder
        }
        return switcher.get(dataset_folder)

    def _create_symbolic_links(self,
                               class_path,
                               class_name,
                               filenames,
                               size_of_no_label_dataset,
                               size_of_train_dataset,
                               size_of_test_dataset,
                               size_of_validation_dataset,
                               is_dataset_class=False,
                               dataset_class_name=""):

        if is_dataset_class:
            self._create_symbolic_links_to_dataset_folder(
                class_path,
                class_name,
                dataset_class_name,
                filenames,
                size_of_no_label_dataset,
                size_of_train_dataset,
                size_of_test_dataset,
                size_of_validation_dataset)

        else:

            # No label
            for file_to_create_symbolic_link in filenames[0:size_of_no_label_dataset]:
                os.symlink(os.path.join(class_path,
                                        file_to_create_symbolic_link),
                           os.path.join(self.experiment_folder,
                                        'no_label',
                                        'no_label',
                                        file_to_create_symbolic_link))
            current_index = size_of_no_label_dataset

            # Test
            for file_to_create_symbolic_link in filenames[current_index:current_index+size_of_test_dataset]:
                self._create_symbolic_link(class_path,
                                           class_name,
                                           file_to_create_symbolic_link,
                                           self.test_dataset_folder)
            current_index = current_index+size_of_test_dataset

            # Validation
            for file_to_create_symbolic_link in filenames[current_index:current_index+size_of_validation_dataset]:
                self._create_symbolic_link(class_path,
                                           class_name,
                                           file_to_create_symbolic_link,
                                           self.validation_dataset_folder)

            current_index = current_index+size_of_validation_dataset

            # Train
            for file_to_create_symbolic_link in filenames[current_index:]:
                self._create_symbolic_link(
                    class_path,
                    class_name,
                    file_to_create_symbolic_link,
                    self.train_dataset_folder)
            current_index = size_of_train_dataset

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
            # shutil.rmtree(folder)
            # os.makedirs(folder)
            pass
