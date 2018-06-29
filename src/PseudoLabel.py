#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings

import numpy as np

from keras import applications
from keras import callbacks as cbks
from keras.applications import Xception, VGG16, VGG19, ResNet50, InceptionV3, MobileNet
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam, TFOptimizer
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from keras.utils import Sequence
from keras.utils import GeneratorEnqueuer
from keras.utils import OrderedEnqueuer

# CONSTANTS
LIST_OF_ACCEPTABLES_ARCHITECTURES = {
    'Xception': Xception,
    'VGG16': VGG16,
    'VGG19': VGG19,
    'ResNet50': ResNet50,
    'InceptionV3': InceptionV3,
    'MobileNet': MobileNet
}

LIST_OF_ACCEPTABLES_OPTIMIZERS = {
    'SGD': SGD,
    'Adagrad': Adagrad,
    'RMSprop': RMSprop,
    'Adadelta': Adadelta,
    'Adam': Adam,
    'Adamax': Adamax,
    'TFOptimizer': TFOptimizer
}

LIST_OF_ACCEPTABLES_LOSS_FUNCTION = [
    'mean_squared_error',
    'mean_absolute_percentage_error',
    'mean_squared_logarithmic_error',
    'squared_hinge',
    'hinge',
    'categorical_hinge',
    'logcosh',
    'categorical_crossentropy',
    'sparse_categorical_crossentropy',
    'binary_crossentropy',
    'kullback_leibler_divergence',
    'poisson',
    'cosine_proximity'
]

LIST_OF_ACCEPTABLES_METRICS = [
    'acc',
    'accuracy',
    'binary_accuracy',
    'categorical_accuracy',
    'sparse_categorical_accuracy',
    'top_k_categorical_accuracy',
    'sparse_top_k_categorical_accuracy'
]


class PseudoLabel:
    """
        Pseudo-label Class
    """

    def __init__(self,
                 image_width=256,
                 image_height=256,
                 train_data_directory="../data/train",
                 validation_data_directory="../data/validation",
                 test_data_directory="../data/test",
                 no_label_data_directory="../data/no_label",
                 batch_size=8,
                 pseudo_label_batch_size=16,
                 epochs=1,
                 class_names=[],
                 architecture="VGG16",
                 image_channels=3,
                 learnin_rate=0.001,
                 save_heights=False,
                 transfer_learning={'use_transfer_learning': False,
                                    'fine_tuning': None},
                 optimizer='SGD',
                 metrics_list=['acc'],
                 loss_function='categorical_crossentropy',
                 class_labels=None):
        """
            Pseudo-label class construtor
        """

        # Atributes declarations
        self.image_width = image_width
        self.image_height = image_height
        self.train_data_directory = train_data_directory
        self.validation_data_directory = validation_data_directory
        self.test_data_directory = test_data_directory
        self.no_label_data_directory = no_label_data_directory
        self.batch_size = batch_size
        self.pseudo_label_batch_size = pseudo_label_batch_size
        self.epochs = epochs
        self.class_names = class_names
        self.architecture = architecture
        self.image_channels = image_channels
        self.learning_rate = learnin_rate
        self.use_transfer_learning = transfer_learning.get('use_transfer_learning')
        self.fine_tuning_rate = transfer_learning.get('fine_tuning')
        self.optimizer = optimizer
        self.metrics_list = metrics_list
        self.loss_function = loss_function
        self.model = None
        self.train_generator = None
        self.validation_generator = None
        self.h5_filename = None
        self.class_labels = class_labels

        # Make your model and dataset
        self.make_data_generators()
        self.make_model(architecture=self.architecture,
                        use_transfer_learning=self.use_transfer_learning,
                        fine_tuning_rate=self.fine_tuning_rate,
                        optimizer=self.optimizer,
                        metrics_list=self.metrics_list,
                        loss_function=self.loss_function)

        self.generate_h5_filename() if save_heights else None

    def make_model(self,
                   architecture=None,
                   use_transfer_learning=False,
                   fine_tuning_rate=None,
                   optimizer='SGD',
                   metrics_list=['accuracy'],
                   loss_function='categorical_crossentropy'):
        """
            Create your CNN keras model

            Arguments:
                architecture (str): architecture of model 
        """
        # Validations
        for metric in metrics_list:
            if metric not in LIST_OF_ACCEPTABLES_METRICS:
                raise ValueError("The specified metric \'" +
                                 metric + "\' is not supported")
        if loss_function not in LIST_OF_ACCEPTABLES_LOSS_FUNCTION:
            raise ValueError("The specified loss function \'" +
                             loss_function + "\' is not supported!")
        if optimizer not in LIST_OF_ACCEPTABLES_OPTIMIZERS.keys():
            raise ValueError("The specified optimizer \'" +
                             optimizer + "\' is not supported!")
        if architecture not in LIST_OF_ACCEPTABLES_ARCHITECTURES.keys():
            raise ValueError("The specified architecture \'" +
                             architecture + "\' is not supported!")
        else:
            if use_transfer_learning and not 0 <= fine_tuning_rate <= 100:
                raise ValueError("The fine tuning rate must be beetween 0 and 100!")
            if use_transfer_learning and fine_tuning_rate == None:
                raise ValueError(
                    "You need to specify a fine tuning rate if you're using transfer learning!")

        # With transfer learning
        if use_transfer_learning:
            self.model = LIST_OF_ACCEPTABLES_ARCHITECTURES.get(architecture)(
                weights="imagenet",
                include_top=False,
                input_shape=(self.image_height, self.image_width, self.image_channels))

            last_layers = len(self.model.layers) - \
                int(len(self.model.layers) * (fine_tuning_rate / 100.))

            for layer in self.model.layers[:last_layers]:
                layer.trainable = False

        # Without transfer learning
        else:
            self.model = LIST_OF_ACCEPTABLES_ARCHITECTURES.get(architecture)(
                weights=None,
                include_top=False,
                input_shape=(self.image_height, self.image_width, self.image_channels))

            for layer in self.model.layers:
                layer.trainable = True

        # Adding the custom Layers
        new_custom_layers = self.model.output
        new_custom_layers = Flatten()(new_custom_layers)
        new_custom_layers = Dense(1024, activation="relu")(new_custom_layers)
        new_custom_layers = Dropout(0.5)(new_custom_layers)
        new_custom_layers = Dense(1024, activation="relu")(new_custom_layers)
        predictions = Dense(self.train_generator.num_classes,
                            activation="softmax")(new_custom_layers)

        # Create the final model
        self.model = Model(inputs=self.model.input, outputs=predictions)

        # Compile model
        self.model.compile(loss=loss_function,
                           optimizer=LIST_OF_ACCEPTABLES_OPTIMIZERS.get(optimizer)(
                               lr=self.learning_rate
                           ),
                           metrics=metrics_list)

    def make_data_generators(self, use_data_augmentation=False):
        """
            Function that initiate the train, validation and test generators with data augumentation
        """
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            horizontal_flip=True,
            fill_mode="nearest",
            zoom_range=0.3,
            width_shift_range=0.3,
            height_shift_range=0.3,
            rotation_range=30)

        self.train_generator = train_datagen.flow_from_directory(
            self.train_data_directory,
            target_size=(self.image_height, self.image_width),
            color_mode='rgb',
            classes=self.class_labels,
            batch_size=self.batch_size,
            shuffle=True,
            class_mode="categorical")

        test_datagen = ImageDataGenerator(
            rescale=1. / 255,
            horizontal_flip=True,
            fill_mode="nearest",
            zoom_range=0.3,
            width_shift_range=0.3,
            height_shift_range=0.3,
            rotation_range=30)

        self.validation_generator = test_datagen.flow_from_directory(
            self.validation_data_directory,
            target_size=(self.image_height, self.image_width),
            color_mode='rgb',
            batch_size=self.batch_size,
            shuffle=True,
            class_mode="categorical")

        no_label_datagen = ImageDataGenerator()

        self.no_label_generator = no_label_datagen.flow_from_directory(
            self.no_label_data_directory,
            target_size=(self.image_height, self.image_width),
            color_mode='rgb',
            batch_size=self.pseudo_label_batch_size,
            shuffle=False,
            class_mode="categorical")

        self.no_label_generator.num_classes = self.validation_generator.num_classes

    def generate_h5_filename(self):
        """
            Generate the .h5 filename. The .h5 file is the file that contains your trained model
        """

        if self.fine_tuning_rate == 100:
            self.h5_filename = self.architecture + \
                '_transfer_learning'
        elif self.fine_tuning_rate == None:
            self.h5_filename = self.architecture + \
                '_without_transfer_learning'
        else:
            self.h5_filename = self.architecture + \
                '_fine_tunning_' + str(self.fine_tuning_rate)

    ################################################################################
    # Semi-supervised - Pseudo label approach
    ################################################################################
    def fit_with_pseudo_label(self,
                              steps_per_epoch,
                              use_checkpoints=False,
                              class_labels=None,
                              verbose=1,
                              use_multiprocessing=False,
                              shuffle=False,
                              workers=1,
                              max_queue_size=10):

        wait_time = 0.01  # in seconds

        self.model._make_train_function()

        # Create a checkpoint callback
        checkpoint = ModelCheckpoint("../models_checkpoints/" + str(self.h5_filename) + ".h5",
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='auto',
                                     period=1)

        # Generate callbacks
        callback_list = []
        if use_checkpoints:
            callback_list.extend(checkpoint)

        # Init train counters
        epoch = 0

        # Prepare display labels.
        out_labels = self.model._get_deduped_metrics_names()
        callback_metrics = out_labels + ['val_' + n for n in out_labels]

        # Prepare train callbacks
        self.model.history = cbks.History()
        callbacks = [cbks.BaseLogger()] + (callback_list or []) + \
            [self.model.history]
        if verbose:
            callbacks += [cbks.ProgbarLogger(count_mode='steps')]
        callbacks = cbks.CallbackList(callbacks)

        # it's possible to callback a different model than self:
        if hasattr(self.model, 'callback_model') and self.model.callback_model:
            callback_model = self.model.callback_model

        else:
            callback_model = self.model

        callbacks.set_model(callback_model)

        is_sequence = isinstance(self.train_generator, Sequence)
        if not is_sequence and use_multiprocessing and workers > 1:
            warnings.warn(
                UserWarning('Using a generator with `use_multiprocessing=True`'
                            ' and multiple workers may duplicate your data.'
                            ' Please consider using the`keras.utils.Sequence'
                            ' class.'))

        if is_sequence:
            steps_per_epoch = len(self.train_generator)
        enqueuer = None

        callbacks.set_params({
            'epochs': self.epochs,
            'steps': steps_per_epoch,
            'verbose': verbose,
            'do_validation': True,
            'metrics': callback_metrics,
        })
        callbacks.on_train_begin()

        try:
            if is_sequence:
                enqueuer = OrderedEnqueuer(self.train_generator,
                                           use_multiprocessing=use_multiprocessing,
                                           shuffle=shuffle)
            else:
                enqueuer = GeneratorEnqueuer(self.train_generator,
                                             use_multiprocessing=use_multiprocessing,
                                             wait_time=wait_time)
            enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            output_generator = enqueuer.get()

            # Train the model
            # Epochs
            while epoch < self.epochs:
                callbacks.on_epoch_begin(epoch)
                steps_done = 0
                batch_index = 0

                # Steps per epoch
                while steps_done < steps_per_epoch:

                    generator_output = next(output_generator)

                    if len(generator_output) == 2:
                        x, y = generator_output
                        sample_weight = None
                    elif len(generator_output) == 3:
                        x, y, sample_weight = generator_output
                    else:
                        raise ValueError('Output of generator should be '
                                         'a tuple `(x, y, sample_weight)` '
                                         'or `(x, y)`. Found: ' +
                                         str(generator_output))

                    #==========================
                    # Mini-batch
                    #==========================
                    print ''
                    print 'Generating pseudo-labels...'
                    no_label_output = self.model.predict_generator(
                        self.no_label_generator, 
                        None, # because the model is instance of sequence
                        verbose=1)

                    # One-hot encoded
                    self.no_label_generator.classes = np.argmax(no_label_output, axis=1)

                    # concat 
                    self.no_label_generator.batch_index = self.train_generator.batch_index-1
                    x_pseudo, y_pseudo = next(self.no_label_generator)
                    x, y = np.concatenate((x, x_pseudo), axis=0), np.concatenate((y, y_pseudo), axis=0)                    

                    if len(generator_output) == 2:
                        x, y = generator_output
                        sample_weight = None
                    elif len(generator_output) == 3:
                        x, y, sample_weight = generator_output
                    else:
                        raise ValueError('Output of generator should be '
                                         'a tuple `(x, y, sample_weight)` '
                                         'or `(x, y)`. Found: ' +
                                         str(generator_output))

                    # build batch logs
                    batch_logs = {}
                    if isinstance(x, list):
                        batch_size = x[0].shape[0]
                    elif isinstance(x, dict):
                        batch_size = list(x.values())[0].shape[0]
                    else:
                        batch_size = x.shape[0]
                    batch_logs['batch'] = batch_index
                    batch_logs['size'] = batch_size
                    callbacks.on_batch_begin(batch_index, batch_logs)
                    
                    # Runs a single gradient update on a single batch of data
                    scalar_training_loss = self.model.train_on_batch(x=x, y=y)

                    if not isinstance(scalar_training_loss, list):
                        scalar_training_loss = [scalar_training_loss]
                    for l, o in zip(out_labels, scalar_training_loss):
                        batch_logs[l] = o

                    callbacks.on_batch_end(batch_index, batch_logs)

                    #==========================
                    # end Mini-batch
                    #==========================

                    batch_index += 1
                    steps_done += 1

                # Epoch finished.
                epoch += 1

        finally:
            if enqueuer is not None:
                enqueuer.stop()

        callbacks.on_train_end()
        return self.model.history
