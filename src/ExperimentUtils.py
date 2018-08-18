import os
import random
import shutil

# TODO - separate methods
# TODO - use os.path.join()

class ExperimentUtils:

    def __init__(self):
        self.experiment_folder = None
        self.train_dataset_folder = None
        self.test_dataset_folder = None
        self.validation_dataset_folder = None
        self.no_label_dataset_folder = None
    

    def create_experiment_dataset(self,
                                  dataset_path,
                                  folder_name_experiment='.experiment',
                                  percent_of_train_dataset=20,
                                  percent_of_test_dataset=60,
                                  percent_of_validation_dataset=20,
                                  percent_of_no_label_dataset=80.0):

        if percent_of_train_dataset+percent_of_test_dataset+percent_of_validation_dataset != 100.0:
            raise ValueError(
                "The sum of train, test and validation dataset must be 100.00!")

        if not (100 > percent_of_no_label_dataset > 0):
            raise ValueError(
                "The value of percent_of_no_label_dataset, must be between 0.00 and 100.00!")

        self.experiment_folder = experiment_folder = dataset_path+'/'+folder_name_experiment
        self.train_dataset_folder = train_dataset_folder = self.experiment_folder+'/train'
        self.test_dataset_folder = test_dataset_folder = self.experiment_folder+'/test'
        self.validation_dataset_folder = validation_dataset_folder = self.experiment_folder+'/validation'
        self.no_label_dataset_folder = no_label_dataset_folder = self.experiment_folder+'/no_label'

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
        if not os.path.exists(self.no_label_dataset_folder+'/no_label/'):
            os.makedirs(self.no_label_dataset_folder+'/no_label/')
        else:
            shutil.rmtree(self.no_label_dataset_folder+'/no_label/')
            os.makedirs(self.no_label_dataset_folder+'/no_label/')

        for class_name in os.listdir(dataset_path):
            if (class_name != folder_name_experiment):
                class_folder = dataset_path+'/'+class_name

                if not os.path.exists(self.experiment_folder+'/train/'+class_name):
                    os.makedirs(self.experiment_folder+'/train/'+class_name)
                else:
                    shutil.rmtree(self.experiment_folder+'/train/'+class_name)
                    os.makedirs(self.experiment_folder+'/train/'+class_name)

                if not os.path.exists(self.experiment_folder+'/test/'+class_name):
                    os.makedirs(self.experiment_folder+'/test/'+class_name)
                else:
                    shutil.rmtree(self.experiment_folder+'/test/'+class_name)
                    os.makedirs(self.experiment_folder+'/test/'+class_name)

                if not os.path.exists(self.experiment_folder+'/validation/'+class_name):
                    os.makedirs(self.experiment_folder+'/validation/'+class_name)
                else:
                    shutil.rmtree(self.experiment_folder+'/validation/'+class_name)
                    os.makedirs(self.experiment_folder+'/validation/'+class_name)

                for (class_path, _, filenames) in os.walk(class_folder):
                    # print class_path
                    class_path = os.path.abspath(class_path)
                    random.shuffle(filenames)

                    size_of_no_label_dataset =   int((len(filenames)*percent_of_no_label_dataset)//100)
                    size_of_train_dataset =      int(((len(filenames)-size_of_no_label_dataset)*percent_of_train_dataset)//100)
                    size_of_test_dataset =       int(((len(filenames)-size_of_no_label_dataset)*percent_of_test_dataset)//100)
                    size_of_validation_dataset = int(((len(filenames)-size_of_no_label_dataset)*percent_of_validation_dataset)//100)

                    for file_to_create_symbolic_link in filenames[0:size_of_no_label_dataset]:
                        os.symlink(class_path+'/'+file_to_create_symbolic_link, self.experiment_folder+'/no_label/no_label/'+file_to_create_symbolic_link)
                    current_index = size_of_no_label_dataset

                    for file_to_create_symbolic_link in filenames[current_index:current_index+size_of_train_dataset]:
                        os.symlink(class_path+'/'+file_to_create_symbolic_link, self.train_dataset_folder+'/'+class_name+'/'+file_to_create_symbolic_link)
                    current_index = size_of_no_label_dataset

                    for file_to_create_symbolic_link in filenames[current_index:current_index+size_of_test_dataset]:
                        os.symlink(class_path+'/'+file_to_create_symbolic_link, self.test_dataset_folder+'/'+class_name+'/'+file_to_create_symbolic_link)
                    current_index = current_index+size_of_test_dataset

                    for file_to_create_symbolic_link in filenames[current_index:-1]:
                        os.symlink(class_path+'/'+file_to_create_symbolic_link, self.validation_dataset_folder+'/'+class_name+'/'+file_to_create_symbolic_link)
                    current_index = size_of_no_label_dataset
