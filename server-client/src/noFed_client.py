import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import  f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import datetime
import os, sys, dill, argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from joblib import load
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier
import csv
import argparse

np.random.seed(42)

class ModelTrainer:
    def __init__(self, clients=1):
        self.model= None
        self.stored_models = []
        self.num_clients = clients
        self.aggregation_data = dict()
        self.client_samples = list()
        self.total_samples = 0
        self.CSV_PATH = './CSV'
        self.DATA_PATH = "./Data/"
        self.TEMP_PATH = './Temp'

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.target_encoder = None
        self.feature_encoders = None

    def load_data(self,dataset_name:str) -> None:
        """
        Loads client data. Concatenates client training sets and uses client test sets separate to model NoFed client behavior.
        Arguments:
            dataset_name {str} -- Name of dataset to load (defines client data to load)
        """
        x_train_set = list()
        y_train_set = list()
        x_test_set = list()
        y_test_set = list()
        for client_id in range(self.num_clients):

            X_train = pd.read_csv(f"{self.DATA_PATH}{dataset_name}_dir/{dataset_name}_client_{client_id}_train.csv", header=0)


            X_test = pd.read_csv(f"{self.DATA_PATH}{dataset_name}_dir/{dataset_name}_client_{client_id}_test.csv", header=0)


            # Sort by index before dropping NA to ensure consistent row ordering
            X_train = X_train.sort_index().dropna()
            X_test = X_test.sort_index().dropna()

            self.y_train = X_train['target'].copy()
            self.y_test = X_test['target'].copy()

            X_train.drop(['target'], axis=1, inplace=True)
            X_test.drop(['target'], axis=1, inplace=True)



            # Prepare target encoder and transform
            self.target_encoder = LabelEncoder()
            # Sort unique values before fitting to ensure consistent encoding
            unique_targets = sorted(self.y_train.unique())
            self.target_encoder.fit(unique_targets)
            self.y_train = self.target_encoder.transform(self.y_train)

            # Transform test data, mapping unknown values to a new class
            self.y_test = np.array([
                self.target_encoder.transform([val])[0] if val in self.target_encoder.classes_
                else -1 for val in self.y_test
            ])

            # Get categorical columns in a deterministic order
            categorical_columns = sorted(X_train.select_dtypes(include=['object', 'category']).columns)

            # Initialize feature encoders dictionary
            self.feature_encoders = {}

            for col in categorical_columns:
                encoder = LabelEncoder()
                # Fit on training data only
                encoder.fit(X_train[col])
                n_classes_feat = len(encoder.classes_)
                self.feature_encoders[col] = encoder

                # Transform training data
                X_train[col] = encoder.transform(X_train[col])
                # Transform test data, mapping unknown values to a new class
                X_test[col] = np.array([
                    encoder.transform([val])[0] if val in encoder.classes_
                    else n_classes_feat for val in X_test[col]
                ])

            self.client_samples.append(X_train.shape[0])
            print(f"Client train samples : {X_train.shape[0]}")
            print(f"Client test samples : {X_test.shape[0]}")
            print(f"Client y train samples: {self.y_train.shape[0]}")
            print(f"Client y test samples: {self.y_test.shape[0]}")

            x_train_set.append(pd.DataFrame(X_train))
            y_train_set.append(pd.DataFrame(self.y_train))
            x_test_set.append(pd.DataFrame(X_test))
            y_test_set.append(pd.DataFrame(self.y_test))

        self.total_samples = sum(self.client_samples)

        self.X_train = pd.concat(x_train_set, axis=0, ignore_index=True)
        self.y_train = pd.concat(y_train_set, axis=0, ignore_index=True)
        self.X_test = x_test_set
        self.y_test = y_test_set


    # Update model with params occured during aggregation
    def updateModel(self):
        """ Update clients' models with aggregated params and return their respective f1s.
        Returns:
            f1s {list} -- f1 score of model trained oer client
        """
        self.model.fit(self.X_train, self.y_train)

        # Get new performance after partial fit
        f1s = self.getModelPerformance()

        return f1s

    def getModelPerformance(self, silent=False):
        """Returns the f1s of the models used
        Arguments:
            silent {bool} -- silent mode
        Returns:
            list -- f1 scores for each client"""

        f1s = list()
        y_pred_full = list()
        y_true_full = list()

        for i in range(self.num_clients):
            y_pred = self.model.predict(self.X_test[i])
            # Calculate F1 score
            f1s.append(f1_score(self.y_test[i], y_pred, average='binary'))
            y_pred_full.append(y_pred)
            y_true_full.append(self.y_test[i])

        y_pred = np.concatenate(y_pred_full)
        y_true = np.concatenate(y_true_full)

        # Print results
        if not silent:
            print(f"-F1 Score: {f1s}")
        # Return respective f1s for each client in list
        return { 'f1' : f1s, 'y_true' : y_true, 'y_pred' : y_pred}


    def update_and_save_data(self, performance_dict, folder_name=None):
        """ Update and save NoFed aggregation data to csv file
        Arguments:
            f1s {list} -- f1 score of model trained per client
            model {str} -- model type
            folder_name {str} -- folder to store final csv file
        """

        #aggregation_f1 = sum(f1s[i] * self.client_samples[i] for i in range(self.num_clients)) / self.total_samples
        aggregation_f1 = f1_score(performance_dict['y_true'], performance_dict['y_pred'], average='binary')

        if folder_name is None:
            folder_name = load(f"{self.TEMP_PATH}/csv_path")

        model = str(folder_name.split('_')[0]) if folder_name is not None else None


        # Ensure the directory exists
        base_path = f"{self.CSV_PATH}/{folder_name}"
        os.makedirs(base_path, exist_ok=True)

        # Save aggregation data to CSV
        nofed_aggregation_filename = f'{model}_nofed_data.csv'
        nofed_aggregation_filename = os.path.join(base_path, nofed_aggregation_filename)
        with open(nofed_aggregation_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['F1 Score'])
            writer.writerow([aggregation_f1])
        print(f'Saved nofed aggregation data to {nofed_aggregation_filename}')




    def create_models(self, combinations, model):
        """Generates models based on combinations and model and saves them to stored_models list
        Arguments:
            combinations {list} -- List of combinations
            model {str} -- Model name
        """
        params = dict()
        if model == "GNB":
            self.model = GaussianNB
        elif model == "SGDC":
            self.model = SGDClassifier
            params['warm_start'] = True
            self.stored_models.append({'warm_start': True})
        elif model == "LogReg":
            self.model = LogisticRegression
            params['warm_start'] = True
            self.stored_models.append({'warm_start': True})
        elif model == "MLPC":
            self.model = MLPClassifier
            params['warm_start'] = True
            self.stored_models.append({'warm_start': True})

        else:
            raise Exception("Model type not supported")

        for param_dict in combinations:
            print(param_dict, type(param_dict))

            for pname, pvalue in param_dict.items():
                params[pname] = pvalue
            self.stored_models.append(params)
        return

    def data_scaler(self):
        """Scales the data for further use"""
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)

        for i in range(len(self.X_test)):

            self.X_test[i] = scaler.transform(self.X_test[i])

        return

def main(num_clients, dataset_name, folder_name):
    trainer = ModelTrainer(num_clients)

    trainer.load_data(dataset_name)

    print("Loading best model")
    trainer.model = load(f"{trainer.TEMP_PATH}/best_model")

    if hasattr(trainer.model, 'random_state'):
        trainer.model.random_state = 42
    #Scale the data
    trainer.data_scaler()


    # Initialize model and fit
    performance_dict = trainer.updateModel()
    trainer.update_and_save_data(performance_dict, folder_name)

    print("Final csv saved to disk...")

parser = argparse.ArgumentParser(description="Specify parameters.")
parser.add_argument('--c', type=int, help='Client number', default=1)
parser.add_argument('--d', type=str, help='Name of the dataset', required=True)
parser.add_argument('--fn', type=str, help='Folder name (where are current fed data stored)', default=None)


args = parser.parse_args()

if __name__ == "__main__":
    """ Valavanis - Argparser to take machine id from terminal params """

    main(args.c, args.d, args.fn)
