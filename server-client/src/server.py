# Cython magic:
from email.policy import default
from FL_cpp_server import py_fl_server, PyServerBuffer, PyBufferManager

import hashlib
import datetime
import os, sys, dill, argparse
import numpy as np
import csv
from scipy.stats import bootstrap
from sklearn.metrics import f1_score
# Common helper methods for server and client
from helper import *

from joblib import dump, load
import threading
import queue
import dill
import time
import random
import json
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

class ServerState:
    """Manages server instance"""
    def __init__(self, num_clients:int, num_of_iterations:int, forceRounds:int, mode:str, portnum:int, combinations:int=1, threshold:int=0.0055, lr:float=0.10):
        self.actual_combinations = None
        self.num_clients = num_clients
        self.num_of_iterations = num_of_iterations
        self.forceRounds = forceRounds

        # Initialize buffers, aggregationDatabuffer has the data to send to a client,
        self.aggregationDatabuffer = PyBufferManager()

        # clientDataBuffers is a manager object that handles one buffer for each client
        self.clientDataBuffers = PyBufferManager()

        self.prevScore = None
        self.newScore = None
        self.local_optimal = 0
        self.stop_rounds = False

        # Public Keys of Clients
        self.client_keys = []
        self.symmetric_keys = []
        self.server_public_key = None
        self.server_private_key = None
        # Performance threshold to stop aggregation rounds
        self.thr = threshold
        #Path that we get model configurations from
        self.config_path = "./config.json"
        #How many combinations to generate per feature
        self.combinations = combinations
        # Plots of total performance and per client perforamnce
        self.client_plots = {}
        self.aggregation_plots = {}
        #Aggregation Round Counter
        self.aggregation_rounds = 1
        #CSV folder
        self.CSV_PATH = './CSV'
        #Temporary files folder
        self.TEMP_PATH = './Temp'
        #Used for creating the plot
        self.client_data = {client_id: {'rounds': [], 'f1s': []} for client_id in range(self.num_clients)}
        self.train_client_data = {client_id: {'rounds': [], 'f1s': []} for client_id in range(self.num_clients)}
        self.w_aggregation_data = {'rounds': [], 'f1s': []}
        self.uw_aggregation_data = {'rounds': [], 'f1s': []}
        self.no_htune_aggregation_data = {'rounds':[], 'f1s':[]}
        #Weighted or unweighted mode
        self.mode = mode
        #Server object used for client-server communication
        # Initialize server object with given parameters
        self.server = py_fl_server(portnum, (num_clients - 1), self.aggregationDatabuffer,
                              self.clientDataBuffers)
        #For MLP
        self.global_coefs_ = None
        self.global_intercepts_ = None
        self.global_params = list()
        self.residual_path = True
        self.LR = lr
        # Start server listening thread
        self.server.run()

    def read_params(self, readKeys=False):
        """ Read params from clients' buffers
        Arguments:
            readKeys: {bool} -- Used to determine whether decryption is needed"""
        # Init parameters
        params = []
        for mid in range(self.num_clients):
            # Check if buffer data writing is complete, else wait for it to finish
            while not self.clientDataBuffers.check_buffer_complete(mid):
                time.sleep(0.1)
            clientBufferIO = BytesIO()
            buffSize = self.clientDataBuffers.get_buffer_size(mid)
            # print(f"Client {mid} buffer size: {buffSize}.")
            clientBufferIO.write(self.clientDataBuffers.get_buffer(mid, buffSize))
            # Reset the BytesIO pointer
            clientBufferIO.seek(0)
            if not readKeys:
                # Decrypt data
                eBytes = clientBufferIO.getvalue()
                dBytes = receive_and_decrypt(eBytes, self.symmetric_keys[mid])
                dBytesIO = BytesIO(dBytes)
                params.append(dill.load(dBytesIO))
            else:
                params.append(dill.load(clientBufferIO))
        #print(f"read params: {params}")
        return params

    def send_params(self, params):
        """ Send params to clients
        Arguments:
            params {dict} -- Params to send"""
        #print(f"Sending params: {params}")
        # Reset the aggregation data buffer before writing
        self.aggregationDatabuffer.clear_all_buffers()

        aggregatedBytes = dill.dumps(params)
        tempByteIO = BytesIO(aggregatedBytes)
        print("Aggregated buffer MD5: " + get_md5_checksum_from_bytesio(tempByteIO))
        tempByteIO.seek(0)

        # Send new params to clients
        for mid in range(self.num_clients):
            # Encrypt using client's public key
            eBytes = encrypt_and_prepare(aggregatedBytes, self.symmetric_keys[mid])
            self.aggregationDatabuffer.set_buffer(mid, eBytes)
            self.aggregationDatabuffer.set_md5(mid, get_md5_checksum_from_bytesio(tempByteIO).encode())

        print(f"Size of params in bytes: {tempByteIO.getbuffer().nbytes}")
        self.clientDataBuffers.clear_all_buffers()

    def compute_total_f1(self, params):
        y_pred_full = list()
        y_true_full = list()

        for i in range(len(params)):
            y_pred_full.append(params[i]['test_perf']['y_pred'])
            y_true_full.append(params[i]['test_perf']['y_true'])
        y_pred = np.concatenate(y_pred_full)
        y_true = np.concatenate(y_true_full)

        return f1_score(y_true, y_pred, average='binary')


    def aggregate_model(self, params, csv_folder):
        """Finds which model configuration is the best on model_hp_voring phase. Then, aggregates model params received from clients.
        Arguments:
            params {dict} -- client params (scores of models if model_hp_voring phase else model params for aggregation )
            csv_folder {str} -- path to folder where model params are saved (optional)
            """

        total_samples = sum(params[i]["num_of_samples"] for i in range(len(params)))

        aggregated_params = dict()

        # Check if it is voting phase
        if params[0]["state"] == 'model_hp_voting':
            # voting, each vote has weight analogous to num of samples
            votes = list()
            for param in params:
                for index, score in enumerate(param["var_perf"]):
                    if index >= len(votes):
                            votes.append(score * (param["num_of_samples"] / total_samples))

                    else:
                            votes[index] = score * (param["num_of_samples"] / total_samples)

            # find best model:
            max_vote = max(votes)
            max_index = votes.index(max_vote)
            print(f"votes: {votes} max index vote: {max_index}")

            aggregated_params["best_index"] = max_index

        #Aggregation phase
        else:

            # for i in range(len(params)):
            #     print(f"Client: {i}, test_performance: {params[i]['test_perf']}"
            #           f", number of samples: {params[i]['num_of_samples']}"
            #           f" total_samples: {total_samples}")

            # find total f1 (weighted or unweighted)
            self.newScore = self.compute_total_f1(params) if params[0]["state"] != 'htune_agg' else 0.0
            # self.newScore = sum(
            #     params[i]["test_perf"] * params[i]["num_of_samples"] for i in range(len(params))) / total_samples \
            #     if self.mode == 'w' or self.mode is None else sum(params[i]["test_perf"] for i in range(len(params))) / len(params)


            #update local optimal
            if self.newScore > self.local_optimal and self.newScore != 0:
                self.local_optimal = self.newScore

            # Check whether the aggregation should continue
            self.stop_rounds = (self.aggregation_rounds >= self.num_of_iterations) or (
                    self.aggregation_rounds >= self.forceRounds)

            # Update plot with new weighted f1
            if params[0]['state'] != 'htune_agg':
                self.update_data(params)

            # Set previous score as current, for the next comparison
            self.prevScore = self.newScore

            if self.mode == 'w' or self.mode is None:
                print("Weighted average score: " + str(self.newScore))
            else:
                print("Unweighted average score: " + str(self.newScore))

            aggregated_params["model_params"] = dict()

            #iterations counter
            iter = 0
            #------------------------------------_#
            # MLPC : LR=0.10
            # SGDC : LR =0.25 , LR=0.30, LR=0.5 works too
            # LogReg : LR = 0.30 exponentially decaying after convergence
            #for every param check whether the param value types are supported and then aggregate
            for param_name, param_value in params[0]["model_params"].items():
                #print(f"Param: {param_name} Type: {type(param_value)}")
                if isinstance(param_value, np.ndarray) or isinstance(param_value, float) or isinstance(param_value, int):

                    param_name_avg = sum(params[i]["model_params"][param_name] * params[i]["num_of_samples"] for i in range(len(params))) / total_samples \
                    if self.mode == 'w' or self.mode is None else sum(params[i]["model_params"][param_name] for i in range(len(params))) / len(params)

                    if params[0]['state'] != 'htune_agg' and self.LR > 0 and self.mode is not None:
                        if self.global_params is not None and len(self.global_params) == len(params[0]["model_params"]):
                            #print(f"Update of global params iteration: {iter}, Global params: {self.global_params}")
                            self.global_params[iter] = param_name_avg * self.LR + self.global_params[iter]  # / 2
                        else:
                            #print(f"Initialization of global params iteration: {iter}, Global params: {self.global_params}")
                            self.global_params.append(param_name_avg)


                        aggregated_params["model_params"][param_name] = self.global_params[iter]
                        iter+=1

                    if self.LR <= 0 or self.mode is None:
                        aggregated_params["model_params"][param_name] = param_name_avg
                        #print(f"No use of global params mode {self.mode}")


                elif isinstance(param_value, list) and (isinstance(param_value[0], np.ndarray)):
                    # Initialize the param with zeros of the same shape
                    param_name_avg = [np.zeros_like(params[0]["model_params"][param_name][layer]) for layer in
                                      range(len(params[0]["model_params"][param_name]))]

                    # Sum the params for each layer
                    for i in range(len(params)):
                        if self.mode == 'w' or self.mode is None:
                            weight = params[i]['num_of_samples'] / total_samples
                        else:
                            weight = 1/len(params)

                        # Sum params layer by layer
                        for layer in range(len(params[i]["model_params"][param_name])):
                                param_name_avg[layer] += weight * params[i]["model_params"][param_name][layer]

                    if params[0]['state'] != 'htune_agg' and self.LR > 0 and self.mode is not None:
                        if self.global_params is not None and len(self.global_params) == len(params[0]["model_params"]):
                            #print(f"Update of global params iteration: {iter}, Global params: {self.global_params}")
                            for layer in range(len(param_name_avg)):
                                self.global_params[iter][layer] = param_name_avg[layer] * self.LR +\
                                                    self.global_params[iter][layer]  # / 2
                        else:
                            #print(f"Initialization of global params iteration: {iter}, Global params: {self.global_params}")
                            self.global_params.append(param_name_avg)



                        aggregated_params["model_params"][param_name] = self.global_params[iter]
                        iter+=1
                    if self.LR <= 0 or self.mode is None:
                        aggregated_params["model_params"][param_name] = param_name_avg
                        #print(f"No use of global params mode {self.mode}")
                else:
                    raise ValueError("Unsupported model param type")


        aggregated_params["stop_sig"] = self.stop_rounds
        aggregated_params["model"] = params[0]['model']


        #print(f"Aggregated params :{aggregated_params}")
        print("Decision to stop: " + str(self.stop_rounds))
        #If aggregation has come to an end save the final plot
        if self.stop_rounds:
            self.save_data_to_csv(aggregated_params['model'], csv_folder)


        #Send params to client
        self.send_params(aggregated_params)


    def send_public_key(self, max_id):
        """Send public key to clients THERE IS NO ENCRYPTION HERE"""
        # Get access to aggregation buffer
        # Reset the aggregation data buffer before writing
        self.aggregationDatabuffer.clear_all_buffers()
        params = {"key": self.server_public_key}
        aggregatedBytes = dill.dumps(params)
        tempByteIO = BytesIO(aggregatedBytes)
        print("Key buffer MD5: " + get_md5_checksum_from_bytesio(tempByteIO))
        tempByteIO.seek(0)
        for mid in range(max_id):
            self.aggregationDatabuffer.set_buffer(mid, aggregatedBytes)
            self.aggregationDatabuffer.set_md5(mid, get_md5_checksum_from_bytesio(tempByteIO).encode())
        print(f"Size of key in bytes: {tempByteIO.getbuffer().nbytes}")

    def send_symmetric_keys(self):
        """Send symmetric keys to clients that are encrypted with clients public keys"""
        # Reset the aggregation data buffer before writing
        self.aggregationDatabuffer.clear_all_buffers()
        for mid in range(self.num_clients):

            # Encrypt the symmetric key
            eBytes = encrypt_message_rsa(self.symmetric_keys[mid], self.client_keys[mid])

            self.aggregationDatabuffer.set_buffer(mid, eBytes)


    def checkScoreDiff(self):
        """Check whether threshold is broken
        Returns:
            False: |newScore - prevScore| > thr
            True: |newScore - prevScore| < thr"""
        if self.newScore is None or self.prevScore is None:
            return False
        else:
            #Do we want the abs??
            return abs(self.newScore - self.prevScore) < self.thr

    def read_config_params(self, modeltype):
        """Reads model configurations from json"""
        with open(self.config_path, 'r') as file:
            params = json.load(file)
        try:
            return params[modeltype]
        except KeyError:
            raise KeyError(f"{modeltype} not found as key in config")

    def generate_combinations(self, config_params, model):
        """Generates random model combinations
        Arguments:
            config_params {dict} -- Dictionary of model parameters
            model -- model name"""


        if model == "SGDC":
            return self.generate_SGDC_combinations(config_params)
        elif model == "LogReg":
            return self.generate_LogReg_combinations(config_params)
        elif model == "MLPC":
            return self.generate_MLPC_combinations(config_params)
        elif model == "GNB":
            return self.generate_GNB_combinations(config_params)
        else:
            raise ValueError(f"Model {model} not implemented")


    def generate_GNB_combinations(self, params):
        """Generates random GNB combinations from config.json. Generates self.combinations for every var_smoothing value.
        Arguments:
            params {dict} -- Dictionary of GNB parameters
        Returns:
            combinations {dict} -- Dictionary of GNB combinations
            {str} -- Model type name"""
        combinations = list()

        for var_smoothing in params['var_smoothing']:

            selected_combinations = set()

            while len(selected_combinations) < self.combinations:
                random_combination = {
                    "var_smoothing": var_smoothing
                }
                selected_combinations.add(frozenset(random_combination.items()))

            # Convert frozensets back to dicts for final output
            combinations.extend(dict(combination) for combination in selected_combinations)

        return combinations, "GNB"

    def generate_SGDC_combinations(self, params):
        """Generates random SGDC combinations from config.json. Generates self.combinations for every loss type.
        Arguments:
            params {dict} -- Dictionary of SGDC parameters
        Returns:
            combinations {dict} -- Dictionary of SGDC combinations
            {str} -- Model type name"""
        combinations = list()


        for loss in params['loss']:

            selected_combinations = set()  # Use set to avoid duplicates

            while len(selected_combinations) < self.combinations:
                random_combination = {
                    "penalty": random.choice(params['penalty']),
                    "alpha": random.choice(params['alpha']),
                    "l1_ratio": random.choice(params['l1_ratio']),
                    "fit_intercept": random.choice(params['fit_intercept']),
                    "max_iter": random.choice(params['max_iter']),
                    "tol": random.choice(params['tol']),
                    "shuffle": random.choice(params['shuffle']),
                    "epsilon": random.choice(params['epsilon']),
                    "learning_rate": random.choice(params['learning_rate']),
                    "average": random.choice(params['average']),
                    "eta0": random.choice(params['eta0']),
                    "loss": loss,
                    "early_stopping": random.choice(params['early_stopping'])
                }

                # Convert dict to frozenset of items for uniqueness
                selected_combinations.add(frozenset(random_combination.items()))

            # Convert frozensets back to dicts for final output
            combinations.extend(dict(combination) for combination in selected_combinations)

        return combinations, "SGDC"


    def generate_LogReg_combinations(self, params):
        """Generates random LogReg combinations from config.json. Generates self.combinations for every solver type.
        Arguments:
            params {dict} -- Dictionary of LogReg parameters
        Returns:
            combinations {dict} -- Dictionary of LogReg combinations
            {str} -- Model type name"""
        combinations = list()


        for solver in params['solver']:

            selected_combinations = set()  # Use set to avoid duplicates

            while len(selected_combinations) < self.combinations:

                if len(selected_combinations) >= 18 and solver == "lbfgs":
                    break
                elif len(selected_combinations) >= 45 and solver == "saga":
                    break

                if solver == 'saga':
                    allowed_penalty = params['penalty']
                else:
                    allowed_penalty = [None, "l2"]


                penalty = random.choice(allowed_penalty)

                if penalty != 'elasticnet':
                    l1_ratio = None
                else:
                    l1_ratio = random.choice(params['l1_ratio'])

                random_combination = {
                    "penalty": penalty,
                    "l1_ratio": l1_ratio,
                    "fit_intercept": random.choice(params['fit_intercept']),
                    "max_iter": random.choice(params['max_iter']),
                    "tol": random.choice(params['tol']),
                    "C": random.choice(params['C']),
                    "intercept_scaling": random.choice(params['intercept_scaling']),
                    "class_weight": random.choice(params['class_weight']),
                    "solver" : solver
                }
                selected_combinations.add(frozenset(random_combination.items()))

            # Convert frozensets back to dicts for final output
            combinations.extend(dict(combination) for combination in selected_combinations)

        return combinations, "LogReg"

    def generate_MLPC_combinations(self, params):
        """Generates random MLPC combinations from config.json. Generates self.combinations for every activation type.
        Arguments:
            params {dict} -- Dictionary of MLPC parameters
        Returns:
            combinations {dict} -- Dictionary of MLPC combinations
            {str} -- Model type name"""
        combinations = list()


        for activation in params['activation']:
            selected_combinations = set()

            while len(selected_combinations) < self.combinations:
                random_combination = {
                    "hidden_layer_sizes": tuple(random.choice(params['hidden_layer_sizes'])),
                    "solver": random.choice(params['solver']),
                    "alpha": random.choice(params['alpha']),
                    "learning_rate": random.choice(params['learning_rate']),
                    "learning_rate_init": random.choice(params['learning_rate_init']),
                    "power_t": random.choice(params['power_t']),
                    "max_iter": random.choice(params['max_iter']),
                    "shuffle": random.choice(params['shuffle']),
                    "tol": random.choice(params['tol']),
                    "momentum": random.choice(params['momentum']),
                    "nesterovs_momentum": random.choice(params['nesterovs_momentum']),
                    "validation_fraction": random.choice(params['validation_fraction']),
                    "max_fun": random.choice(params['max_fun']),
                    "beta_1": random.choice(params['beta_1']),
                    "beta_2": random.choice(params['beta_2']),
                    "epsilon": random.choice(params['epsilon']),
                    "activation": activation
                }
                selected_combinations.add(frozenset(random_combination.items()))

            # Convert frozensets back to dicts for final output
            combinations.extend(dict(combination) for combination in selected_combinations)

        return combinations, "MLPC"

    def send_model_params(self, combinations, model):
        """Sends model combinations to client
        Arguments:
            combinations {dict} -- Dictionary of model combinations
            model {str} -- Model name
        """
        params = dict()
        params['combinations'] = combinations
        params['model'] = model
        params['mode'] = 'w'


        self.send_params(params)

        self.server.aggregationDone(False)


    def update_data(self, params):
        """
        Updates client and aggregation data (f1 per round)
        Arguments:
            params {dict} -- Dictionary of model parameters of clients
        """
        round_num = self.aggregation_rounds
        self.aggregation_rounds += 1

        if self.mode == 'w':
            # Update each client's data
            for client_id in range(self.num_clients):
                self.client_data[client_id]['rounds'].append(round_num)
                self.client_data[client_id]['f1s'].append(params[client_id]['test_perf']['f1'])

                self.train_client_data[client_id]['rounds'].append(round_num)
                self.train_client_data[client_id]['f1s'].append(params[client_id]['train_perf']['f1'])

            # Update weighted aggregation data
            self.w_aggregation_data['rounds'].append(round_num)
            self.w_aggregation_data['f1s'].append(self.newScore)
        elif self.mode == 'uw':
            # Update unweighted aggregation data
            self.uw_aggregation_data['rounds'].append(round_num)
            self.uw_aggregation_data['f1s'].append(self.newScore)
        elif self.mode is None:
            self.no_htune_aggregation_data['rounds'].append(round_num)
            self.no_htune_aggregation_data['f1s'].append(self.newScore)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def save_data_to_csv(self,model, csv_folder=None):
        """
        Saves all stored data for each client and the aggregations to CSV files.
        Arguments:
              model {str} -- Model name
              csv_folder {str} -- Folder where to save data (optional)
        """
        if self.mode == 'w':
            # Ensure the directory exists
            base_path = f'{self.CSV_PATH}/{model}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'

            # Extract the part after self.CSV_PATH
            relative_path = os.path.relpath(base_path, self.CSV_PATH)

            file_path = os.path.join(self.TEMP_PATH, 'csv_path')
            # Dump the dictionary to the file
            with open(file_path, 'wb') as file:
                dump(relative_path, file)
            print(f"relative_path: {relative_path}")


        else:

            #If csv_folder is defined use it else use the previously stored csv_path
            if csv_folder is not None:
                base_path = f'{self.CSV_PATH}/{csv_folder}'
            else:
                base_path = f'{self.CSV_PATH}/{load(f"{self.TEMP_PATH}/csv_path")}'

        #if folder does not exist, create it before saving the data
        os.makedirs(base_path, exist_ok=True)

        log_filename = f'log.txt'
        log_filepath = os.path.join(base_path, log_filename)

        best_model_path = f"{self.TEMP_PATH}/best_model"

        if os.path.exists(best_model_path) and self.mode is not None:
            best_model = load(best_model_path)
        else:
            best_model = None

        if not os.path.exists(log_filepath):
            with open(log_filepath, "w") as file:
                file.write(f"Dataset: {load(f'{self.TEMP_PATH}/dataset_name')}\n"
                           f"Clients: {self.num_clients}\n"
                           f"Iterations arg: {self.num_of_iterations}\n"
                           f"Actual aggregation rounds number: {self.aggregation_rounds}\n"
                           f"Model Name: {model}\n"
                           f"LR: {self.LR}\n"
                           f"Actual combinations generated: {self.actual_combinations}\n"
                           # f"Best model: {vars(load(f"{self.TEMP_PATH}/best_model"))}\n"
                           f"Combinations generated: {self.combinations}\n")
                           #f"Combinations: {load(f'{self.TEMP_PATH}/combinations')}\n")
                if best_model is not None:
                    file.write(f"Best model: {vars(best_model)}\n")

        # Save clients and aggregation data to CSV
        if self.mode == 'w':
            # Save client data to CSV (train and test f1s)
            for client_id in range(self.num_clients):
                client_filename = f'{model}_client_{client_id}_data.csv'
                client_filepath = os.path.join(base_path, client_filename)
                with open(client_filepath, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Round', 'F1 Score'])
                    writer.writerows(zip(self.client_data[client_id]['rounds'], self.client_data[client_id]['f1s']))
                print(f'Saved client {client_id} data to {client_filename}')

                train_client_filename = f'{model}_train_client_{client_id}_data.csv'
                train_client_filepath = os.path.join(base_path, train_client_filename)
                with open(train_client_filepath, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Round', 'F1 Score'])
                    writer.writerows(
                        zip(self.train_client_data[client_id]['rounds'], self.train_client_data[client_id]['f1s']))
                print(f'Saved train client {client_id} data to {train_client_filename}')

            #Save aggregation data to CSV (weighted or unweighted)
            w_aggregation_filename = f'{model}_weighted_aggregation_data.csv'
            w_aggregation_filepath = os.path.join(base_path, w_aggregation_filename)
            with open(w_aggregation_filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Round', 'F1 Score'])
                writer.writerows(zip(self.w_aggregation_data['rounds'], self.w_aggregation_data['f1s']))
            print(f'Saved weighted aggregation data to {w_aggregation_filename}')
        elif self.mode == 'uw':
            uw_aggregation_filename = f'{model}_unweighted_aggregation_data.csv'
            uw_aggregation_filepath = os.path.join(base_path, uw_aggregation_filename)
            with open(uw_aggregation_filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Round', 'F1 Score'])
                writer.writerows(zip(self.uw_aggregation_data['rounds'], self.uw_aggregation_data['f1s']))
            print(f'Saved unweighted aggregation data to {uw_aggregation_filename}')
        elif self.mode is None:
            nohtune_aggregation_filename = f'{model}_notune_aggregation_data.csv'
            nohtune_aggregation_filepath = os.path.join(base_path, nohtune_aggregation_filename)
            with open(nohtune_aggregation_filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Round', 'F1 Score'])
                writer.writerows(zip(self.no_htune_aggregation_data['rounds'], self.no_htune_aggregation_data['f1s']))
            print(f'Saved notune aggregation data to {nohtune_aggregation_filename}')
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        optimal_filename = f'{model}_{self.mode}_optimal.csv'
        optimal_filepath = os.path.join(base_path, optimal_filename)
        with open(optimal_filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([self.local_optimal])

    def init_encryption(self):
        """
        Shares server rsa public key and encrypted symetric keys with clients
        """
        # Encryption:
        # 1. Public key exchange:
        # Generate RSA keys
        self.server_private_key, self.server_public_key = generate_rsa_keys()
        # Receive client key/secret in params
        client_keys_dict = self.read_params(True)
        self.client_keys = [entry['key'] for entry in client_keys_dict]
        # Send key/secret to client as aggregated data
        self.send_public_key(self.num_clients)
        # Clear key buffers, in order to reuse on next iteration
        self.clientDataBuffers.clear_all_buffers()
        self.server.aggregationDone(False)


        # 2. Generate and send encrypted symmetric keys:
        # Dummy receive
        for mid in range(self.num_clients):
            while not self.clientDataBuffers.check_buffer_complete(mid):
                time.sleep(0.1)

        self.symmetric_keys = [generate_aes_key() for _ in range(self.num_clients)]
        #print(f"Symmetric Key: {self.symmetric_keys}")
        # Send key/secret to client as aggregated data
        self.send_symmetric_keys()
        # Clear key buffers, in order to reuse on next iteration
        self.clientDataBuffers.clear_all_buffers()
        self.server.aggregationDone(False)

        self.read_params()
        return

    def send_model_combinations(self, model_type:str) -> None:
        """Generates random model combinations (of type model_type) for clients by reading config.json file
        Arguments: model_type {str} -- Type of model
        """
        # read config file
        config_params = self.read_config_params(model_type)
        # generate combinations
        #print(f"config_params: {config_params}")
        combinations, model = self.generate_combinations(config_params, model_type)
        self.actual_combinations = len(combinations)
        combinations_dict = {"combinations": combinations, "model": model}

        #Check whether directory exists
        os.makedirs(self.TEMP_PATH, exist_ok=True)
        file_path = os.path.join(self.TEMP_PATH, 'combinations')
        # Dump the dictionary to the file
        with open(file_path, 'wb') as file:
            dump(combinations_dict, file)
        #print(f"combinations: {combinations}")

        self.send_model_params(combinations, model)

        return

    def exec_aggregation(self, csv_folder):
        """
        Reads params from clients and processes them as needed (aggregation or voting phase)
        Arguments:
            csv_folder {str} -- Folder where csv files will be saved (optional)
        """

        # Read buffers to get params
        params = self.read_params()
        # Aggregate the given parameters
        self.aggregate_model(params, csv_folder)
        # Notify the handler threads that the aggregated data is ready to be sent to the client
        self.server.aggregationDone(self.stop_rounds)


    def final_check(self):
        """Checks why aggregation stopped and outputs the reason"""
        if self.aggregation_rounds >= self.num_of_iterations:
            print("Stopped due to iteration limit reached")

        if self.aggregation_rounds >= self.forceRounds:
            print("Stopped due to force rounds reached")

def main(portnum:int=8080, numOfClients:int=3, numOfIterations:int=100, forceRounds:int=20, modelType:str='SGDC', mode:str='w', csv_folder:str=None, combinations:int=1, LR:float=0.10):

    server_state = ServerState(numOfClients, numOfIterations, forceRounds, mode, portnum, combinations,lr=LR)
    print("Starting server at:", datetime.datetime.now().strftime("%H:%M:%S"))

    #Enable encrypted server-client communication
    server_state.init_encryption()

    if server_state.mode == 'w':
        #Send random generated model combinations to clients
        server_state.send_model_combinations(modelType)
    elif server_state.mode == 'uw':

        server_state.send_params({'mode' : 'uw', 'model': modelType, 'stop_sig': False})
        server_state.server.aggregationDone(False)
    elif server_state.mode is None:

        server_state.send_params({'mode' : None, 'model': modelType, 'stop_sig': False})
        server_state.server.aggregationDone(False)
    else:
        raise ValueError(f"Invalid mode: {server_state.mode}")

    # Loop the process if stop_rounds is False (includes voting and aggregation phase)
    while not server_state.stop_rounds:

        # Either aggregates or votes as needed (aggregation or voting phase)
        server_state.exec_aggregation(csv_folder)

    # Check why the aggregation stopped
    server_state.final_check()

    # Wait for all the processes to complete and then terminate
    server_state.server.allDone()


if __name__ == "__main__":
    """ Valavanis - Argparser to take machine id from terminal params """
    parser = argparse.ArgumentParser(description="Give model parameters.")
    parser.add_argument('--p', type=int, help='Port number', required=True)
    parser.add_argument('--c', type=int, help='Number of Clients' , required=True)
    parser.add_argument('--i', type=int, help='Number of max Iterations', default=100)
    parser.add_argument('--f', type=int, help='Number of Forced Rounds', default=20)
    parser.add_argument('--m', type=str, choices=['GNB','SGDC','LogReg','MLPC'], help='Model Type', default='SGDC')
    parser.add_argument('--w', type=str, help=' w for Weighted ,uw for Unweighted, None to turn off htune.', choices=['w','uw',None], default=None)
    parser.add_argument('--folder', type=str, help='CSV Folder name (Where will you save the data csv\'s)', default=None)
    parser.add_argument('--combs', type=int, help='How many combinations per target feature to generate', default=1)
    parser.add_argument('--lr', type=float, help='Learning Rate for Residual Path', default=0.10)
    args = parser.parse_args()
    # Run main function with given args
    main(portnum=args.p, numOfClients=args.c, numOfIterations=args.i, forceRounds=args.f, modelType=args.m, mode=args.w, csv_folder=args.folder, combinations=args.combs, LR=args.lr)
