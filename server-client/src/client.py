# Cython magic:
import time, json
from io import BytesIO
# Cython imports to use C++ infrastructure
from FL_cpp_client import py_fl_client, PyClientBuffer
# Machine Learning client imports
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from joblib import dump, load
from sklearn.neural_network import MLPClassifier
#some parallelism tools for faster hyperparameter tuning
from joblib import Parallel, delayed
from tempfile import mkdtemp
import shutil
import datetime
import argparse
from sklearn.utils.class_weight import compute_sample_weight

# Common helper methods for server and client
from helper import *

#For SVM hyperparameter fine tuning
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

"""
Create client trainer  : trainer = ModelTrainer()
Create Hyper-param Tuner: tuner = HyperparameterTuner()

#fine tune the model in order to get performance over each configuration (K-Fold cross validation)
    f1_list = tuner.hyperparameter_tune_model(3,50, client, trainer)
    
# Aggregation  rounds 
    train / test 
"""

np.random.seed(42)


class ModelTrainer:
    """Manages client instance"""
    def __init__(self, ip_address=None, portnum=None):
        #Main Folder Paths
        self.sample_weights = None
        self.initial_f1 = None
        self.DATA_PATH = "./Data/"
        self.TEMP_PATH = "./Temp/"

        self.model = None

        #Used for client-server communication (Cython)
        self.aggregationBuffer = PyClientBuffer()
        self.paramsBuffer = PyClientBuffer()

        self.aggregationRound = 0
        self.stop_rounds = False
        self.client_private_key = None
        self.client_public_key = None
        self.server_public_key = None
        self.symmetric_key = None
        self.stored_models = []
        self.stop_rounds = False

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.target_encoder = None
        self.feature_encoders = None

        #Used in Hyperparameter fine tuning
        self.X_val = None
        self.y_val = None
        self.X_h_train = None
        self.y_h_train = None

        #Used for server-client communication
        if ip_address is None or portnum is None:
            raise ValueError("Invalid or None values in ip_address and portnum")
        # Init client, with IP, portnum and Cython buffers as parameters
        self.client = py_fl_client(ip_address, portnum, self.paramsBuffer, self.aggregationBuffer)
        # Get client's machine ID
        self.mid = self.client.getMachineID()
        # Participate in Federated Learning using separate C++ thread
        self.client.participate()

    def load_data(self, dataset_name: str, client_id: int) -> None:
        """Loads the client's train and test data.
        Arguments:
            dataset_name {str} -- Name of the dataset to load
            client_id {int} -- ID of the client (determines which part of the data to load)
            """
        # Store dataset name for further usage
        # Check whether directory exists
        os.makedirs(self.TEMP_PATH, exist_ok=True)
        file_path = os.path.join(self.TEMP_PATH, 'dataset_name')
        # Dump the dataset_name to the file
        with open(file_path, 'wb') as file:
            dump(dataset_name, file)

        """ Load data csv file """
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

        self.X_train = X_train
        self.X_test = X_test
        self.sample_weights = compute_sample_weight(class_weight="balanced", y=self.y_train)

    def readParams(self, readKeys=False):
        """ Get new params from the buffer (server filled buffer) """
        while not self.aggregationBuffer.check_complete():
            print("waiting for new params...")
            time.sleep(0.1)
        # Create BytesIO object
        clientBufferIO = BytesIO()
        # Get buffer size for read and write read data to BytesIO object
        buffSize = self.aggregationBuffer.size()
        print(f"Bufsize: {buffSize}")
        clientBufferIO.write(self.aggregationBuffer.read(buffSize))
        # Reset the BytesIO pointer
        aBuffMD5 = get_md5_checksum_from_bytesio(clientBufferIO)
        print("MD5: " + aBuffMD5)
        clientBufferIO.seek(0)
        if not readKeys:
            # Decrypt data
            eBytes = clientBufferIO.getvalue()
            dBytes = receive_and_decrypt(eBytes, self.symmetric_key)
            dBytesIO = BytesIO(dBytes)

            # Load the decrypted and serialized data to memory
            params = dill.load(dBytesIO)
        else:
            # Load the serialized data to memory
            params = dill.load(clientBufferIO)
        #print("-------------------------")
        #print(f"Received params: {params}")
        #print("-------------------------")
        #print("-------------------------")


        return params

    def readSymmetricKeys(self):
        """Read symmetric keys sent from the server """
        while not self.aggregationBuffer.check_complete():
            time.sleep(0.1)
        # Create BytesIO object
        clientBufferIO = BytesIO()
        # Get buffer size for read and write read data to BytesIO object
        buffSize = self.aggregationBuffer.size()
        clientBufferIO.write(self.aggregationBuffer.read(buffSize))
        # Reset the BytesIO pointer
        aBuffMD5 = get_md5_checksum_from_bytesio(clientBufferIO)
        print("MD5: " + aBuffMD5)
        clientBufferIO.seek(0)
        eBytes = clientBufferIO.getvalue()
        return eBytes

    def readServerPublicKey(self):
        """ Read the public server's key"""
        while not self.aggregationBuffer.check_complete():
            print("waiting for new params...")
            time.sleep(0.1)
        # Create BytesIO object
        clientBufferIO = BytesIO()
        # Get buffer size for read and write read data to BytesIO object
        buffSize = self.aggregationBuffer.size()
        clientBufferIO.write(self.aggregationBuffer.read(buffSize))
        # Reset the BytesIO pointer
        aBuffMD5 = get_md5_checksum_from_bytesio(clientBufferIO)
        print("MD5: " + aBuffMD5)
        clientBufferIO.seek(0)
        # Load the serialized data to memory
        params = dill.load(clientBufferIO)
        return params

    def useBestModel(self, params):
        """Use model indicated by the best_index in params
        Arguments:
             params """
        self.model = self.model(**self.stored_models[params["best_index"]])

        if hasattr(self.model, 'random_state'):
            self.model.random_state = 42

        # Check whether directory exists
        os.makedirs(self.TEMP_PATH, exist_ok=True)
        file_path = os.path.join(self.TEMP_PATH, 'best_model')
        # Dump the best model to the Temp folder for future usage
        with open(file_path, 'wb') as file:
            dump(self.model, file)

        #Scale the clients data
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        return

    #Update model with params occured during aggregation
    def updateModel(self, params, X_train=None, X_test=None, y_train=None, y_test=None, model=None):
        """ Update model with the aggregated params sent from the server
        Arguments:
            params {dict} -- params sent from the server
            X_train {np.ndarray} -- training data
            X_test {np.ndarray} -- testing data
            y_train {np.ndarray} -- training labels
            y_test {np.ndarray} -- testing labels
            model {Model} -- model to be updated
        Returns:
            res {dict} -- result of the update
        """
        model = model if model is not None else self.model
        f1 = 0
        i = 0

        #If no data provided use the stored data (normal update phase) else used the data provided (hyperparameter tune phase)
        if X_train is None or y_train is None:
            X_train, y_train, X_test, y_test, fine_tune = self.X_train, self.y_train, self.X_test, self.y_test, False
        else:
            X_train, y_train, X_test, y_test, fine_tune = X_train, y_train, X_test, y_test, True

        if fine_tune:
            initial_f1 = self.getModelPerformance(X_test, y_test, model=model)['f1']
            sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
        else:
            initial_f1 = self.initial_f1
            sample_weights = self.sample_weights

        #Fit the model with the client data. In case of aggregation round and if partial_fit function exist in the model
        if self.aggregationRound != 0 or fine_tune:
            #Update model with new params
            #print(f"Params: {params}")
            for pname in params["model_params"]:
                setattr(model, pname, params["model_params"][pname])

            if hasattr(model, 'partial_fit') and callable(model.partial_fit):
                if fine_tune:
                    if params['model'] != "MLPC":
                         #Train-Update model based on received params
                        model.partial_fit(X_train, y_train, sample_weight=sample_weights)
                        #print("sample weights used")
                    else:
                        # Train-Update model based on received params
                        model.partial_fit(X_train, y_train)
                        #print("sample weights not used")
                else:
                    while f1 <= initial_f1 and i<100:


                        if params['model'] != "MLPC":
                             #Train-Update model based on received params
                            model.partial_fit(X_train, y_train, sample_weight=sample_weights)
                            #print("sample weights used")
                        else:
                            # Train-Update model based on received params
                            model.partial_fit(X_train, y_train)
                            #print("sample weights not used")

                        f1 = self.getModelPerformance(X_test, y_test, model=model)['f1']
                        i+=1

            else:
                model.fit(X_train, y_train)

        else:
            #Train the model without the aggregation parameters
            model.fit(X_train, y_train)
            self.initial_f1 = self.getModelPerformance(X_test, y_test, model=model)['f1']

        # Get message to stop from stop_sig variable
        self.stop_rounds = params["stop_sig"]
        print("Decision to stop: " + str(self.stop_rounds))

        # Get new performance after partial fit
        res = {'test': self.getModelPerformance(X_test, y_test, model=model)}
        if not fine_tune:
            res['train'] = self.getModelPerformance(X_train, y_train, model=model)

        #print(f"Final dict: {res}")

        return res

    def writeParamsBuffer(self, scores, model_type='', state='', model=None):
        """ Write new params to the buffer for the server
        Arguments:
            scores {dict} -- scores sent to the server (train, test scores)
            model_type {str} -- type of the model
            state {str} -- state: model_hp_voting|None
            model {Model} -- model to be updated """
        params = dict()
        params["num_of_samples"] = self.X_train.shape[0]
        params["var_perf"] = 0.0 if 'val' not in scores else scores['val']
        params["train_perf"] = 0.0 if 'train' not in scores else scores['train']
        params["test_perf"] = 0.0 if 'test' not in scores else scores['test']
        params["state"] = state
        params["model"] = model_type

        if model is not None:
            model_to_use = model
        else:
            model_to_use = self.model

        #if voting phase
        if state == 'model_hp_voting':
            pass

        elif model_type == "SGDC" or model_type == "LogReg":

            params["model_params"] = {
                "coef_": model_to_use.coef_.copy(),
                "intercept_": model_to_use.intercept_.copy()
            }
            #print(f"Current SGDC: {params}")
        elif model_type == "MLPC":

            params["model_params"] = {
                "coefs_": model_to_use.coefs_.copy(),
                "intercepts_": model_to_use.intercepts_.copy(),
            }
        elif model_type == "GNB":
            params["model_params"] = {
                "var_": model_to_use.var_.copy(),
                "theta_": model_to_use.theta_.copy()
            }
        elif model_type == '':
            params = {
                "dummy": "dummy params"
            }
        else:
            raise ValueError("Unsupported model")

        pBytes = dill.dumps(params)
        tempByteIO = BytesIO(pBytes)
        print(f"Size of params in bytes: {tempByteIO.getbuffer().nbytes}")
        pBuffMD5 = get_md5_checksum_from_bytesio(tempByteIO)
        print("MD5: " + pBuffMD5)
        tempByteIO.seek(0)
        eBytes = encrypt_and_prepare(pBytes, self.symmetric_key)
        # Write new params to buffer, update MD5, set writing as complete
        self.paramsBuffer.write(eBytes)
        self.paramsBuffer.set_md5(pBuffMD5.encode())
        self.paramsBuffer.set_complete()
        #print(f'Sending params {params}')

    def writeKeyToBuffer(self, dummy=False):
        """ Write client's private key to the buffer
        Arguments:
            dummy : used to determine whether a key or not is sent"""
        # Send key:
        # Write key to buffer, update MD5, set writing as complete
        if dummy:
            dummyBytes = b'1234'
            self.paramsBuffer.write(dummyBytes)
            self.paramsBuffer.set_md5(b'75be2fbcb73cbf391d8bbbdce2ab47c9')
        else:
            # Setup params dictionary
            params = {"key": self.client_public_key}
            pBytes: bytes = dill.dumps(params)
            tempByteIO: BytesIO = BytesIO(pBytes)
            print(f"Size of key in bytes: {tempByteIO.getbuffer().nbytes}")
            pBuffMD5 = get_md5_checksum_from_bytesio(tempByteIO)
            print("Key MD5: " + pBuffMD5)
            tempByteIO.seek(0)
            self.paramsBuffer.write(pBytes)
            self.paramsBuffer.set_md5(pBuffMD5.encode())
        self.paramsBuffer.set_complete()

    def getModelPerformance(self, X, y, model):
        """Returns the f1 of the model used
        Arguments:
            X {array-like} -- test data
            y {array-like} -- test labels
            model {object} -- model to use
        Returns:
            float -- f1 score"""

        # Calculate F1 score
        return { 'f1' : f1_score(y, model.predict(X), average='binary'), 'y_true' : y, 'y_pred' : model.predict(X)}

    def create_models(self, combinations, model):
        """Generates models based on combinations and model and saves them to stored_models list
        Arguments:
            combinations {list} -- List of combinations
            model {str} -- Model name
        """
        params = dict()
        #currently not functional: CHECK SVC
        if model == "GNB":
            self.model = GaussianNB
        elif model == "SGDC":
            self.model = SGDClassifier
            params['warm_start'] = True
            # Add default params (empty dict)
            self.stored_models.append({})
        elif model == "LogReg":
            self.model = LogisticRegression
            params['warm_start'] = True
            # Add default params (empty dict)
            self.stored_models.append({})
        elif model == "MLPC":
            self.model = MLPClassifier
            params['warm_start'] = True
            # Add default params (empty dict)
            self.stored_models.append({})

        else:
            raise Exception("Model type not supported")

        if combinations is not None:
            #create model params (dict) for every model combination in combinations dict
            for param_dict in combinations:
                #print(param_dict, type(param_dict))

                for pname, pvalue in param_dict.items():
                    params[pname] = pvalue
                self.stored_models.append(params)
        return

    def init_encryption(self):
        """
        Initializes encryption by generating rsa keys for client and sharing the public key with the server.
        Receives and stores server public rsa key and symmetric encryption key for future use, enabling encrypted
        server-client communication.
        Returns:
            params : {dict} -- model combinations generated by the server
        """

        # Generate RSA keys
        self.client_private_key, self.client_public_key = generate_rsa_keys()

        # 1. Share public key:
        # Write public key to paramsBuffer
        self.writeKeyToBuffer()
        # Notify C++ that the params are ready
        self.client.setParamsReady(self.stop_rounds)
        # Get server public key from params
        self.server_public_key = self.readParams(True)["key"]
        # Parameters read, clean buffer
        self.aggregationBuffer.clear()

        # 2. Share encrypted symmetric key:
        # Write dummy params so that server can send symmetric key
        self.writeKeyToBuffer(dummy=True)
        # Notify C++ that the params are ready
        self.client.setParamsReady(self.stop_rounds)
        # Receive encrypted symmetric key from server
        encrypted_symmetric_key = self.readSymmetricKeys()
        # Decrypt the symmetric key using client's private key
        self.symmetric_key = decrypt_message_rsa(encrypted_symmetric_key, self.client_private_key)
        # Clear aggregation buffer
        self.aggregationBuffer.clear()
        # Write dummy params and send to server (so that server thread can continue)
        self.writeParamsBuffer(scores={})
        # Notify C++ that the params are ready
        self.client.setParamsReady(self.stop_rounds)
        # Read params from server
        params = self.readParams()
        # Clear buffers
        self.aggregationBuffer.clear()
        return params

    def exec_aggregation(self, model_type, performance):
        """ Aggregates client params until global model convergence (stop_sig == True) or max number of iterations reached
        Arguments:
            model_type {str} -- Type of model to use
            performance {dict} -- Performance metrics
        """

        # Aggregate until convergence
        while not self.stop_rounds:
            # Write new params to paramsBuffer (send weights)
            self.writeParamsBuffer(scores=performance, model_type=model_type)
            # Notify C++ that the params are ready
            self.client.setParamsReady(self.stop_rounds)
            # Get new params from the aggregation buffer and wait if they are not ready
            params = self.readParams()
            # update stop rounds
            self.stop_rounds = params["stop_sig"]
            print("Aggregation Round " + str(self.aggregationRound) + ": Decision to stop loop:" + str(
                self.stop_rounds))
            # Parameters read, no need to keep buffer
            self.aggregationBuffer.clear()
            # Fit the aggregated parameters
            performance = self.updateModel(params)
            # If server closes aggregation notify client thread to exit
            if self.stop_rounds:
                self.client.setParamsReady(self.stop_rounds)
            # Increase aggregation rounds
            self.aggregationRound += 1
        return

    def load_best_model(self, params):
        """Used in unweighted aggregation, skips the hyperparameter tuning phase and directly loads the best model (used in weighted aggregation)
        Arguments:
            params {dict} -- Model parameters
        Returns:
            performance {dict} -- Performance metrics
            params["model"] {str} -- Model type name"""
        self.model = load(f"{self.TEMP_PATH}/best_model")

        if hasattr(self.model, 'random_state'):
            self.model.random_state = 42


        # update stop rounds
        self.stop_rounds = params["stop_sig"]
        print(
            "Round " + str(self.aggregationRound) + ": Decision to stop loop:" + str(self.stop_rounds))
        # Parameters read, no need to keep buffer
        self.aggregationBuffer.clear()

        # Scale the clients data
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        #Initialize model and fit
        performance = self.updateModel(params)
        return performance, params["model"]

    def load_default_model(self, params):
        self.create_models(combinations=None, model=params['model'])
        self.model = self.model()

        # Model-specific random state settings
        if hasattr(self.model, 'random_state'):
            self.model.random_state = 42


        # update stop rounds
        self.stop_rounds = params["stop_sig"]

        # Parameters read, no need to keep buffer
        self.aggregationBuffer.clear()

        # Scale the clients data
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        # Initialize model and fit
        performance = self.updateModel(params)
        return performance, params["model"]


class HyperparameterTuner:
    """ Responsible for tuning hyperparameters of stored models"""

    def __init__(self, trainer, splits, dataset_name, max_models=500):
        self.model = None
        self.model_type = None
        self.trainer = trainer
        self.splits = splits
        self.max_models = max_models
        self.dataset_name = dataset_name

    def model_fine_tune(self, model_combs):
        """Initiates hyperparameter tuning procedure and sends final model scores to server for voting to start
        Arguments:
            model_combs {dict} -- Model combinations generated by the server
        Returns:
            performance {dict} -- Performance metrics
            params["model"] {str} -- Model type name"""


        # create models from model_combs
        self.trainer.create_models(model_combs['combinations'], model_combs['model'])
        """-----------------------------------------------------------------------------------------------------------------
                                                     Hyper-Parameter Fine tuning 
        -----------------------------------------------------------------------------------------------------------------"""
        # hyperparameter fine tune those models
        print("Starting Hyperparameter Tuning")
        self.model_type = model_combs['model']
        #Perform hyperparameter fine tune using stratified kfold
        f1_list = self.k_fold_loop(self.splits, self.max_models)

        # send all model f1s (we send weights later)
        self.trainer.writeParamsBuffer(scores={'val': f1_list}, model_type=model_combs['model'],
                                       state='model_hp_voting')
        # Notify C++ that the params are ready
        self.trainer.client.setParamsReady(self.trainer.stop_rounds)
        #Read new paramaters from the aggregation buffer
        params = self.trainer.readParams()
        self.trainer.aggregationBuffer.clear()

        #Use model chosen by server
        self.trainer.useBestModel(params)


        # update stop rounds
        self.trainer.stop_rounds = params["stop_sig"]
        #print(
        #    "Round " + str(self.trainer.aggregationRound) + ": Decision to stop loop:" + str(self.trainer.stop_rounds))
        # Parameters read, no need to keep buffer
        self.trainer.aggregationBuffer.clear()

        #Initialize model and fit
        performance_dict = self.trainer.updateModel(params)
        return performance_dict, params["model"]

    def k_fold_loop(self, n_splits=5, max_models=50):
        """Hyperparameter tuning all stored models using StratifiedKFold
            Arguments:
                 n_splits {int} -- Number of folds
                 max_models {int} -- Maximum number of models
            Returns:
                f1s {list} -- Validation F1-scores of each model"""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = [0] * len(self.trainer.stored_models)

        X = self.trainer.X_train.to_numpy() if isinstance(self.trainer.X_train, pd.DataFrame) else self.trainer.X_train
        y = self.trainer.y_train.to_numpy() if isinstance(self.trainer.y_train, pd.DataFrame) else self.trainer.y_train

        for train_index, validation_index in skf.split(X, y):
            KFOLD_X_train, KFOLD_X_val = X[train_index], X[validation_index]
            KFOLD_y_train, KFOLD_y_val = y[train_index], y[validation_index]

            scaller = StandardScaler()
            KFOLD_X_train = scaller.fit_transform(KFOLD_X_train)
            KFOLD_X_val = scaller.transform(KFOLD_X_val)

            #limit models used for hyperparameter tune
            model_limiter = 0

            for index, params in enumerate(self.trainer.stored_models):

                if model_limiter >= max_models:
                    break
                model = self.trainer.model(**params)

                # Model-specific random state settings
                if hasattr(self.model, 'random_state'):
                    self.model.random_state = 42

                model.fit(KFOLD_X_train, KFOLD_y_train)

                #aggregate and update model
                self.trainer.writeParamsBuffer(scores={}, model_type=self.model_type, model=model, state='htune_agg')
                # Notify C++ that the params are ready
                self.trainer.client.setParamsReady(self.trainer.stop_rounds)
                # Get new params from the aggregation buffer and wait if they are not ready
                agg_params = self.trainer.readParams()
                self.trainer.aggregationBuffer.clear()
                print("Aggregation Round " + str(self.trainer.aggregationRound) + ": Decision to stop loop:" + str(
                    self.trainer.stop_rounds))
                # Parameters read, no need to keep buffer
                self.trainer.aggregationBuffer.clear()
                # Fit the aggregated parameters
                res = self.trainer.updateModel(params=agg_params, X_train=KFOLD_X_train, X_test=KFOLD_X_val,
                                               y_train=KFOLD_y_train, y_test=KFOLD_y_val,
                                               model=model)

                scores[index] += res['test']['f1'] / n_splits  # Store the f1 scores of the Validation set
                model_limiter += 1
        return scores


def main(ip_address: str, portnum: int, client_id: int, dataset_name: str, splits: int):
    """------------------------------------------------------------------------------------
                            Initialization of main components
    ---------------------------------------------------------------------------------------"""

    # Create client trainer
    trainer = ModelTrainer(ip_address, portnum)
    # Load Clients training data
    trainer.load_data(dataset_name, client_id)

    # Create Hyper-parameter tuner class
    tuner = HyperparameterTuner(trainer=trainer, dataset_name=dataset_name, splits=splits)

    # Enable encrypted server-client communication
    params = trainer.init_encryption()
    """------------------------------------------------------------------------------------
                        Hyper-parameter fine-tuning during the K-Fold cross validation
       ---------------------------------------------------------------------------------------"""
    if params['mode'] == 'w':
        performance_dict, model_type = tuner.model_fine_tune(params)
    elif params['mode'] == 'uw':
        performance_dict, model_type = trainer.load_best_model(params)
    elif params['mode'] is None:
        performance_dict, model_type = trainer.load_default_model(params)
    else:
        raise ValueError('Invalid mode')

    """-----------------------------------------------------------------------------------------------------------------
                                                     Aggregation 
    -----------------------------------------------------------------------------------------------------------------"""

    #Performs aggregation until convergence
    trainer.exec_aggregation(performance=performance_dict, model_type=params['model'])

    # Store model to disk
    dump(trainer.model, f'{trainer.DATA_PATH}/model_{trainer.mid}')
    print("Final model saved to disk...")


parser = argparse.ArgumentParser(description="Specify parameters.")
parser.add_argument('--ip', type=str, help='IP Address', required=True)
parser.add_argument('--p', type=int, help='Port number', required=True)
parser.add_argument('--cid', type=int, help='client_id', required=True)
parser.add_argument('--d', type=str, help='name of the dataset', required=True)
parser.add_argument('--splits', type=int, help='number of folds in hyperparameter tunining phase', default=3)
args = parser.parse_args()

if __name__ == "__main__":
    """ Valavanis - Argparser to take machine id from terminal params """
    main(args.ip, args.p, args.cid, args.d, args.splits)
