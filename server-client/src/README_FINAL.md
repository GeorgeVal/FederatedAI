# Federated Learning Framework

This project implements a federated learning framework with a client-server architecture, supporting multiple machine learning models and secure communication.

## Overview

The framework allows multiple clients to collaboratively train machine learning models while keeping their data private. It supports various ML models including Gaussian Naive Bayes, SGD Classifier, Logistic Regression and Multi-layer Perceptron.

## Requirements

Required Python packages are listed in `requirements.txt`. Main dependencies include:

- Python 3.12.3
- scikit-learn
- pandas
- numpy
- cryptography
- Cython
- matplotlib
- datasets

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Project Structure

### Core Components

- `server.py`: Main server implementation handling model aggregation and client coordination
- `client.py`: Client implementation for local model training and server communication
- `helper.py`: Utility functions for encryption, data handling, and checksums
- `setup.py`: Cython build configuration for C++ components

## C++ TCP Server-Client Module

The framework uses a C++ backbone for efficient network communication, wrapped with Cython for Python integration.

### Net_lib - Common Helper Methods

Core networking utilities for server-client communication including:
- Header construction and parsing
- File transfer operations
- Socket management
- Error handling

### Shared Buffer

Thread-safe buffer implementation for data management:
- `write()`: Append data to buffer
- `read()`: Extract data from buffer
- `size()`: Get current buffer size
- `clear()`: Reset buffer contents
- `getBufferPtr()`: Direct buffer access

### Buffer Manager

Manages multiple SharedBuffer instances for server-side data storage:
- Buffer creation and allocation
- Thread-safe buffer access
- Resource cleanup
- Dynamic buffer management

### Server (TCPServer)

Core server implementation supporting:
- Multi-client connections
- Thread-per-client handling
- Aggregation coordination
- Data synchronization

### Client (TCPClient)

Client-side implementation featuring:
- Server connection management
- File transfer capabilities
- Metadata handling
- Response processing

### Cython Interfaces

Python wrappers for C++ components:
- `PyTCPServer`: Server wrapper
- `PyServerBuffer`: Buffer wrapper
- `PyBufferManager`: Buffer management wrapper
- `PyTCPClient`: Client wrapper

Each wrapper provides Pythonic access to the underlying C++ functionality while maintaining performance.

### Configuration and Data Handling

- `config.json`: Model hyperparameter configurations
- `split_data.py`: Data preprocessing and splitting utility
- `plot_builder.py`: Visualization tools for model performance
- `noFed_client.py`: Non-federated learning implementation for comparison

### Data Directory Structure

```
.
├── Data/           # Training and test data
├── Temp/          # Temporary files
└── CSV/           # Results and metrics
```

## Usage

### 1. Data Preparation

The framework includes a robust data splitting utility (`split_data.py`) that prepares datasets for federated learning scenarios. It supports both CSV files and specific datasets (e.g., adult-census-income).

#### Features:
- Stratified splitting to maintain class distribution
- Support for both balanced and unbalanced data distribution across clients
- Automatic handling of categorical features (one-hot encoding)
- Train-test splitting for each client's portion

#### Usage:
```bash
python split_data.py --n <num_clients> --filename <dataset> --test_size <ratio> --target_col <column_name> [--split_client_portion <proportions>]
```

Parameters:
- `--n`: Number of clients (required)
- `--filename`: Dataset name or path to CSV file (required)
- `--test_size`: Proportion of data for testing (required)
- `--target_col`: Name of the target variable column (required)
- `--split_client_portion`: Optional list of integers defining uneven data distribution

Examples:
1. Equal distribution across 3 clients:
```bash
python split_data.py --n 3 --filename data.csv --test_size 0.2 --target_col label
```

2. Uneven distribution (20%, 30%, 50%):
```bash
python split_data.py --n 3 --filename data.csv --test_size 0.2 --target_col label --split_client_portion 20 30 50
```

Output:
- Creates separate train and test files for each client
- Files are named as `{filename}_client{id}_train.csv` and `{filename}_client{id}_test.csv`
- Preserves data distribution through stratified splitting
- Handles categorical features automatically through one-hot encoding

#### Built-in Dataset Support:
Currently supports:
- adult-census-income (from scikit-learn datasets)
  

### 2. Start Server

```bash
python3 server.py --p <port> --c <num_clients> --m <model_type> --w <weight_mode> --lr <learning-rate>
```

Parameters:
- `--p`: Port number
- `--c`: Number of clients
- `--m`: Model type (GNB/SGDC/LogReg/MLPC)
- `--w`: Weighting mode (w/uw for weighted/unweighted)
- `--i`: Maximum iterations (default: 100)
- `--f`: Forced rounds (default: 20)
- `--combs`: Combinations per feature (default: 1)
- `--lr`: Learning rate for residual path (default:0.1)
### 3. Start Clients

For each client:
```bash
python3 client.py --ip <server_ip> --p <port> --cid <client_id> --d <dataset_name>
```

### 4. Complete Training Cycle and Visualization

For complete analysis and visualization, run the following sequence:

1. Weighted Federated Learning:
```bash
python3 server.py --p <port> --c <num_clients> --m <model_type> --w w --folder <folder_name>
# Start corresponding clients...
```

2. Unweighted Federated Learning:
```bash
python3 server.py --p <port> --c <num_clients> --m <model_type> --w uw --folder <folder_name>
# Start corresponding clients...
```

3. Non-Federated Learning:
```bash
python3 noFed_client.py --c <num_clients> --d <dataset_name> --fn <folder_name>
```

4. Generate Visualization:
```bash
python3 plot_builder.py --c <num_clients> --fn <folder_name>
```

Note: The plot_builder.py script requires data from all three runs (weighted, unweighted, and non-federated) to generate complete comparisons. Make sure to use the same folder_name across all runs.

## Security Features

The framework implements secure communication using:
- RSA encryption for key exchange
- AES encryption for data transmission
- MD5 checksums for data integrity

## Model Types Supported

1. Gaussian Naive Bayes (GNB)
2. Stochastic Gradient Descent Classifier (SGDC)
3. Logistic Regression (LogReg)
4. Multi-layer Perceptron Classifier (MLPC)

## Hyperparameter Tuning

The framework includes automated hyperparameter tuning using k-fold cross-validation. Configuration parameters for each model type are defined in `config.json`.

## Performance Analysis

- Results are saved in CSV format for each client and aggregation
- Performance metrics include F1 score
- Visualization tools for comparing client and aggregated model performance
- Timing information for performance analysis

## Non-Federated Comparison

Use `noFed_client.py` to compare federated learning results with traditional centralized learning:

```bash
python3 noFed_client.py --c <num_clients> --d <dataset_name> --fn <folder_name>
```

## Build Instructions

To build the Cython components:
```bash
python3 setup.py build_ext --inplace
```

## Notes

- Ensure all clients have access to their respective portions of the dataset
- The server must be started before clients attempt to connect
- Check the `CSV/` directory for results and metrics

## References


1. Cryptography Implementation:
   - Python Cryptography Library Documentation: https://cryptography.io/en/latest/
   - GeeksforGeeks RSA Implementation Guide: https://www.geeksforgeeks.org/rsa-algorithm-cryptography/

2. Machine Learning Components:
   - Scikit-learn Documentation: https://scikit-learn.org/
   - GeeksforGeeks Machine Learning Tutorials: https://www.geeksforgeeks.org/machine-learning/

3. Network Programming:
   - Python Socket Programming: https://docs.python.org/3/library/socket.html
   - GeeksforGeeks Socket Programming: https://www.geeksforgeeks.org/socket-programming-python/

4. Data Handling and Visualization:
   - Pandas Documentation: https://pandas.pydata.org/docs/
   - Matplotlib Documentation: https://matplotlib.org/
   - Seaborn Documentation: https://seaborn.pydata.org/

5. Network Programming:
   - Python Socket Programming: https://docs.python.org/3/library/socket.html
   - GeeksforGeeks Socket Programming: https://www.geeksforgeeks.org/socket-programming-python/
   - Socket Programming in C++: https://www.geeksforgeeks.org/socket-programming-in-cpp/
   - Socket Programming CC: https://www.geeksforgeeks.org/socket-programming-cc/
   - Multi-Client Server: https://www.geeksforgeeks.org/socket-programming-in-cc-handling-multiple-clients-on-server-without-multi-threading/
   - C++ Multithreaded Client-Server Example: https://github.com/RedAndBlueEraser/c-multithreaded-client-server/
   - Cython Documentation: https://cython.readthedocs.io/en/latest/src/userguide/index.html
