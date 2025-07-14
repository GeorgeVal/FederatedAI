
from sklearn.datasets import make_blobs
from sklearn.model_selection import StratifiedKFold
import argparse
import pandas as pd 
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "./Data/"

"""Create the dataset and split this data into n portions, where n is the number of clients
(n-splits) = n or read CSV file and split it towards n clients
"""
def create_data(csv, n_splits, samples=0, centroids=0, variance=0, features=0 ):
    if csv==0:
        """Create the blobs dataset"""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        X, y = make_blobs(n_samples=samples, centers=centroids, cluster_std=variance, n_features=features)

        for i, (_, test_index) in enumerate(skf.split(X, y)):
            # Create DataFrames for features and target
            fold_X = pd.DataFrame(X[test_index], columns=[f'feature_{j}' for j in range(features)])
            fold_Y = pd.Series(y[test_index], name='target')

            # Merge X and Y
            fold_data = pd.concat([fold_X, fold_Y], axis=1)

            # Save to CSV
            # Check if the folder exists
            if not os.path.exists(DATA_PATH):
                # If it doesn't exist, create it
                os.makedirs(DATA_PATH)
            fold_data.to_csv(f"{DATA_PATH}data_portion_{i}.csv", header=True, index=False)
    else:
        #Find the number of rows in the  data_portion_0
        if csv == -1:
            filename = "data_portion_0"
            print(f"Finding length of {filename}.csv")
            df = pd.read_csv(f"{DATA_PATH}{filename}.csv")
            print(f"CSV has length of {len(df)}")
            return
        #big csv with separate file for label y, tailored for ARP_MitM_dataset.csv
        if csv == 1:
            print("Reading Big CSV files")
            # Step 1: Read the dataset in chunks
            chunk_size = 100000  # Adjust this value based on your memory limits

            # Initialize empty lists to store data
            X_chunks = []
            y_chunks = []

            # Load the dataset and labels in chunks (assuming no header)
            for chunk in pd.read_csv(f"{DATA_PATH}ARP_MitM_dataset.csv", header=None, chunksize=chunk_size):
                X_chunks.append(chunk)

            for chunk in pd.read_csv(f"{DATA_PATH}ARP_MitM_labels.csv", header=None, chunksize=chunk_size):
                y_chunks.append(chunk.iloc[:, 1])  # Extract label column
            print("Done with those chunks")

            # Concatenate all chunks into a single DataFrame and Series
            X = pd.concat(X_chunks, ignore_index=True)
            y = pd.concat(y_chunks, ignore_index=True)

            # Ensure labels in 'y' are converted to integers (or the appropriate type)
            y = pd.to_numeric(y, errors='coerce')  # Convert to numeric, setting invalid parsing as NaN

            # Drop any rows where 'y' is NaN, and reindex both X and y accordingly
            valid_indices = y.notna()  # Get a boolean mask for valid (non-NaN) rows
            X = X.loc[valid_indices].reset_index(drop=True)  # Keep only valid rows in X
            y = y.loc[valid_indices].reset_index(drop=True)  # Keep only valid rows in y


            #Drop the last row
            y = y.iloc[:-1]

            # # Normalize and scale the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Convert the scaled data back to a DataFrame
            X = pd.DataFrame(X_scaled, columns=X.columns)

        #small single csv with header
        else:
            print("Reading small single preprocessed CSV file")
            # Step 1: Read the entire CSV into a DataFrame
            dataset_name = "heart_disease_preprocessed"
            df = pd.read_csv(f"{DATA_PATH}{dataset_name}.csv")

            # # Columns where a value of 0 is implausible and should be considered as missing
            # zero_impute_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
            #
            # # Replace zero values with NaN (to later handle missing values)
            # df[zero_impute_columns] = df[zero_impute_columns].replace(0, np.nan)
            #
            # # Fill NaN values with the median of each column to handle missing data
            # df[zero_impute_columns] = df[zero_impute_columns].fillna(df[zero_impute_columns].median())

            # Separate features (X) and target (y)
            X = df.drop('HeartDisease', axis=1)
            y = df['HeartDisease']


            # Normalize and scale the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Convert the scaled data back to a DataFrame
            X = pd.DataFrame(X_scaled, columns=X.columns)

        '''If number of splits are 3 then break data into 3 parts containing 50% 30% 20% of samples respectively'''
        if n_splits == 3:
            dataset_name = "heart_disease_preprocessed"
            # Split the data into 3 parts while maintaining stratification
            X_temp, X_train_2, y_temp, y_train_2 = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
            X_train_0, X_train_1, y_train_0, y_train_1 = train_test_split(X_temp, y_temp, test_size=0.375, stratify=y_temp,
                                                              random_state=42)  # 0.375 * 0.8 = 0.3

            X_train_0, X_test_0, y_train_0,  y_test_0 = train_test_split(X_train_0, y_train_0, test_size=0.2, shuffle=True, stratify=y_train_0)
            X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_train_1, y_train_1, test_size=0.2, shuffle=True,stratify=y_train_1)
            X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_train_2, y_train_2, test_size=0.2, shuffle=True,stratify=y_train_2)
            # Assign splits
            splits = {
                f"{dataset_name}_client0_train.csv": (X_train_0, y_train_0),  # 50% (train + test)
                f"{dataset_name}_client1_train.csv": (X_train_1, y_train_1),  # 30% (train + test)
                f"{dataset_name}_client2_train.csv": (X_train_2, y_train_2),  # 20% (train + test)
                f"{dataset_name}_client0_test.csv": (X_test_0, y_test_0),
                f"{dataset_name}_client1_test.csv": (X_test_1, y_test_1),
                f"{dataset_name}_client2_test.csv": (X_test_2, y_test_2)
            }

            # Save each split to a file
            if not os.path.exists(DATA_PATH):
                os.makedirs(DATA_PATH)

            for file_name, (split_X, split_y) in splits.items():
                fold_X = pd.DataFrame(split_X.values, columns=[f'feature_{j}' for j in range(X.shape[1])])
                fold_Y = pd.Series(split_y.values, name='target')
                fold_data = pd.concat([fold_X, fold_Y], axis=1)
                fold_data.to_csv(f"{DATA_PATH}{file_name}", header=True, index=False)
        else:
            # Step 2: Initialize StratifiedKFold with shuffling
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

            print("Starting StratifiedKFold")
            # Step 3: Loop over each fold
            for i, (_, test_index) in enumerate(skf.split(X, y)):
                # Create DataFrames for features and target
                fold_X = pd.DataFrame(X.iloc[test_index].values, columns=[f'feature_{j}' for j in range(X.shape[1])])
                fold_Y = pd.Series(y.iloc[test_index].values, name='target')

                # Merge X and Y into a single DataFrame
                fold_data = pd.concat([fold_X, fold_Y], axis=1)

                # Check if the folder exists, if not, create it
                if not os.path.exists(DATA_PATH):
                    os.makedirs(DATA_PATH)

                # Save the fold data to a CSV file
                fold_data.to_csv(f"{DATA_PATH}data_portion_{i}.csv", header=True, index=False)

"""Argparser"""
parser = argparse.ArgumentParser(description="Create dataset and split into portions.")
parser.add_argument('--csv', type=int, help='CSV mode', default=0)
parser.add_argument('--n', type=int, help='Number of splits for StratifiedKFold', default=0)
parser.add_argument('--s', type=int, help='Number of samples in the dataset', default=0)
parser.add_argument('--v', type=int, help='Variance of clusters', default=0)
parser.add_argument('--c', type=int, help='Number of centroids for make_blobs',default=0)
parser.add_argument('--f', type=int, help='Number of features for make_blobs', default=0)
args = parser.parse_args()

if __name__ == "__main__":
    if None in [args.csv, args.s,  args.c,  args.v, args.f, args.n,]:
        create_data(0,3,500, 2, 15, 10)
    else:
        if args.csv != 0:
            create_data(args.csv,args.n)
        else:
            create_data(args.csv,args.n, args.s, args.v, args.c, args.f )