import os, argparse
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from ucimlrepo import fetch_ucirepo

DATA_PATH = "./"

def process_repo(repo):
    # Convert features to DataFrame
    X = pd.DataFrame(repo.data.features) if not isinstance(repo.data.features,
                                                           pd.DataFrame) else repo.data.features

    # Convert targets to Series, handling potential multi-column case
    y_df = pd.DataFrame(repo.data.targets) if not isinstance(repo.data.targets,
                                                             pd.DataFrame) else repo.data.targets

    y = y_df.iloc[:, 0]
    data = pd.concat([X, y], axis=1)

    return data, y.name

def load_custom_dataset(filename):
    """
    Loads dataset based on filename, processes it, applies one-hot encoding, and handles missing values.
    """
    preprocessed = False
    if filename == "bank":
        bank = fetch_ucirepo(id=222)

        data, target_col = process_repo(bank)

        print(data.head())
    elif filename == "breast-cancer":
        breast = fetch_ucirepo(id=17)

        data, target_col = process_repo(breast)

        print(data.head())
    elif filename == "amazon_software_reviews":

        amazon  = load_dataset("contemmcm/amazon_reviews_polarity_2013", "software")

        data = amazon['complete'].to_pandas()
        data.drop(['review/summary', 'review/text'],
                axis='columns', inplace=True)

        target_col = 'review/score'

        print(data.head())
    elif filename == "card_fraud":
        statlog_german_credit_data = fetch_ucirepo(id=144)

        data, target_col = process_repo(statlog_german_credit_data)

        print(data.head())
    elif filename == "adult-census-income":
        # fetch dataset
        census_income = fetch_ucirepo(id=20)

        data, target_col = process_repo(census_income)

        # # Convert the training and test datasets to pandas DataFrames
        # data = data['train'].to_pandas()
        # target_col = 'income'
        # Map the income column to binary values: 0 for <=50K, 1 for >50K
        data[target_col] = data[target_col].apply(lambda x: 1 if x == '>50K' else 0)
    elif filename == "heart_disease_preprocessed":
        data = pd.read_csv(filename+".csv", header=0)
        target_col = 'HeartDisease'
    else:
        raise ValueError(f"Unknown dataset: {filename}")

    return data, target_col

def save_csv(client_id, filename, X_train, X_test, y_train, y_test, postfix=".csv"):
    """
    Save Client train and test data to CSV files.
    Arguments:
        client_id {int} -- Client ID.
        filename {str} -- Initial dataset CSV filename.
        X_train {pd.DataFrame} -- Training dataset.
        X_test {pd.DataFrame} -- Testing dataset.
        y_train {pd.DataFrame} -- Training dataset.
        y_test {pd.DataFrame} -- Testing dataset.
        postfix {str} -- Postfix filename.
    """
    # Save train and test splits
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    output_dir = f"{filename}_dir"
    os.makedirs(output_dir, exist_ok=True)

    train_data.to_csv(os.path.join(output_dir, f"{filename}_client_{client_id}_train{postfix}"), index=False, header=True)
    test_data.to_csv(os.path.join(output_dir, f"{filename}_client_{client_id}_test{postfix}"), index=False, header=True)


def split_data(n_splits: int, filename: str, test_size: float, split_size: list) -> None:
    """
    Splits data into n portions, where n is the number of clients.
    Arguments:
        n_splits (int): Number of splits.
        filename (str): Name of the CSV dataset file.
        test_size (float): Ratio of test size.
        split_size (list): List of split sizes.
        target_col (str): Name of the target column.
    """

    data, target_col = load_custom_dataset(filename)

    print(f'Loaded {filename} with {data.shape[0]} samples.')
    # Rename the target column to 'target' for internal processing
    data = data.rename(columns={target_col: 'target'})
    print(data.head())
    #Equal client splitting
    if len(split_size) == 0:
        print('Equal client splitting ...')
        # Equal splitting using StratifiedKFold
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for client_id, (_, test_index) in enumerate(kfold.split(data, data['target'])):
            client_portion = data.iloc[test_index].reset_index(drop=True)

            # Split client_portion into train and test
            X = client_portion.drop(columns=['target'])
            y = client_portion['target']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=42
            )
            #Save client train and test data to CSVs
            save_csv(client_id,filename, X_train, X_test, y_train, y_test)


    #Unbalanced client splitting (split sizes defined by split_size param)
    else:
        print('Unbalanced client splitting ...')
        total_data = data.shape[0]
        split_ratios = [size / sum(split_size) for size in split_size]  # Normalize split_size to proportions
        remaining_data = data
        loop = 1
        for client_id, split_ratio in enumerate(split_ratios):
            client_size = int(total_data * split_ratio)  # Calculate number of rows for this client
            print(f'Client {client_id} split projected with {client_size} samples.')

            # Adjust split_ratio for the remaining data
            adjusted_ratio = split_ratio * total_data / remaining_data.shape[0]
            print(f'Adjusted {adjusted_ratio} samples ...')

            if loop < n_splits:

                # Split remaining_data into a client portion and remaining portion
                client_portion, remain_portion = train_test_split(
                    remaining_data, test_size=(1 - adjusted_ratio), stratify=remaining_data['target'], random_state=42
                )


                X = client_portion.drop(columns=['target'])
                y = client_portion['target']

                loop+=1
            else:
                #Final split
                X = remaining_data.drop(columns=['target'])
                y = remaining_data['target']

            # Further split the client portion into train and test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=42
            )

            # Save train and test splits
            save_csv(client_id, filename, X_train, X_test, y_train, y_test)

            # Update remaining_data for the next iteration
            remaining_data = remain_portion



"""Argparser"""
parser = argparse.ArgumentParser(description="Create data splits from single large file into separate client portions with already defined train and test portions per client.")
parser.add_argument('--n', type=int, help='Number of splits for StratifiedKFold', required=True)
parser.add_argument('--filename', type=str, help='name of the original input file', required=True)
parser.add_argument('--test_size', type=float, help='size of the testing portion. Each client would split his data by provided number in order to keep data for testing.', required=True)
parser.add_argument('--split_client_portion', nargs="+", help='list of integer split of the original dataset per each client. Use in case when clients have different volume portion of data. Example: --n 3 --split_client_portion 20 30 50', type=int, default=[])
#parser.add_argument('--target_col', type=str, help='column name of target variable', required=True)
args = parser.parse_args()

if __name__ == "__main__":
    if len(args.split_client_portion) > 0 and len(args.split_client_portion) != args.n:
        raise ValueError('The number of splits should be equal to the number of clients')

    split_data(n_splits=args.n, filename=args.filename, test_size= args.test_size ,split_size=args.split_client_portion)
