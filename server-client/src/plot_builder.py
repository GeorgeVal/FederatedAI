import os
import csv
import argparse
import matplotlib.pyplot as plt


def read_optimal_value(csv_folder, model):
    """Reads the single optimal F1 score from optimal.csv."""
    optimal_filepath = os.path.join(csv_folder, f'{model}_w_optimal.csv')
    if not os.path.exists(optimal_filepath):
        return None

    with open(optimal_filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        return float(next(reader)[0])  # Assuming F1 score is in the second column


def generate_plot_from_csv(num_clients, folder_name):
    """
    Generates a plot using data from CSV files.
    """
    csv_folder = f"./CSV/{folder_name}"
    model = str(folder_name.split('_')[0])
    optimal_f1 = read_optimal_value(csv_folder, model)
    print(f"Optimal F1 score: {optimal_f1}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('Round')
    ax.set_ylabel('F1 Score')
    ax.set_title(f'F1 Score per Round for Clients and Aggregation for {model}')
    ax.grid(True)

    colormap = plt.cm.viridis
    colors = [colormap(i / (num_clients - 1)) for i in range(num_clients)]

    for client_id in range(num_clients):
        color = colors[client_id % len(colors)]

        for data_type in ['client', 'train_client']:
            filepath = os.path.join(csv_folder, f'{model}_{data_type}_{client_id}_data.csv')
            if os.path.exists(filepath):
                with open(filepath, 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    next(reader)
                    rounds, f1s = zip(*((int(row[0]), float(row[1])) for row in reader))
                    linestyle = '-' if data_type == 'client' else '--'
                    ax.plot(rounds, f1s, marker='o', color=color, linestyle=linestyle,
                            label=f'{data_type.capitalize()} {client_id}')

    aggregation_files = {
        'Weighted Aggregation': 'weighted_aggregation_data.csv',
        'Unweighted Aggregation': 'unweighted_aggregation_data.csv',
        'No Tune Aggregation': 'notune_aggregation_data.csv'
    }

    for label, filename in aggregation_files.items():
        filepath = os.path.join(csv_folder, f'{model}_{filename}')
        if os.path.exists(filepath):
            with open(filepath, 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                rounds, f1s = zip(*((int(row[0]), float(row[1])) for row in reader))
                if label == 'Weighted Aggregation' and optimal_f1 is not None:
                    for i, f1 in enumerate(f1s):
                        color = 'red' if f1 == optimal_f1 else 'black'
                        ax.scatter(rounds[i], f1, color=color, marker='x', s=50, zorder=3)
                if label == 'Weighted Aggregation':
                    ax.plot(rounds, f1s, linestyle='--', color='black', label=label, marker='x')
                elif label == 'Unweighted Aggregation':
                    ax.plot(rounds, f1s, linestyle='-', color='grey', label=label, marker='x')
                elif label == 'No Tune Aggregation':
                    ax.plot(rounds, f1s, linestyle='-', color='green', label=label, marker='x')
                else:
                    raise ValueError("Unknown label")

    noFed_filepath = os.path.join(csv_folder, f'{model}_nofed_data.csv')
    if os.path.exists(noFed_filepath):
        with open(noFed_filepath, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                f1s = float(row[0])
                ax.axhline(y=f1s, color='red', linestyle='--', linewidth=1.5, label=f'NoFed')

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plot_filepath = os.path.join(csv_folder, f'{model}_plot.png')
    fig.savefig(plot_filepath)
    print(f'Saved plot as {plot_filepath}')
    plt.close(fig)


def main(num_clients, folder_name, process_all):
    if process_all:
        csv_root = './CSV'
        for folder in os.listdir(csv_root):
            folder_path = os.path.join(csv_root, folder)
            if os.path.isdir(folder_path):
                print(f'Processing folder: {folder}')
                generate_plot_from_csv(num_clients, folder)
    else:
        generate_plot_from_csv(num_clients, folder_name)


parser = argparse.ArgumentParser(description="Specify parameters.")
parser.add_argument('--c', type=int, help='Number of clients', required=True)
parser.add_argument('--fn', type=str, help='Folder name (where to get data and save png)', required=False)
parser.add_argument('--all', action='store_true', help='Process all folders inside CSV')
args = parser.parse_args()

if __name__ == "__main__":
    main(args.c, args.fn, args.all)
