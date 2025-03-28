file_path = 'binarized_datasets.txt'


with open(file_path, 'r') as file:
    file_content = []
    line = file.readline()
    
    while line:
        name = line.split("/")[-1].split("\\")[0].split("_")[0] + "_" + line.split("/")[-1].split("\\")[0].split("_")[1] + "_" + line.split("/")[-1].split("\\")[0].split("_")[2]
        file_content.append(name)
        line = file.readline()

# # print(file_content)

# counter = 0
# for i, row in witty_results.iterrows():
#     counter += 1 if row[0] in file_content else 0

# print(counter)
number_binarized_datasets = 580
number_binarized_datasets_and_finish_running= 291 
number_not_same_datasets = 230

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
filename = "bobotree_results3.csv"  # Replace with actual file
columns = ["dataset", "tree_size", "size_B", "time_A", "time_B", "num_rows", "num_cols", "num_features", "hamming_dist", "num_cuts"]
df = pd.read_csv(filename, delim_whitespace=True, names=columns)

# List of binarized datasets (replace with actual filenames)

# Categorize datasets
df["binarized"] = df["dataset"].isin(file_content)
df["same_tree_size"] = df.apply(lambda row: row["tree_size"] == row["size_B"] if row["binarized"] else None, axis=1)

# Count mismatches
mismatch_count = df[df["binarized"] & (df["same_tree_size"] == False)].shape[0]
print(f"Number of binarized datasets with different tree sizes: {mismatch_count}")

# Split datasets into groups
group_not_binarized = df[~df["binarized"]]
group_binarized_same = df[df["binarized"] & (df["same_tree_size"] == True)]
group_binarized_diff = df[df["binarized"] & (df["same_tree_size"] == False)]

def plot_scatter(group, color_by, filename):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(group["time_A"]/ 1000, group["time_B"] / 1000, c=group[color_by], cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label=color_by)

     # Draw diagonal line
    min_val = min(group["time_A"].min() / 1000, group["time_B"].min() / 1000)
    max_val = max(group["time_A"].max() / 1000, group["time_B"].max() / 1000)
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='red', label='Equal Time')

    plt.xlabel("Time Bobotree (seconds)")
    plt.ylabel("Time Witty (seconds)")
    plt.title(f"Scatterplot (Colored by {color_by.replace('_', ' ').title()})")
    plt.savefig(filename, dpi=300)
    plt.close()

# Define attributes to color by and corresponding filenames
attributes = {
    "num_rows": "number_rows",
    "num_cols": "number_columns",
    "tree_size": "tree_size",
    "hamming_dist": "hamming_distance",
    "num_cuts": "number_cuts"
}

# Generate scatter plots for each group
for group, label in zip([group_not_binarized, group_binarized_same, group_binarized_diff],
                        ["not_binarized", "binarized_same_tree", "binarized_diff_tree"]):
    for attr, attr_name in attributes.items():
        filename = f"plots/scatterplot_{label}_{attr_name}.png"
        plot_scatter(group, attr, filename)