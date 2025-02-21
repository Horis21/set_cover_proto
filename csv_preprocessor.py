import pandas as pd

def convert_to_dtcontrol_format(input_csv, output_csv, permissive=False, control_dim=1):
    """
    Convert CSV to the format required by DTControl.
    
    :param input_csv: Path to the input CSV file (with "parameters" and "label" columns).
    :param output_csv: Path to the output CSV file formatted for DTControl.
    :param permissive: Boolean, True if the controller is permissive, False if non-permissive.
    :param control_dim: Number of control variables.
    """
    # Step 1: Load the input CSV file
    df = pd.read_csv(input_csv, sep=" ", header=None)

    state_dim= df.shape[1] -1 

    # Shift the first column (class) to the end
    reordered_df = df.iloc[:, 1:].copy()  # All columns except the first
    reordered_df['class'] = df.iloc[:, 0]  # Add the first column (class) to the end

    # Step 4: Open the output file and write the metadata lines
    with open(output_csv, 'w', newline='') as file:
        # Metadata line for permissiveness
        file.write("#PERMISSIVE\n" if permissive else "#NON-PERMISSIVE\n")

        # Metadata line for dimensions
        file.write(f"#BEGIN {state_dim} {control_dim}\n")

        # Step 5: Write each row of the final DataFrame in the required format
        reordered_df.to_csv(file, index=False, header=False)

    print(f"Conversion completed. Output saved to {output_csv}")

import pandas as pd

def convert_csv_witty(input_file, output_file):
    # Read the space-separated CSV file
    df = pd.read_csv(input_file, sep=' ', header=None)

    # Swap the first column with the last column
    cols = df.columns.tolist()
    cols.append(cols.pop(0))  # Move first column to the end
    df = df[cols]
    
    # Save to a comma-separated CSV file
    df.to_csv(output_file, index=False, header=False)
names = ['data/monk3_bin.csv','experiment_datasets/helicopter/helicopter_13_versus_all.csv','experiment_datasets/cartpole/cartpole_12_versus_all.csv','experiment_datasets/cartpole/cartpole_15_versus_all.csv','experiment_datasets/cartpole/cartpole_18_versus_all.csv','experiment_datasets/cartpole/cartpole_20_versus_all.csv','experiment_datasets/cartpole/cartpole_21_versus_all.csv','experiment_datasets/cartpole/cartpole_22_versus_all.csv','experiment_datasets/cartpole/cartpole_28_versus_all.csv','experiment_datasets/cartpole/cartpole_29_versus_all.csv','experiment_datasets/cartpole/cartpole_43_versus_all.csv','experiment_datasets/cartpole/cartpole_44_versus_all.csv','experiment_datasets/cartpole/cartpole_45_versus_all.csv','experiment_datasets/cartpole/cartpole_46_versus_all.csv','experiment_datasets/cartpole/cartpole_47_versus_all.csv','experiment_datasets/cartpole/cartpole_48_versus_all.csv','experiment_datasets/cartpole/cartpole_52_versus_all.csv','experiment_datasets/cartpole/cartpole_74_versus_all.csv','experiment_datasets/10rooms/10rooms_8_versus_all.csv','experiment_datasets/10rooms/10rooms_9_versus_all.csv','data/hepatitis.csv','experiment_datasets/10rooms/10rooms_7_versus_all.csv','experiment_datasets/helicopter/helicopter_12_versus_all.csv','data/primary-tumor-clean.csv','data/lymph.csv','data/vote.csv','data/tic-tac-toe.csv']
#names = ['experiment_datasets/helicopter/helicopter_12_versus_all.csv']
for file in names:
    input_csv =  file  # Path to your original CSV file
    output_csv =  'witty_datasets/' + file.split(".")[0].split("/")[-1] +'_witty.csv'
    convert_csv_witty(input_csv, output_csv)
