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

names = ['experiment_datasets/helicopter/helicopter_12_versus_all.csv']
for file in names:
    input_csv =  file  # Path to your original CSV file
    output_csv =  file.split(".")[0].split("/")[-1] +'_witty.csv'
    convert_csv_witty(input_csv, output_csv)
