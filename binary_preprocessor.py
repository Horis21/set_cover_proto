import os
import numpy as np
import pandas as pd
 

def check_impossible_split(df : pd.DataFrame):
    pos = set(tuple(x[1:]) for x in df[df[0] == 1].values)
    neg = set(tuple(x[1:]) for x in df[df[0] == 0].values)

    return len(pos & neg) == 0

# assign directory
directory = 'data'
# iterate over files in 
# that directory
for filename in os.scandir(directory):
    if filename.is_file():
        df =  pd.read_csv(filename.path, sep=" ", header=None)
        if check_impossible_split(df):
            print(filename.path)