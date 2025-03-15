import os
from classes.Solver import Solver
from classes.HashableDataFrame import HashableDataFrame
import time
import pstats
import cProfile
import pandas as pd
from wakepy import keep

if __name__ == "__main__":
    #df = pd.read_csv("data/diabetes.csv", sep=" ", header=None)
    #df = pd.read_csv("data/hepatitis.csv", sep=" ", header=None)
    #df = pd.read_csv("test.csv", sep=" ", header=None)
    #df = pd.read_csv("experiment_datasets\\10rooms\\10rooms_2_versus_all.csv", sep=" ", header=None)
    #df = pd.read_csv("data/tic-tac-toe.csv", sep=" ", header=None)
    # df = pd.read_csv("data/vote.csv", sep=" ", header=None)
    #df = pd.read_csv("experiment_datasets/cartpole/cartpole_21_versus_all.csv", sep=" ", header=None)
    # solver = Solver(name = 'monk3', sample_size=1000, MIP_gap=0.2)
    # with keep.running():
    #     with cProfile.Profile() as profile:
    #         print(solver.solve(df))

    #         results = pstats.Stats(profile)
    #         results.sort_stats(pstats.SortKey.CUMULATIVE)
    #         results.print_stats()
    # names = ['10rooms/10rooms_8_versus_all.csv','10rooms/10rooms_9_versus_all.csv','helicopter/helicopter_13_versus_all.csv','10rooms/10rooms_7_versus_all.csv','helicopter/helicopter_12_versus_all.csv']
    #names = ['data/monk3_bin.csv','experiment_datasets/helicopter/helicopter_13_versus_all.csv','experiment_datasets/cartpole/cartpole_12_versus_all.csv','experiment_datasets/cartpole/cartpole_15_versus_all.csv','experiment_datasets/cartpole/cartpole_18_versus_all.csv','experiment_datasets/cartpole/cartpole_20_versus_all.csv','experiment_datasets/cartpole/cartpole_21_versus_all.csv','experiment_datasets/cartpole/cartpole_22_versus_all.csv','experiment_datasets/cartpole/cartpole_28_versus_all.csv','experiment_datasets/cartpole/cartpole_29_versus_all.csv','experiment_datasets/cartpole/cartpole_43_versus_all.csv','experiment_datasets/cartpole/cartpole_44_versus_all.csv','experiment_datasets/cartpole/cartpole_45_versus_all.csv','experiment_datasets/cartpole/cartpole_46_versus_all.csv','experiment_datasets/cartpole/cartpole_47_versus_all.csv','experiment_datasets/cartpole/cartpole_48_versus_all.csv','experiment_datasets/cartpole/cartpole_52_versus_all.csv','experiment_datasets/cartpole/cartpole_74_versus_all.csv','experiment_datasets/10rooms/10rooms_8_versus_all.csv','experiment_datasets/10rooms/10rooms_9_versus_all.csv','data/hepatitis.csv','experiment_datasets/10rooms/10rooms_7_versus_all.csv','experiment_datasets/helicopter/helicopter_12_versus_all.csv','data/primary-tumor-clean.csv','data/lymph.csv','data/vote.csv','data/tic-tac-toe.csv']
    # names = ['sampled_from_witty/lupus_0.5_6.csv_bobotree.csv_binarized.csv']
    # names = ['experiment_datasets/helicopter/helicopter_12_versus_all.csv']
    # sample_sizes = [50,100,150,200,300,400,500,600,700,800,900,1000]
    # gaps = [0.05,0.1,0.15,0.2,0.3]
    # nrs_runs = [1,2,3,5,7,10]
    # strategies = ['set-cover']
    # gaps = [0.4,0.5,0.6,0.7,0.8,0.9]
    # nrs_runs = [1,3,5,7]
    # Initialize an empty DataFrame to store results
    
    number_binarized_datasets= 580
    output_csv = 'bobotree_results3.csv'
    witty_results = pd.read_csv("Witty_results - Copy.csv",sep=";" , header = None) 
    # print(witty_results.shape)
    results = pd.DataFrame(columns=[    #     'name', 'size', 'witty_size, 'elapsed_time', 'witty_time, 'number_instances', 'number_features', 'number_features_bobo', 'hamming_distance', 'cuts'
                ])
    directory = 'sampled_from_witty'
    # for index, filename in enumerate(os.scandir(directory)):
    #     if filename.is_file():
    for i, row in witty_results.iterrows():
            if(row[13] == -1):
                continue # Skip datasets where the "efficient" algorithm timeouts

            name = row[2].split("/")[-1].split("\\")[-1]
            input_csv = 'sampled_from_witty/' + name + "_bobotree.csv"  # Path to your original CSV file
            df = pd.read_csv(input_csv, sep=" ", header=None)
                       
            solver = Solver(name = name, sample_size= int(0.66 * len(df)), MIP_gap= 0.3, cover_runs=7, lower_bound_strategy="set-cover")
            

            start_time = time.time()
            size, depth, explored, avg_question_length =  solver.solve(df)
            if size == -1:
                continue
            elapsed_time = time.time() - start_time 
            
            # Append the results as a new row
            results = pd.concat([results, pd.DataFrame([{
                'name': name,
                'size': size,
                'witty_size': row[13],
                'elapsed_time': elapsed_time * 1000,
                'witty_time': row[9],
                'number_instances': row[3],
                'number_features': row[6], 
                'number_features_bobo': df.shape[1] -1, 
                'hamming_distance': row[25],
                'cuts': row[26]
            }])], ignore_index=True)

            # print(results)
            with open(output_csv, 'w', newline='') as file:
                results.to_csv(file, sep=' ', index=False, header=False)
                