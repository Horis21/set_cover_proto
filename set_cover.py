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
    names = ['data/vote.csv','data/lymph.csv','data/tic-tac-toe.csv']
    # names = ['experiment_datasets/helicopter/helicopter_12_versus_all.csv']
    # sample_sizes = [50,100,150,200,300,400,500,600,700,800,900,1000]
    # gaps = [0.05,0.1,0.15,0.2,0.3]
    # nrs_runs = [1,2,3,5,7,10]
    # strategies = ['set-cover']
    # gaps = [0.4,0.5,0.6,0.7,0.8,0.9]
    # nrs_runs = [1,3,5,7]
    # Initialize an empty DataFrame to store results
    
    output_csv = 'bobotree_results2csv' 
    results = pd.DataFrame(columns=[    #     'name', 'size', 'explored', 'elapsed_time', 'query_time'
                ])
    for name in names:
        filename = name
        name = name.split(".")[0].split("/")[-1]
        df = pd.read_csv(filename, sep=" ", header=None)
        
        solver = Solver(name = name, sample_size= int(0.66 * len(df)), MIP_gap= 0.3, cover_runs=7, lower_bound_strategy="set-cover")
        

        start_time = time.time()
        size, depth, explored, query_time =  solver.solve(df)
        elapsed_time = time.time() - start_time - query_time
        
        # Append the results as a new row
        results = pd.concat([results, pd.DataFrame([{
            'name': name,
            'size': size,
            'explored': explored,
            'elapsed_time': elapsed_time,
            'query_time"': query_time
        }])], ignore_index=True)

        print(results)
        with open(output_csv, 'w', newline='') as file:
            results.to_csv(file, sep=' ', index=False, header=False)
            