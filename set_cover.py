from classes.Solver import Solver
from classes.HashableDataFrame import HashableDataFrame
import time
import pstats
import cProfile
import pandas as pd
from wakepy import keep

if __name__ == "__main__":
    #df = pd.read_csv("data/diabetes.csv", sep=" ", header=None)
    # df = pd.read_csv("data/monk3_bin.csv", sep=" ", header=None)
    #df = pd.read_csv("data/hepatitis.csv", sep=" ", header=None)
    #df = pd.read_csv("test.csv", sep=" ", header=None)
    #df = pd.read_csv("experiment_datasets\\10rooms\\10rooms_2_versus_all.csv", sep=" ", header=None)
    #df = pd.read_csv("data/tic-tac-toe.csv", sep=" ", header=None)
    # df = pd.read_csv("data/vote.csv", sep=" ", header=None)
    #df = pd.read_csv("experiment_datasets/cartpole/cartpole_21_versus_all.csv", sep=" ", header=None)
    # solver = Solver(sample_size=1000, MIP_gap=0.2)
    # with keep.running():
    #     with cProfile.Profile() as profile:
    #         print(solver.solve(df))

    #         results = pstats.Stats(profile)
    #         results.sort_stats(pstats.SortKey.CUMULATIVE)
    #         results.print_stats()
    #names = ['10rooms/10rooms_8_versus_all.csv','10rooms/10rooms_9_versus_all.csv','helicopter/helicopter_13_versus_all.csv','10rooms/10rooms_7_versus_all.csv','helicopter/helicopter_12_versus_all.csv']
    names = ['data/primary-tumor-clean.csv','data/vote.csv','data/tic-tac-toe.csv','data/monk3_bin.csv','data/hepatitis.csv','experiment_datasets/helicopter/helicopter_13_versus_all.csv','experiment_datasets/cartpole/cartpole_12_versus_all.csv','experiment_datasets/cartpole/cartpole_15_versus_all.csv','experiment_datasets/cartpole/cartpole_18_versus_all.csv','experiment_datasets/cartpole/cartpole_20_versus_all.csv','experiment_datasets/cartpole/cartpole_21_versus_all.csv','experiment_datasets/cartpole/cartpole_22_versus_all.csv','experiment_datasets/cartpole/cartpole_28_versus_all.csv','experiment_datasets/cartpole/cartpole_29_versus_all.csv','experiment_datasets/cartpole/cartpole_43_versus_all.csv','experiment_datasets/cartpole/cartpole_44_versus_all.csv','experiment_datasets/cartpole/cartpole_45_versus_all.csv','experiment_datasets/cartpole/cartpole_46_versus_all.csv','experiment_datasets/cartpole/cartpole_47_versus_all.csv','experiment_datasets/cartpole/cartpole_48_versus_all.csv','experiment_datasets/cartpole/cartpole_52_versus_all.csv','experiment_datasets/cartpole/cartpole_74_versus_all.csv','experiment_datasets/10rooms/10rooms_8_versus_all.csv','experiment_datasets/10rooms/10rooms_9_versus_all.csv']
    # sample_sizes = [50,100,150,200,300,400,500,600,700,800,900,1000]
    # gaps = [0.05,0.1,0.15,0.2,0.3]
    # nrs_runs = [1,2,3,5,7,10]
    # strategies = ['set-cover']

    # Initialize an empty DataFrame to store results
    
    for name in names:
        print(name)
        # results = pd.DataFrame(columns=[    #     'name', 'sample_size', 'gap','nr_runs', 'strategy', 'size', 'depth', 'explored', 'elapsed_time'
        # ])
        results = pd.DataFrame(columns=[    #     'name', 'explored', 'elapsed_time'
        ])
        filename = name
        output_csv = name + '_one-off_results.csv'
        df = pd.read_csv(filename, sep=" ", header=None)
        if len(Solver().one_off_features(HashableDataFrame(df))) == 0:
            continue 
        
        solver = Solver(lower_bound_strategy='one-off')
        start_time = time.time()
        size, depth, explored =  solver.solve(df)
        elapsed_time = time.time() - start_time
        
        # Append the results as a new row
        results = pd.concat([results, pd.DataFrame([{
            'name': name,
            'explored': explored,
            'elapsed_time': elapsed_time
        }])], ignore_index=True)
        print(results)
        
        with open(output_csv, 'w', newline='') as file:
            results.to_csv(file, sep=' ', index=False, header=False)
            