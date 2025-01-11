from classes.Solver import Solver
import time
import pstats
import cProfile
import pandas as pd
from wakepy import keep

if __name__ == "__main__":
    #df = pd.read_csv("data/diabetes.csv", sep=" ", header=None)
    #df = pd.read_csv("data/monk3_bin.csv", sep=" ", header=None)
    df = pd.read_csv("data/hepatitis.csv", sep=" ", header=None)
    #df = pd.read_csv("test.csv", sep=" ", header=None)
    solver = Solver(sample_size=5000, MIP_gap=0.2)
    with cProfile.Profile() as profile:
        print(solver.solve(df))

        results = pstats.Stats(profile)
        results.sort_stats(pstats.SortKey.CUMULATIVE)
        results.print_stats()
    # names = ['diabetes', 'australian-credit','german-credit','hepatitis']
    # sample_sizes = [50,100,150,200]
    # gaps = [0.05,0.1,0.15,0.2,0.3]
    # nrs_runs = [1,2,3,5,7,10]
    # strategies = ['both', 'one-off', 'set-cover']

    # # Initialize an empty DataFrame to store results
    
    # for name in names:
    #     results = pd.DataFrame(columns=[    #     'name', 'sample_size', 'gap','nr_runs', 'strategy', 'size', 'depth', 'explored', 'elapsed_time'
    #     ])
    #     filename = 'data/' + name + '.csv'
    #     output_csv = name + '_binary_classification_results.csv'
    #     df = pd.read_csv(filename, sep=" ", header=None)
    #     for sample_size in sample_sizes:
    #         for gap in gaps:
    #                 for nr_runs in  nrs_runs:
    #                     for strategy in strategies:
    #                         solver = Solver(sample_size, gap, nr_runs, strategy)
    #                         start_time = time.time()
    #                         size, depth, explored =  solver.solve(df)
    #                         elapsed_time = time.time() - start_time
                            
    #                         # Append the results as a new row
    #                         results = pd.concat([results, pd.DataFrame([{
    #                             'name': name,
    #                             'sample_size': sample_size,
    #                             'gap': gap,
    #                             'nr_runs': nr_runs,
    #                             'strategy': strategy,
    #                             'size': size,
    #                             'depth': depth,
    #                             'explored': explored,
    #                             'elapsed_time': elapsed_time
    #                         }])], ignore_index=True)
    #                         gc.collect()
    #     print(results)
    #     with open(output_csv, 'w', newline='') as file:
    #         results.to_csv(file, sep=' ', index=False, header=False)
            
    