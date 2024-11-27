from classes.Solver import Solver
import time
import pandas as pd

if __name__ == "__main__":
    #df = pd.read_csv('test.csv', sep=" ", header=None)
    df = pd.read_csv('data/monk3_bin.csv', sep=" ", header=None)
    solver = Solver()
    print(solver.solve(df))
    # names = ['anneal', 'australian-credit','german-credit','hepatitis']
    # sample_sizes = [50,100,150,200]
    # gaps = [0.01,0.05,0.1,0.15,0.2,0.3]
    # parent_influences = [0.01,0.1,0.15,0.2,0.3]
    # set_cover_influences = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7]
    # nrs_runs = [1,2,3,4,5,6,7,8,9,10]
    # strategies = ['both', 'one-off', 'set-cover']

    # # Initialize an empty DataFrame to store results
    
    # for name in names:
    #     results = pd.DataFrame(columns=[
    #     'name', 'sample_size', 'gap', 'parent_influence', 'set_cover_influence', 
    #     'nr_runs', 'strategy', 'size', 'depth', 'explored', 'elapsed_time'
    #     ])
    #     filename = 'data/' + name + '.csv'
    #     output_csv = name + '_binary_classification_results.csv'
    #     df = pd.read_csv(filename, sep=" ", header=None)
    #     for sample_size in sample_sizes:
    #         for gap in gaps:
    #             for parent_influence in parent_influences:
    #                 for set_cover_influence in set_cover_influences:
    #                     for nr_runs in  nrs_runs:
    #                         for strategy in strategies:
    #                             solver = Solver(sample_size, gap, parent_influence, set_cover_influence, nr_runs)
    #                             start_time = time.time()
    #                             size, depth, explored =  solver.solve(df)
    #                             elapsed_time = time.time() - start_time

    #                              # Append the results as a new row
    #                             results = pd.concat([results, pd.DataFrame([{
    #                                 'name': name,
    #                                 'sample_size': sample_size,
    #                                 'gap': gap,
    #                                 'parent_influence': parent_influence,
    #                                 'set_cover_influence': set_cover_influence,
    #                                 'nr_runs': nr_runs,
    #                                 'strategy': strategy,
    #                                 'size': size,
    #                                 'depth': depth,
    #                                 'explored': explored,
    #                                 'elapsed_time': elapsed_time
    #                             }])], ignore_index=True)
    #     print(results)
    #     with open(output_csv, 'w', newline='') as file:
    #         results.to_csv(file, sep=' ', index=False, header=False)
            
    