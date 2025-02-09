import time
import pandas as pd
from pystreed import STreeDClassifier

names = ['data/primary-tumor-clean.csv','data/vote.csv','data/tic-tac-toe.csv','data/monk3_bin.csv','data/hepatitis.csv','experiment_datasets/10rooms/10rooms_8_versus_all.csv','experiment_datasets/10rooms/10rooms_9_versus_all.csv','experiment_datasets/helicopter/helicopter_13_versus_all.csv','experiment_datasets/cartpole/cartpole_12_versus_all.csv','experiment_datasets/cartpole/cartpole_15_versus_all.csv','experiment_datasets/cartpole/cartpole_18_versus_all.csv','experiment_datasets/cartpole/cartpole_20_versus_all.csv','experiment_datasets/cartpole/cartpole_21_versus_all.csv','experiment_datasets/cartpole/cartpole_22_versus_all.csv','experiment_datasets/cartpole/cartpole_28_versus_all.csv','experiment_datasets/cartpole/cartpole_29_versus_all.csv','experiment_datasets/cartpole/cartpole_43_versus_all.csv','experiment_datasets/cartpole/cartpole_44_versus_all.csv','experiment_datasets/cartpole/cartpole_45_versus_all.csv','experiment_datasets/cartpole/cartpole_46_versus_all.csv','experiment_datasets/cartpole/cartpole_47_versus_all.csv','experiment_datasets/cartpole/cartpole_48_versus_all.csv','experiment_datasets/cartpole/cartpole_52_versus_all.csv','experiment_datasets/cartpole/cartpole_74_versus_all.csv']
max_depth=20

results = pd.DataFrame(columns=[    #     'name', 'size', 'elapsed_time'
        ])
for name in names:

    df = pd.read_csv(name, sep=" ", header=None)

    X = df.iloc[:, 1:]
    y = df[0]

    model = STreeDClassifier("cost-complex-accuracy", max_depth = max_depth, cost_complexity=1/(len(df)*2**(max_depth-1)), time_limit=60000)
    start_time = time.time()
    model.fit(X,y)
    elapsed_time = time.time() - start_time
    size =  model.get_n_leaves() -1

    print(size, elapsed_time)
    results = pd.concat([results, pd.DataFrame([{
                                'name': name,
                                'size': size,
                                'elapsed_time': elapsed_time
                            }])], ignore_index=True)

    
    # Make predictions
    y_pred = model.predict(X)

    # Calculate misclassifications
    misclassifications = (y_pred != y).sum()

    if misclassifications != 0:
        print("wtf")

# output_csv = 'streed_results.csv'
# with open(output_csv, 'w', newline='') as file:
#             results.to_csv(file, sep=' ', index=False, header=False)