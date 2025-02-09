import time
import pandas as pd
from pystreed import STreeDClassifier

names = ['data\vote.csv','data\tic-tac-toe.csv','data\monk3_bin.csv','']
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

    results = pd.concat([results, pd.DataFrame([{
                                'name': name,
                                'size': size,
                                'elapsed_time': elapsed_time
                            }])], ignore_index=True)

    size =  model.get_n_leaves() -1

    # Make predictions
    y_pred = model.predict(X)

    # Calculate misclassifications
    misclassifications = (y_pred != y).sum()

    if misclassifications != 0:
        print("wtf")

output_csv = 'streed_results.csv'
with open(output_csv, 'w', newline='') as file:
            results.to_csv(file, sep=' ', index=False, header=False)