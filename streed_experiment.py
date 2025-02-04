import pandas as pd
from pystreed import STreeDClassifier

#df = pd.read_csv("data/hepatitis.csv", sep=" ", header=None)
#df = pd.read_csv("test.csv", sep=" ", header=None)
#df = pd.read_csv("data/vote.csv", sep=" ", header=None)
df = pd.read_csv("experiment_datasets/cartpole/cartpole_12_versus_all.csv", sep=" ", header=None)

X = df.iloc[:, 1:]
y = df[0]

max_depth=20
model = STreeDClassifier("cost-complex-accuracy", max_depth = max_depth, cost_complexity=1/(len(df)*2**(max_depth-1)), time_limit=60000)
model.fit(X,y)

print("size: ", model.get_n_leaves() -1)
print("depth: ", model.get_depth())

# Make predictions
y_pred = model.predict(X)

# Calculate misclassifications
misclassifications = (y_pred != y).sum()

# Output the number of misclassifications
print(f"Number of misclassifications: {misclassifications}")