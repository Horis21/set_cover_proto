import pandas as pd
from pystreed import STreeDClassifier

df = pd.read_csv("data/diabetes.csv", sep=" ", header=None)
#df = pd.read_csv("test.csv", sep=" ", header=None)
X = df.iloc[:, 1:]
y = df[0]

max_depth=20
model = STreeDClassifier("cost-complex-accuracy", max_depth = max_depth, cost_complexity=1/(len(df)*max_depth))
model.fit(X,y)

print(model.get_n_leaves() -1)
model.print_tree()