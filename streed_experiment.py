import pandas as pd
from pystreed import STreeDClassifier

df = pd.read_csv("data/hepatitis.csv", sep=" ", header=None)
#df = pd.read_csv("test.csv", sep=" ", header=None)
X = df.iloc[:, 1:]
y = df[0]

max_depth=20
model = STreeDClassifier("cost-complex-accuracy", max_depth = max_depth, cost_complexity=1/(2**(max_depth-1)))
model.fit(X,y)

print(model.get_n_leaves() -1)

# Make predictions
y_pred = model.predict(X)

# Calculate misclassifications
misclassifications = (y_pred != y).sum()

# Output the number of misclassifications
print(f"Number of misclassifications: {misclassifications}")