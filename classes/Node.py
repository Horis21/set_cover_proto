class Node:
    def __init__(self, df, f, parent):
        self.df = df
        self.f = f
        self.left = None
        self.right = None
        self.parents = []  # Initialize parents list
        if parent is not None:
            self.parents.append(parent)  # Add parent to the list

    def addParent(self, parent):
        self.parents.append(parent)  # Add parent to the list

