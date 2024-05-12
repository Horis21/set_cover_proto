class Node:
    def __init__(self, df, f, isLeft, parent):
        self.df = df
        self.f = f
        self.isLeft = isLeft
        self.lowers = {}
        self.uppers = {}
        self.solutions = {}
        self.left = None
        self.right = None
        self.parents = []  # Initialize parents list
        if parent is not None:
            self.parents.append(parent)  # Add parent to the list

    def addParent(self, parent):
        self.parents.append(parent)  # Add parent to the list

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return True

    def __lt__(self, other):
        return True

    def __gt__(self, other):
        return True

    def __le__(self, other):
        return True

    def __ge__(self, other):
        return True
    
    def __str__(self):
        return "feature: " + str(self.f) + " ,left: " + str(self.left) + " ,right: "+ str(self.right)

