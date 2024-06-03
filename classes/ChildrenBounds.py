class ChildrenBounds:
    def __init__(self, lower):
        self.left = 0 if lower else 20000000
        self.right = 0 if lower else 20000000