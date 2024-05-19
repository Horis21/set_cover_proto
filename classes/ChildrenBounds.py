import sys

class ChildrenBounds:
    def __init__(self, lower):
        self.left = 0 if lower else sys.maxsize
        self.right = 0 if lower else sys.maxsize