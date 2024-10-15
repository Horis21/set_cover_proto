import queue
from classes.Cache import Cache
import copy

class Node:
    def __init__(self, df, parent_feat, parent ,isLeft):
        self.df = df
        self.parent_feat = parent_feat
        self.isLeft = isLeft
        self.f = None
        self.upper = 20000000
        self.improving = 20000000
        self.lower = 0
        self.feasible = True
        self.lefts = {}
        self.best_f = None
        self.rights = {}
        self.best = None
        self.left = None
        self.right = None
        self.parent = parent

    def backpropagate(self, cache : Cache):
        print("Backpropagating")
        if not self.feasible:
            return
        
        if self.update_local_bounds(cache) and self.parent is not None:
             parent = self.parent
             parent.backpropagate(cache) #Backpropagate further only if bounds were updated


    def link_and_prune(self, solution, cache : Cache):
        self.left = solution.left
        self.right = solution.right
        
        self.f = solution.f

        if self.parent is not None:
            self.parent.backpropagate(cache)
        self.cut_branches() # Found solution no need to search anymore            
        
    #Mark subproblem solved
    def mark_ready(self, cache : Cache):

        if self.f is not None:
            self.save_best(self.f)
        else:
            self.best = self
        cache.put_solution(self.df, self)
        
        
    def save_best(self, f):
        self.best.f = f
        self.best.left = self.lefts[f].best
        self.best.right = self.rights[f].best
        self.best.lower = self.lower
        self.best.upper = self.upper

    def update_improving(self):
        if self.parent is None:
            return 
        parent = self.parent

        f = self.parent_feat
        if self.isLeft:
            sibling = parent.rights[f]
        else:
            sibling = parent.lefts[f]

        self.improving = min(self.improving, parent.improving - sibling.lower)

    def update_local_bounds(self, cache : Cache):
        childrenUpper = 20000000
        childrenLower = 20000000

        self.update_improving()

        pos_features = cache.get_possbile_feats(self.df)
        #This could be optimized in a future versions by checking if bounds for all childs have been added and only checking 
        # if the bound for the feature we are updating is changing the bound for the whole node
        for feature in pos_features:
            left = self.lefts[feature]
            right =  self.rights[feature]

            upperBound = left.upper + right.upper + 1
            if upperBound < childrenUpper:
                childrenUpper = upperBound
                if upperBound < self.upper:
                    self.best_f = feature
            self.improving = min(self.improving, upperBound-1)

            lowerBound = left.lower + right.upper + 1

            #Prune infeasible children pair
            if lowerBound > self.improving:
                left.cut_branches()
                right.cut_branches()

            childrenLower = min(childrenLower, lowerBound)

        if childrenLower == 20000000:
            childrenLower = 0 #Lower bound is 0 if no feasible childrent have been yet created

        updated = False
        if childrenUpper < self.upper:
            if self.parent is None:
                print("updated the best for root at some point")
            cache.put_upper(self.df, childrenUpper)
            #New best solution found
            self.save_best(self.best_f)
            self.put_node_upper(childrenUpper)
            if cache.put_upper(self.df, childrenUpper):
                cache.put_best(self.df, self.best) # Only if it's better than that we have in the cache
            updated = True

        if childrenLower > self.lower:
            cache.put_lower(self.df, childrenLower)
            self.put_node_lower(childrenLower)
            updated = True

        if not updated:
            return False

        if self.parent is None:
            print(f"Updated local bounds for root lower = {self.lower}, upper = {self.upper}")
        #print("Lefts and rights: ")

        if self.lower == self.upper:
            if self.parent is None:
                print("found root solution")
            self.link_and_prune(self.best, cache)
            self.mark_ready(cache)
        
        return True

        
    def cut_branches(self):
        #print(f"Cutting: node = {str(self)}")
        self.feasible = False
        for left in self.lefts.values():
            left.cut_branches()
        for rigth in self.rights.values():
            rigth.cut_branches()



    def print_solution(self):
        size = 0
        q = queue.Queue()
        q.put(self)
        while not q.empty():
            node = q.get()
            print(node)
            if node.f is None:
                print("ok check this leaf: ")
                print(node.df)
            else:
                size += 1
            if node.left is not None:
                q.put(node.left)
            if node.right is not None:
                q.put(node.right)
        print("size of the tree: ", size)

    def put_node_upper(self, bound):
        previous_upper = self.upper
        self.improving = min(self.improving, bound-1)
        self.upper = min(self.upper, bound)
        # if self.upper != previous_upper:
        #     print(f"Updated node upper bound:node = {str(self)}, new upper = {self.upper}")

    def put_node_lower(self, bound):
        # print("with bound =", bound)
        previous_lower = self.lower
        self.lower = max(self.lower, bound)
        # if self.lower != previous_lower:
        #     print(f"Updated node lower bound: node = {str(self)}, new lower = {self.lower}")

    # def __str__(self):
    #     return "dataset: " + str(self.df) + " parent_feat: " + str(self.parent_feat) + " is_left: " + str(self.isLeft) + " feasible: " + str(self.feasible) + " feature: " + str(self.f) + " upper: " + str(self.upper) + " lower: " + str(self.lower)
    
    def __str__(self):
        if self.parent is None:
            return "root"
        direction = "left" if self.isLeft else "right"
        return str(self.parent) + " " + direction + " " + str(self.parent_feat)

    #Comparison methods otherwise priority queue bricks
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
    
   

