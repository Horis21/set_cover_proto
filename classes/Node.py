import queue
from classes.Cache import Cache

class Node:
    def __init__(self, df = None, parent_feat = None, parent = None ,isLeft = None, is_one_off_child = None, set_cover_counts = None):
        self.df = df
        self.parent_feat = parent_feat
        self.isLeft = isLeft
        self.f = None
        self.upper = 20000000
        self.improving = 20000000
        self.lower = 0
        self.dt = {}
        self.feasible = True
        self.is_one_off_child = is_one_off_child
        self.set_cover_counts = set_cover_counts
        self.lefts = {}
        self.rights = {}
        self.best = None
        self.left = None
        self.right = None
        self.parent = parent

    def backpropagate(self, cache : Cache):
        if not self.feasible:
            return
        
        parent = self.parent
        if self.update_local_bounds(cache) and parent is not None:
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
        self.parent = None       
        self.best = self #Sanity check
        
        cache.put_solution(self.df, self)
        
        
    def save_best(self, f, from_sol = None):
        self.best.f = f
        if from_sol is None:
            self.best.left = self.lefts[f].best
            self.best.right = self.rights[f].best
        else:
            self.best.left = from_sol.left
            self.best.right = from_sol.right
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

        self.improving = min(self.improving, parent.improving - sibling.lower, self.upper-1)

    def update_local_bounds(self, cache : Cache):
        childrenUpper = 20000000
        childrenLower = 20000000

        updated = False

        cachedUB = cache.get_upper(self.df)
        if cachedUB < self.upper:
            self.upper = cachedUB
            self.best = cache.get_best(self.df)
        
            updated = True


        pos_features = cache.get_possbile_feats(self.df)
        #This could be optimized in a future versions by checking if bounds for all childs have been added and only checking 
        # if the bound for the feature we are updating is changing the bound for the whole node
        for feature in pos_features:
            left = self.lefts[feature]
            right =  self.rights[feature]

            upperBound = left.upper + right.upper + 1

            if upperBound < childrenUpper:
                childrenUpper = upperBound
                best_feat = feature

            childrenLower = min(childrenLower,  left.lower + right.upper + 1)

        #New best solution found
        if childrenUpper < self.upper:  
            self.save_best(best_feat)
            self.put_node_upper(childrenUpper)
            if cache.put_upper(self.df, childrenUpper):
                cache.put_best(self.df, self.best) # Only if it's better than that we have in the cache (everytime anyway I think)
            updated = True

        if childrenLower > self.lower:
            cache.put_lower(self.df, childrenLower)
            self.put_node_lower(childrenLower)
            updated = True

        self.update_improving()

        if not updated:
            return False

        if self.parent is None:
            print(f"Updated local bounds for root lower = {self.lower}, upper = {self.upper}")

        #Cannot improve anymore
        if self.lower == self.upper or self.lower > self.improving:
            if self.parent is None:
                print("found root solution")
            self.link_and_prune(self.best, cache)
            if self.lower == self.upper:
                self.mark_ready(cache) #Only if optimal solution is verified

            return False
        else:
            for feature in pos_features:
                left = self.lefts[feature]
                right =  self.rights[feature]

                if left.lower + right.lower + 1 > self.improving:
                    left.cut_branches()
                    right.cut_branches()

        return True

        
    def cut_branches(self):
        self.feasible = False
        self.df = None
        self.best = self
        self.parent = None
        for left in self.lefts.values():
            left.cut_branches()
        for right in self.rights.values():
            right.cut_branches()
        self.lefts = {}
        self.rights = {}

    def print_solution(self):
        size = 0
        depth = 0
        q = queue.Queue()
        q.put((self, 0))
        while not q.empty():
            pair = q.get()
            node = pair[0]
            depth = max(depth,pair[1])
            print(node)
            if node.f is None:
                print("above is leaf")
            else:
                size += 1
            if node.left is not None:
                q.put((node.left, pair[1] + 1))
            if node.right is not None:
                q.put((node.right, pair[1] + 1))
        return size, depth

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

    #Comparison methods for Node class
    def __eq__(self, other):
        if not isinstance(other, Node):
            raise TypeError('Can only compare two Nodes')
        if self.parent == other.parent and self.parent_feat == other.parent_feat:
            return True
        else:
            return False   

        

    def __lt__(self, other):
        if not isinstance(other, Node):
            raise TypeError('Can only compare two Nodes')
        if not self.feasible:
            return True
        if self.lower < other.lower:
            return True
        elif self.lower == other.lower and self.is_one_off_child and not other.is_one_off_child:
            return True
        elif self.lower == other.lower and self.set_cover_counts > other.set_cover_counts:
            return True
        return False

    def __le__(self, other):
        if not isinstance(other, Node):
            raise TypeError('Can only compare two Nodes')
        if not self.feasible:
            return True
        if self.lower < other.lower:
            return True
        elif self.lower == other.lower and self.is_one_off_child and not other.is_one_off_child:
            return True
        elif self.lower == other.lower and self.set_cover_counts >= other.set_cover_counts:
            return True
        return False

   

