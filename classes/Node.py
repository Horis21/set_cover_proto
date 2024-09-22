from classes.ChildrenReady import ChildrenReady
from classes.Cache import Cache
import copy

class Node:
    def __init__(self, df, parent_feat, parent ,isLeft):
        self.df = df
        self.parent_feat = parent_feat
        self.isLeft = isLeft
        self.f = None
        self.upper = 20000000
        self.lower = 0
        self.feasible = True
        self.solutions = {}
        self.lefts = {}
        self.rights = {}
        self.best = None
        self.left = None
        self.right = None
        self.parent = parent

    def backpropagate(self, cache : Cache):
        print("Backpropagating")
        if not self.feasible:
            return
        if self.parent is None:
            return
        
        parent = self.parent
        f = self.parent_feat

        if parent.update_local_bounds(cache, f):
            #Backpropagate further only if bounds were updated
            parent.backpropagate(cache)

    def link_and_prune(self, solution, cache : Cache):
        if solution.lower > self.upper:
            # print("wrongfully linked")
            self.cut_branches()
            return
        # if self.parent is None:
        #     print("feature for root solution: ", solution.f)

        self.upper = solution.upper
        self.lower = solution.lower
        self.left = solution.left
        self.right = solution.right
        
        self.f = solution.f

        self.feasible = True #Sanity check hehe

        self.prune_siblings(cache)
        self.backpropagate(cache)
        self.mark_ready(cache)

    def prune_siblings(self, cache : Cache):
        parent = self.parent

        if parent is None:
            return
        
        for feat in cache.get_possbile_feats(parent.df):
            if feat != self.parent_feat:
                parent.lefts[feat].cut_branches()
                parent.rights[feat].cut_branches()

    
    def check_ready(self, cache : Cache):

        #If lower bound is not equal to upper means that there still might be better solutions out there
        if self.lower != self.upper or not self.feasible:
            return
        for f in cache.get_possbile_feats(self.df):
            if self.solutions.get(f) is None or self.lefts.get(f) is None or self.rights.get(f) is None:
                continue
            left = self.lefts[f]
            right = self.rights[f]
            #Check that both children from that branch are marked as solved and the solution can't be improved
            if self.solutions[f].left and self.solutions[f].right and cache.get_lower(left.df) + cache.get_lower(right.df) + 1 == self.lower:
                #print(f"Solution found: node = {str(self)}, f = {f}")
                self.f = f
                self.left = self.lefts[f]
                self.right = self.rights[f]
                self.mark_ready(cache)
              
                if self.parent is not None:
                    self.prune_siblings(cache)
                
                return
            
        
    #Mark subproblem solved
    def mark_ready(self, cache : Cache):
        # self.cut_branches()
        # if not self.feasible:
        #     return

        cache.put_solution(self.df, self)
        if self.parent is None:
            return
        f = self.parent_feat
        parent = self.parent

        if parent.solutions.get(f) is None:
            parent.solutions[f] = ChildrenReady()

        if self.isLeft:
            parent.solutions[f].left = True
            if not parent.solutions[f].right and parent.rights.get(f) is not None:
                parent.rights[f].check_ready(cache)
        else:
            parent.solutions[f].right = True
            if not parent.solutions[f].left and parent.lefts.get(f) is not None:
                parent.lefts[f].check_ready(cache)


        #If sibling is solved as well maybe parent is also solved
        if parent.solutions[f].left and parent.solutions[f].right:
            parent.check_ready(cache)
        
    def save_best(self, f):
        best = Node(self.df, self.parent_feat, self.parent, self.isLeft)
        best.f = f
        best.feasible = True #sanity check :()
        best.left = self.lefts[f].best
        best.right = self.rights[f].best
        best.lower = self.lower
        best.upper = self.upper
        self.best = best

    def update_local_bounds(self, cache : Cache, f):

        #Prune subbranch if children pair is infeasible
        if self.lefts.get(f) is not None and self.rights.get(f) is not None and (not self.lefts[f].feasible or not self.rights[f].feasible or self.lefts[f].lower + self.rights[f].lower + 1 > self.upper):
                #print(f"Pruning subrannch: node = {str(self)}, branch = {f},lower = {self.lower}, upper = {self.upper}")
                if self.lefts.get(f) is not None:
                    self.lefts[f].cut_branches()
                if self.rights.get(f) is not None:
                    self.rights[f].cut_branches()
                return False
      
        upper = 20000000
        lower = 20000000

        pos_features = cache.get_possbile_feats(self.df)
        #This could be optimized in a future versions by checking if bounds for all childs have been added and only checking 
        # if the bound for the feature we are updating is changing the bound for the whole node
        for feature in pos_features:
            if self.lefts.get(feature) is None or not self.lefts[feature].feasible or self.rights.get(feature) is None or not self.rights[feature].feasible:
                continue #Don't update with bounds from infeasible children
            upper = min(upper, self.lefts[feature].lower + self.rights[feature].lower + 1)
            lower = min(lower, self.lefts[feature].lower + self.rights[feature].upper + 1)


        updated = False
        if upper < self.upper:
            if self.parent is None:
                print("updated the best for root at some point")
            cache.put_upper(self.df, upper)
            #New best solution found
            self.save_best(f)
            self.put_node_upper(upper)
            cache.put_upper(self.df, upper)
            updated = True
        if lower > self.lower:
            cache.put_lower(self.df, lower)
            self.put_node_lower(lower)
            updated = True

        if not updated:
            return False

        if self.parent is None:
            print(f"Updated local bounds for root lower = {self.lower}, upper = {self.upper}")
        #print("Lefts and rights: ")

        if self.lower == self.upper:
            # if self.parent is None:
                # print("found root solution")
            #self.save_best(f)
            self.link_and_prune(self.best, cache)
            return True #because link and prune already backpropagates

        #Prune whole branch if infeasible
        if self.lower > self.upper:
            #print(f"Pruning: node = {str(self)}, lower = {self.lower}, upper = {self.upper}")
            self.cut_branches()
            return False
        
        return True

        
    def cut_branches(self):
        #print(f"Cutting: node = {str(self)}")
        self.feasible = False
        for left in self.lefts.values():
            left.cut_branches()
        for rigth in self.rights.values():
            rigth.cut_branches()
        # if self.parent is not None:
        #     if self.isLeft:
        #         self.parent.rights[self.parent_feat].cut_branches
        #     else:
        #         self.parent.lefts[self.parent_feat].cut_branches



    def put_node_upper(self, bound):
        previous_upper = self.upper
        self.upper = min(self.upper, bound)
        # if self.upper != previous_upper:
            # print(f"Updated node upper bound:node = {str(self)}, new upper = {self.upper}")

    def put_node_lower(self, bound):
        # print("with bound =", bound)
        previous_lower = self.lower
        self.lower = max(self.lower, bound)
        # if self.lower != previous_lower:
            # print(f"Updated node lower bound: node = {str(self)}, new lower = {self.lower}")

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
    
   

