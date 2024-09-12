from classes.ChildrenBounds import ChildrenBounds
from classes.ChildrenReady import ChildrenReady
from classes.Cache import Cache


class Node:
    def __init__(self, df, parent_feat, parent ,isLeft):
        self.df = df
        self.parent_feat = parent_feat
        self.isLeft = isLeft
        self.lowers = {}
        self.f = None
        self.upper = 20000000
        self.lower = 0
        self.feasible = True
        self.uppers = {}
        self.solutions = {}
        self.lefts = {}
        self.rights = {}
        self.best = None
        self.left = None
        self.right = None
        self.parent = parent


    def prune_siblings(self, cache : Cache):
        parent = self.parent
        for feat in cache.get_possbile_feats(parent.df):
            if feat != self.parent_feat:
                parent.lefts[feat].cut_branches()
                parent.rights[feat].cut_branches()


    def link_and_prune(self, solution, cache : Cache):
        sol= self.deepcopy(solution)
        sol.parent_feat = self.parent_feat
        sol.parent = self.parent 
        sol.parent.f = self.parent_feat
        sol.isLeft = self.isLeft
        if self.isLeft:
            self.parent.lefts[self.parent_feat] = sol
        else:
            self.parent.rights[self.parent_feat] = sol
        sol.prune_siblings(cache)
        sol.mark_ready(cache)
        sol.parent.update_local_bounds(cache, self.parent_feat)
    
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
        

    def update_local_bounds(self, cache : Cache, f):

        #Prune subbranch if children pair is infeasible
        if self.lefts.get(f) is None or self.rights.get(f) is None or not self.lefts[f].feasible or not self.rights[f].feasible or self.lowers[f].left + self.lowers[f].right + 1 > self.upper:
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
        for f in pos_features:
            if (self.lefts.get(f) is not None and not self.lefts[f].feasible) or (self.rights.get(f) is not None and not self.rights[f].feasible):
                continue #Don't update with bounds from infeasible children
            seen = True
            if self.lowers.get(f) is None:
                self.lowers[f] =  ChildrenBounds(True)
            if self.uppers.get(f) is None:
                self.uppers[f] = ChildrenBounds(False)
            upper = min(upper, self.uppers[f].left + self.uppers[f].right + 1)
            lower = min(lower, self.lowers[f].left + self.lowers[f].right + 1)


        updated = False
        if upper < self.upper:
            self.put_node_upper(upper)
            updated = True
        if lower > self.lower:
            self.put_node_lower(lower)
            updated = True

        if not updated:
            return False

        if self.parent is None:
            print(f"Updated local bounds for root lower = {self.lower}, upper = {self.upper}")
        #print("Lefts and rights: ")

        if self.lower == self.upper:
            self.check_ready(cache)
            return True

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

    #Duplicated code can be refactored
    def add_child_lower(self, f, isLeft,bound):
        if self.lowers.get(f) is None:
            self.lowers[f] = ChildrenBounds(True)
        if isLeft:
            self.lowers[f].left = bound
        else:
            self.lowers[f].right = bound
        #print(f"Added child lower bound: node = {str(self)}, isLeft = {isLeft}, bound = {bound}")

    def add_child_upper(self, f, isLeft,bound):
        if self.uppers.get(f) is None:
            self.uppers[f] = ChildrenBounds(False)
        if isLeft:
            self.uppers[f].left = bound
        else:
            self.uppers[f].right = bound
        #print(f"Added child upper bound:node = {str(self)}, isLeft = {isLeft}, bound = {bound}")


    def put_node_upper(self, bound):
        previous_upper = self.upper
        self.upper = min(self.upper, bound)
        #if self.upper != previous_upper:
            #print(f"Updated node upper bound:node = {str(self)}, new upper = {self.upper}")

    def put_node_lower(self, bound):
        previous_lower = self.lower
        self.lower = max(self.lower, bound)
        #if self.lower != previous_lower:
            #print(f"Updated node lower bound: node = {str(self)}, new lower = {self.lower}")

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
    
   

