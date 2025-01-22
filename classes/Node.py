import queue
from classes.CustomPQ import CustomPQ
from classes.Cache import Cache

class Node:
    def __init__(self, df = None, parent_feat = None, parent = None ,isLeft = None, is_one_off_child = None, set_cover_counts = None):
        self.df = df
        self.parent_feat = parent_feat
        self.isLeft = isLeft
        self.f = None
        self.upper = 200000
        self.improving = 200000
        self.lower = 0
        self.dt = {}
        self.depth = 0 if parent is None else parent.depth + 1
        self.feasible = True
        self.is_one_off_child = is_one_off_child
        self.set_cover_counts = set_cover_counts
        self.lefts = {}
        self.rights = {}
        self.best = None
        self.left = None
        self.right = None
        self.parent = parent
        self.expanded = False
        self.pq = queue.PriorityQueue() # Local priority queue for hierarchical management

    def add_to_queue(self, child):
        self.pq.put(child)

    def peek_pq(self):
        return self.pq.peek()
    
    def get_pq(self):
        return self.pq.get()

    def backpropagate(self, cache : Cache):
        parent = self.parent
        if self.update_local_bounds(cache) and parent is not None: # Update local bounds
             parent.backpropagate(cache) #Backpropagate further only if bounds were updated


    def link_and_prune(self, solution, cache : Cache):
        print(f"link and prune for {self}")
        # Save the optimal feature left and right children that occur after splitting on it
        self.f = solution.f
        self.left = solution.left
        self.right = solution.right
        
        # Backpropagate the now solved now
        if self.parent is not None:
            self.parent.backpropagate(cache)
        self.cut_branches() # Found solution no need to search anymore in this subtree         
        
    #Mark subproblem solved
    def mark_ready(self, cache : Cache): 
  
        self.best = self #Sanity check to ensure solution stored in the cache have a best tree saved. Also helps garbage collector as less distinct nodes are referenced
        
        # Store solution in cache
        cache_entry = cache.get_entry(self.df)
        cache_entry.put_solution(self)
        
        
    def save_best(self, f, from_sol = None):
        # Save the best feature to split found so far
        self.best.f = f

        
        if from_sol is None: # Save the best left and right children after splitting
            self.best.left = self.lefts[f].best
            self.best.right = self.rights[f].best
        else: # Else if this function is called for linking with a solution stored in the cache, then update with the left and right children that the solution has. The lef and right chuildren objects of the original node are not fully solved yet
            self.best.left = from_sol.left
            self.best.right = from_sol.right

        # Also save the best bounds
        self.best.lower = self.lower
        self.best.upper = self.upper

    def update_improving(self):
        # If root node, then the improving UB = best solution found so far -1. This value gets put in the improving bound when updating the UB (put_node_upper method)
        if self.parent is None:
            return 
        parent = self.parent

        f = self.parent_feat
        if self.isLeft:
            sibling = parent.rights[f]
        else:
            sibling = parent.lefts[f]

        # Compute the improving bound with the sibling LB and parent improving UB
        self.improving = min(self.improving, parent.improving - sibling.lower, self.upper-1)

    def update_local_bounds(self, cache : Cache):
        # Initilaize UB and LB that come from the children nodes with infeasibly high values
        childrenUpper = 20000000
        childrenLower = 20000000

        # Flag for whether bounds got updated
        updated = False

        # Check if there is a better UB in the cache
        cache_entry = cache.get_entry(self.df)
        cachedUB = cache_entry.get_upper()
        if cachedUB < self.upper:
            self.upper = cachedUB
            self.best = cache_entry.get_best() # Also get the best solution so far, since there is a better bound in the cache, it means that a better solution has been found

            updated = True

        cachedLB = cache_entry.get_lower()
        if cachedLB > self.lower:
            self.lower = cachedLB
            updated = True

        # For all splitting features
        pos_features = cache_entry.get_possbile_feats()
        for feature in pos_features:
            left = self.lefts[feature]
            right =  self.rights[feature]

           
            upperBound = left.upper + right.upper + 1 # classic UB
            

            if upperBound < childrenUpper: # If this is the best solution found so far
                childrenUpper = upperBound 
                best_feat = feature # Store the feature responsible

            childrenLower = min(childrenLower,  left.lower + right.lower + 1)
            

        #New best solution found
        if childrenUpper < self.upper:  # If we found a better solution
            self.save_best(best_feat) # Save best solution
            self.put_node_upper(childrenUpper) # Store the solution size
            if cache_entry.put_upper(childrenUpper): # Update the cache
                cache_entry.put_best(self.best) # Only if it's better than that we have in the cache (should always be true as no multi threading or paralization yet)
            updated = True

        # Same with the lower bound, store and update
        if childrenLower > self.lower:
            cache_entry.put_lower(childrenLower)
            self.put_node_lower(childrenLower)
            updated = True

        # Update the improving bound of the node. This will be useful for pruning now infeasible children. I think this also must be done after the previous computations as to have the most up-to-date UB
        self.update_improving()

        if not updated: # If no bounds were updated stop backpropagation
            return False


        # Print when root bounds get updated
        if self.parent is None:
            print(f"Updated local bounds for root lower = {self.lower}, upper = {self.upper}")

        print(f"Updated local bounds for node: {self}")
        print(f"lower = {self.lower}, upper = {self.upper}")

        #Cannot improve anymore if ..
        if self.lower == self.upper or self.lower > self.improving:
            if self.parent is None:
                print("found root solution")
            self.link_and_prune(self.best, cache) # Save best solution for this node
            if self.lower == self.upper:
                self.mark_ready(cache) #Store solution in cache / mark node as solved only if UB == LB, otherwise we just ran out of nodes to use

            return False # Return false simply because link_and_prune also forwards the propgation, so no need to do it twice
        else: # If we can still improve, prune infeasible children
            for feature in pos_features:
                left = self.lefts[feature]
                right =  self.rights[feature]

                if left.lower + right.lower + 1 > self.improving: # If LB > improving, no need to ever explore
                    left.cut_branches_infeasible()
                    right.cut_branches_infeasible()

        return True # continue backpropagation
    
    def cut_branches_infeasible(self):
        self.feasible = False
        self.df = None # Remove the dataframe reference so garbage collector could maybe hopefully pick it up
        self.best = None # Remove references to other nodes
    
        # Prune all children as well
        for left in self.lefts.values():
            left.cut_branches_infeasible() 
        for right in self.rights.values():
            right.cut_branches_infeasible()

        # Remove references to children
        self.lefts = {}
        self.rights = {}

    # Method that prunes a subtree
    def cut_branches(self):
        self.feasible = False # Mark node as infeasible
        self.df = None # Remove the dataframe reference so garbage collector could maybe hopefully pick it up
        self.best = self # We still need to save the best solution, since we will be cutting the branches off of solved subtrees as well. 
        # Prune all children as well
        for left in self.lefts.values():
            left.cut_branches() 
        for right in self.rights.values():
            right.cut_branches()

        # Remove references to children
        self.lefts = {}
        self.rights = {}

    # Some printing 
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
        self.improving = min(self.improving, bound-1) # Also update improving bound
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
        if self.depth < other.depth:
            return True
        if self.depth > other.depth:
            return False
        if self.lower < other.lower:
            return True
        if self.lower > other.lower:
            return False
        if self.is_one_off_child and not other.is_one_off_child:
            return True
        if other.is_one_off_child:
            return False
        if self.set_cover_counts > other.set_cover_counts:
            return True
        if self.set_cover_counts < other.set_cover_counts:
            return False
        if self.upper < other.upper:
            return True
        return False

    def __le__(self, other):
        if not isinstance(other, Node):
            raise TypeError('Can only compare two Nodes')
        if not self.feasible:
            return True
        if self.depth < other.depth:
            return True
        if self.depth > other.depth:
            return False
        if self.lower < other.lower:
            return True
        if self.lower > other.lower:
            return False
        if self.is_one_off_child and not other.is_one_off_child:
            return True
        if other.is_one_off_child:
            return False
        if self.set_cover_counts > other.set_cover_counts:
            return True
        if self.set_cover_counts < other.set_cover_counts:
            return False
        if self.upper <= other.upper:
            return True
        return False

   

