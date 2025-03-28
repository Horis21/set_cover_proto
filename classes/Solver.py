import time
import pandas as pd
import numpy as np
import gurobipy as grb
from gurobipy import GRB, quicksum
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from classes.Cache import Cache
from classes.Node import Node
from classes.HashableDataFrame import HashableDataFrame
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_distances_argmin_min
import gc
import sys

class Solver:
    def __init__(self, name, sample_size = 200, MIP_gap = 0.05, cover_runs = 1, lower_bound_strategy = 'both'):
        self.sample_size = sample_size
        self.MIP_gap = MIP_gap
        self.cover_runs = cover_runs
        self.lower_bound_strategy = lower_bound_strategy
        self.cache = Cache()
        self.explored = 0
        self.start_time = time.time()
        self.name = name

    def split_differnce_table(self, node : Node, feat):
        parent_dt = node.dt

        # Initialize empty dictionaries
        left_dt = {}
        right_dt = {}

        for dif in parent_dt.keys(): # For each row in the difference table
            if dif[feat] != 0:
                continue # Rows in the dt that mark a difference in the splitting feature are removed by splitting so they won't appear in either children difference table

            # Go through all saved feature values for the dt row. Here each value represents the feature value (0 or 1) that both feature vectors from the dataset had the same i.e. a ZERO in the difference table
            for feats in parent_dt[dif]:
                if feats[feat] == 0: # If they both had a zero, put in the left_dt
                    if left_dt.get(dif) is None:
                        left_dt[dif] = set()

                    # This row in the dt will also be present in the left's child dt
                    left_dt[dif].add(feats)

                else: # Otherwise in right difference table
                    if right_dt.get(dif) is None:
                        right_dt[dif] = set()

                    # Same for right child case
                    right_dt[dif].add(feats)


        # Save dts after splitting
        node.lefts[feat].dt = left_dt
        node.rights[feat].dt = right_dt

        


    def get_difference_table(self, node : Node):
        if node.parent is None: #Always compoute dt for root from scratch
            self.compute_difference_table(node)

        elif not node.dt: #Instead of computing from 0, use parent's dt for faster computation
            self.split_differnce_table(node.parent, node.parent_feat)

        # Store it in the cahce
        cache_entry = self.cache.get_entry(node.df)
        cache_entry.put_dt(node.dt)
        return self.get_dt(node)


    def get_dt(self, node : Node):
        # Create a pd.Dataframe from all the keuys from the difference table dictionary object.
        return pd.DataFrame(node.dt.keys())      



    def compute_difference_table(self, df):
        # Split positive and negative instances (column 0 is the label)
        pos = df[df[0] == 1].iloc[:, 1:].to_numpy()  # Convert to NumPy arrays (faster)
        neg = df[df[0] == 0].iloc[:, 1:].to_numpy()


        # Compute all pairwise XOR differences in a vectorized way

        diff_matrix = np.logical_xor(pos[:, None, :], neg[None, :, :])  # Shape: (len(pos), len(neg), num_features)

        

        # Filter out identical rows

        mask = diff_matrix.sum(axis=2) > 0  # Identify non-identical pairs

        filtered_diffs = diff_matrix[mask]  # Apply mask

        

        # Convert back to DataFrame

        diff_df = pd.DataFrame(filtered_diffs)

        return diff_df


        
    
    def find_set_cover(self, diff_df, verbose=True):
        model = grb.Model("set_cover")
        if not verbose:
            model.setParam('OutputFlag', 0)
            model.Params.LogToConsole = 0
        
        model.setParam('MIPGap', self.MIP_gap)
        # Decision variables
        ncols = len(diff_df.columns)
        x = {}
        for j in range(ncols):
            x[j] = model.addVar(vtype=GRB.BINARY, name=f"x[{j}]")

        # Objective function: minimize the number of subsets used
        model.setObjective(quicksum(x[j] for j in range(ncols)), GRB.MINIMIZE)

        # Constraints: ensure every element is covered
        for i, row in diff_df.iterrows():
            model.addConstr(quicksum(x[j] for j, val in enumerate(row) if val) >= 1, f"cover_{i}")

        # Solve
        # model = model.relax()
        model.optimize()

        # Retrieve solution
        if model.status == GRB.OPTIMAL:
             # Access relaxed variable values
            # relaxed_vars = model.getVars()  # This gets all variables in the relaxed model
            # selected_subsets = [j for j, var in enumerate(relaxed_vars) if var.X > 0.5]
            selected_subsets = [j for j, var in x.items() if var.X > 0.5]
            return selected_subsets
        else:
            print("No solution found")
            sys.exit(1)

    def get_sample(self, df):
        if df.size == 0:
            print("get sample df  is empty")
        labels = np.unique(df[0].values)
        cluster_size = int(np.floor(self.sample_size / len(labels))) # Generate balanced clusters

        sample = pd.DataFrame()
        for label in labels: # For each class compute a cluster
            class_points = df[df[0] == label] # Get all points from the class
            clustered_points = self.apply_clustering(class_points, cluster_size) # Apply clustering algo
            sample = pd.concat([sample, clustered_points]) # Add them to the sample

        if sample.size == 0:
            print("final sample is empty")
            print(df)
        return sample
    
    def get_random_sample(self, df):
        # Random sample alternative
        return df if df.shape[0] <= self.sample_size else df.sample(self.sample_size)

    def apply_clustering(self, df, cluster_size):
        if df.size == 0:
            print("applying clustering on empty sample")

        # If not enough points from this class, then just return all of them
        if df.shape[0] <= cluster_size:
            return df
        
        # Initialize an empty list to hold the representative samples
        representative_samples = []


        # Apply Agglomerative Clustering with Hamming distance

        # DelftBlue syntax
        #clustering = AgglomerativeClustering(n_clusters=cluster_size, affinity='precomputed', linkage = 'average') 
        # Otherwise
        clustering = AgglomerativeClustering(n_clusters=cluster_size, metric='precomputed', linkage = 'average')
        
        # Compute the pairwise Hamming distance matrix
        hamming_distances = pairwise_distances(df, metric='hamming')
        
        data = df.copy()

        # Fit the clustering model
        cluster_labels = clustering.fit_predict(hamming_distances)
        data['cluster'] = cluster_labels

        for cluster_id in range(cluster_size):
            # Get all points in the current cluster
            cluster_points = data[data['cluster'] == cluster_id].drop(columns=['cluster'])

            # Calculate the mean point for the cluster (binary mean vector)
            binary_mean = (cluster_points.mean(axis=0) >= 0.5).astype(int).values.reshape(1, -1)

            # Find the point closest to the binary mean using Hamming distance
            closest_idx, _ = pairwise_distances_argmin_min(binary_mean, cluster_points, metric='hamming')

            representative_samples.append(df.loc[cluster_points.index[closest_idx[0]]])

        representative_samples = pd.DataFrame(representative_samples)
        if representative_samples.size == 0:
            print("representative samples is empty")
            print(df)
        return representative_samples

    def set_cover_features(self, node : Node):
        df = node.df
        lower_bound = 0
        counts = np.zeros(df.shape[1] - 1, dtype=int)
        if df.shape[0] <= self.sample_size:
            diff_table = self.compute_difference_table(df)
            features = self.find_set_cover(diff_table, verbose=False)

            for f in features:
                counts[f] += 1

            return counts, len(features)
        # repeat multiple times
        for i in range(self.cover_runs):
            # Get a random sample from the rows of the difference table (This could be improved by using clustering)
            #sample = self.get_sample(df)
            sample = self.get_random_sample(df)
            diff_table = self.compute_difference_table(sample)
            features = self.find_set_cover(diff_table, verbose=False)

            # the lower bound is the maximum of all the lower bounds we have obtained
            if len(features) > lower_bound:
                lower_bound = len(features)
                best_cover = features
            #print(f"Iteration {i+1} resulting features: {features}")
            # Count the features used. 
            for f in features:
                counts[f] += 1
        
        # print the features in order of importance
        feature_order = np.argsort(counts)[::-1]
        # for f in feature_order:
        #     if counts[f] == 0: break
            #print(f"Feature {f}: required {counts[f]} times")

        # print the final lower bound
        #print(f"Require at least {lower_bound} features")
        return counts, lower_bound
        

        
    def one_off_features(self, df):
        pos_features = self.possible_features(df)

        # Split positive and negative instances (column 0 is the label)
        pos = set(
                    tuple(
                        x[i + 1] if i in pos_features else 0  # Adjust for column 1 = feature 0
                        for i in range(len(x) - 1)            # Skip the label column
                    )
                    for x in df[df[0] == 1].values            # Select positive instances
                )
        neg = set(
                    tuple(
                        x[i + 1] if i in pos_features else 0
                        for i in range(len(x) - 1)
                    )
                    for x in df[df[0] == 0].values            # Select negative instances
                )

        features = set() # Initialize one-off features foubd as a set

        # Search from the class with least number of instances
        if len(pos) < len(neg):
            search = pos
            opp = neg
        else:
            search = neg
            opp = pos
        
        for row in search:
            for f in pos_features:
                if f in features: #If we already found a one-off based on this feature continue
                    continue
                new_row = np.copy(row)
                new_row[f] = np.logical_xor(1, new_row[f]) #Compute the off-by-one feature vector
                new_row = tuple(new_row)
                #Check in the opposing class if it exists and add to the found features
                if opp.__contains__(new_row):
                    features.add(f)
        return list(features)

    #Split data based on feature f
    def split(self, hashable_df : HashableDataFrame, f):
        """
        Splits the dataset based on the given feature.

        Args:
            hashable_df (HashableDataFrame): The dataset to split.
            feature (int): The feature index for splitting.

        Returns:
            Tuple[HashableDataFrame, HashableDataFrame]: Left and right subsets.
        """
        df = hashable_df.get_df()
        indices = hashable_df.get_indices()
        condition = df.loc[indices, f + 1] == 0  # Adjust for feature index
        left = hashable_df.subset(condition)
        right = hashable_df.subset(~condition)
        return left, right

    #Check if the dataset is pure
    def check_leaf(self, df):
        if df.empty or df[df[0] == 0].size == 0 or df[df[0] == 1].size == 0:
            return True
        else:
            return False

    def transformTree(self, node_id, tree, parent=None, isLeft = None):
        if tree is None:
            return Node(parent =  parent,isLeft = isLeft)
        # Create a new node object
        if tree.children_left[node_id] != tree.children_right[node_id]:  # It's a decision node
            feature = tree.feature[node_id]
            new_node = Node(parent=parent, isLeft = isLeft)
            new_node.f = feature
            if parent is not None:
                new_node.parent_feat = parent.f
            
            # Recursively create left and right children
            left_child = self.transformTree(tree.children_left[node_id],tree ,new_node, True)
            right_child = self.transformTree(tree.children_right[node_id],tree, new_node, False)

            left_child.parent_feat = feature
            right_child.parent_feat = feature

            left_child.isLeft = True
            right_child.isLeft = False
            
            # Add children to the current node
            new_node.left = left_child
            new_node.right = right_child

        else:  # It's a leaf node
            new_node = Node(parent_feat = parent.f, parent= parent, isLeft = isLeft)
        
        return new_node

    #Fast forward upper bound of a dataset
    def fast_forward(self, df):
        if self.check_leaf(df):
            return None
        else:
            X = df[df.columns[1:]]
            cart = DecisionTreeClassifier()
            cart.fit(X, df[0])
            return cart
        
    def possible_features(self, df, parent = None):
            # Check if already in cache
            cache_entry = self.cache.get_entry(df)
            features = cache_entry.get_possbile_feats()
            if features is not None:
                return features
        
            # Initialize with all features in the dataset
            features = set([i for i in range(df.shape[1]-1)])
            features_present = np.zeros(df.shape[1]-1)
            features_always_present = np.ones(df.shape[1]-1)

            for i, row in df.iterrows():
                features_present = np.logical_or(features_present, row[1:]) # Mark features that are present at least once in the dataset (OR)
                features_always_present = np.logical_and(features_always_present, row[1:]) # Mark features that are always present in the dataset (AND)

            for i, f in enumerate(features_present):
                if not f: 
                    features.remove(i) #Remove all features that aren't present in any instance

            for i, f in enumerate(features_always_present):
                if f: 
                    features.remove(i) #Remove all features that are present in all instances

            # Get the possible features of the present node, and compute intersection to remove features that weren't possible for the parent and therefore, won't be in the child as well. This speeds up the next part
            if parent is not None:
                parent_cache_entry = self.cache.get_entry(parent.df)
                parent_possible_feats = parent_cache_entry.get_possbile_feats()
                features = set(features) & set(parent_possible_feats)

            # Iterate collumns to remove duplicat or complementary features among columns
            cols = set()
            for i, col in enumerate(df.columns[1:]):
                if i not in features:
                    continue #Don't bother with already ignored features

                if col in cols:
                    features.remove(i) #Redundant feature, since the column is already present in the set
                else:
                    cols.add(col)
            
            for i, col in enumerate(df.columns[1:]):
                if i not in features:
                    continue #Don't bother with already ignored features

                #Check if its complement feature is present
                complement = np.logical_xor(col,col)
                if complement in cols: #Check if the complement of this feature exists
                    features.remove(i)
                    cols.remove(col) #Remove so that we don't remove the complement as well, when we get to the feature

            cache_entry.put_possible_feats(features)
            return features
        
    #Return one_offs, cover_features and set cover lower bound    
    def get_features(self, node : Node):
        df = node.df
        parent = node.parent
        feature = node.parent_feat # This is the feature that the parent splits on to create this current node
        cache_entry = self.cache.get_entry(node.df)
        one_offs = cache_entry.get_one_offs()
        if one_offs is None and (self.lower_bound_strategy == 'one-off' or self.lower_bound_strategy == 'both'):
            if parent is None:
                # Compute one-off features from scratch for the root node
                one_offs = self.one_off_features(df)
                cache_entry.put_one_offs(one_offs)
            else:
                # Otherwise simply remove the parent_feat from the list, since it was used to split 
                parent_cache_entry = self.cache.get_entry(parent.df)
                one_offs = parent_cache_entry.get_one_offs()
                if feature in one_offs:
                    one_offs.remove(feature)
                cache_entry.put_one_offs(one_offs)
        cover_features =  cache_entry.get_set_cover()
        set_cover_lower_bound = 0
        # Get set-cover lower bound from cache or compute if not present
        if cover_features is None and (self.lower_bound_strategy == 'set-cover' or self.lower_bound_strategy == 'both'):
            cover_features, set_cover_lower_bound = self.set_cover_features(node)
            cache_entry.put_set_cover(cover_features)
        return one_offs, cover_features, set_cover_lower_bound
        

    def computeUB(self, node: Node):
        data = node.df
        cache_entry = self.cache.get_entry(node.df)
        if cache_entry.get_upper() is None:
            cart = self.fast_forward(data) #Compute fast forward upper bound
            ff = cart.get_n_leaves() - 1 if cart is not None else 0

            # Save upper bound
            cache_entry.put_upper(ff)
            # Transform the CART tree to our Node class
            best = self.transformTree(0, cart.tree_, node.parent, node.isLeft)
            # Store the node
            cache_entry.put_best(best)
        node.put_node_upper(cache_entry.get_upper()) #Add the dataset upper bound to the node upper bound
        node.best = cache_entry.get_best()
        node.best.parent_feat = node.parent_feat # Updae the parent_feat of the best solution, since it is possible that a different one was used when splititng the parent, than for the one stored in the cache

    def computeLB(self, node: Node):
        data = node.df
        cache_entry = self.cache.get_entry(node.df)     
        if cache_entry.get_lower() is None:
            one_offs, cover_features, set_cover_lower_bound = self.get_features(node) # Retrieve the one-off features, set-cover features and set-cover lowe bound
            if self.lower_bound_strategy == 'both':     
                local_lower_bound = max(len(one_offs), set_cover_lower_bound) # If both lower bound strategies are used, use the maximum
            elif self.lower_bound_strategy == 'one-off':
                local_lower_bound = len(one_offs)
            elif self.lower_bound_strategy == 'set-cover':
                local_lower_bound = set_cover_lower_bound
            cache_entry.put_lower(local_lower_bound) #Add lower bound based on set cover and one_offs
        node.put_node_lower(cache_entry.get_lower())


    def mark_leaf(self, node : Node):
        # Set and store all bounds to 0
        cache_entry = self.cache.get_entry(node.df)
        cache_entry.put_lower(0)
        node.put_node_lower(0)
        node.put_node_upper(0)
        cache_entry.put_upper(0)
        node.best = node
        node.df = None # Garbage collector pls
        node.mark_ready(self.cache) # Node is solved

    def find_next(self, node : Node):
        while not node.pq.empty():
            next = node.get_pq()
            if not next.feasible:
                continue # Skip infeasible nodes and pop from PQ 

            if not next.expanded:
                return next # Feasible and not expanded node is good!
            
            next_child = self.find_next(next) # Recursive call on expanded, but still feasibile node to find best children node to expand
            if next_child is not None:
                return next_child # Return the found node, since it's good (only other return checks for not expanded)
            
            # Else go to next node, don't care about next to add back to PQ since no feasibile children nodes
        
        return None # If no feasible next node found
    
    def update_sibling(self, node : Node):
        parent = node.parent

        if not node.sibling.feasible:
            return # idc for infeasible nodes
        
        add_back = []
        while not parent.pq.empty():
            next = parent.get_pq()
            if next.feasible:
                add_back.append(next) # Save popped nodes for readding

            if next.parent_feat == node.parent_feat: # Found the sibling
                break 

        for sibling in add_back:
            node.parent.add_to_queue(sibling) # Add back the popped nodes
    
    def add_back_to_pq(self, node : Node):
        parent = node.parent
        if parent is None:
            return # No need to add these nodes back (we only talking about root here most likely)
        
        if not node.feasible:
            node.parent = None # Clean references for gc maybe only if node is infeasible anyway
        else:
            self.update_sibling(node) # Update priority of sibling
            parent.add_to_queue(node) # Add back to parent's PQ

        self.add_back_to_pq(parent) # Recursive call for parent since it was popped as well

    def solve(self, orig_df):
        df = HashableDataFrame(orig_df)    
        root = Node(df)

        # UB and LB for root
        self.computeLB(root)
        self.computeUB(root)

        # Expand root node
        self.expand_node(root)

        while not root.pq.empty():
            if time.time() - self.start_time > 3600:
                return -1, -1, -1, -1
            node = self.find_next(root)
            if node is None:
                break # No nodes left to search \ expand
            
            self.expand_node(node)

            self.add_back_to_pq(node) # Add back to PQ to reupdate order

        # size, depth = root.print_solution()
        # root._recursive_print_tree()
        # self.cache.write_bounds(self.name)
       
        # avg_question_length = root.queryAll(orig_df)
        
        # print("time: ", elapsed_time)
        return root.upper, 1, 1, 1
        #return size, depth, self.explored, avg_question_length

    def expand_node(self, node : Node):
        node.expanded = True
        data = node.df
        cache_entry = self.cache.get_entry(node.df)
        solution = cache_entry.get_solution() #Check if solution already found
        if solution is not None:
            # Update with bounds from the solution
            node.lower = solution.lower
            node.upper = solution.upper

            node.improving = solution.improving

            node.save_best(solution.f, solution)

            # Link with solution optimal feature and resulting children
            node.link_and_prune(solution, self.cache)

            return
    
        self.explored += 1  #Only updated explored nodes if no solutuion already in cache

        need_LB = [] #Store nodes for which computing the LB might be needed
        early_solution = False

        #Search for all features
        pos_features = self.possible_features(data, node.parent)

        one_offs, cover_features, set_cover_lower_bound = self.get_features(node)
        for i in pos_features:
            #Split the data based on feature i
            left_df, right_df = self.split(data, i)

            # Store whether a one-off feature was used for splitting
            is_one_off_child = True if one_offs is not None and i in one_offs else False 

            # Count how important the feature to split is in the set-cover
            set_cover_counts = cover_features[i] if cover_features is not None else 0

            # Create children nodes
            left = Node(left_df, i, node, True, is_one_off_child,set_cover_counts)
            right = Node(right_df, i, node, False, is_one_off_child, set_cover_counts)

            # Sibling references
            left.sibling = right
            right.sibling = left

            # If leaf, mark as leaf. Otherwise compute the UB (fast-forward) and store for possible later computation of LB
            if self.check_leaf(left_df):
                self.mark_leaf(left)
            else:
                need_LB.append(left)
                self.computeUB(left)
            if self.check_leaf(right_df):
                self.mark_leaf(right)
            else:
                need_LB.append(right)
                self.computeUB(right)               

            # Store a reference in the parent for all children
            node.lefts[i] = left
            node.rights[i] = right

            if left.upper + right.upper + 1 == node.lower: #Early solution found
                # Update upper bound
                node.upper = node.lower

                # Save best solution (actual solution)
                node.save_best(i)

                # Link with the solution
                node.link_and_prune(node.best, self.cache)
                node.mark_ready(self.cache) # Node is solved
                early_solution = True
                break #solution found here no need to search further
        
    
        if not early_solution: # If no early solution, then compute LB for all non-leaf children nodes and backpropagate
            for child in need_LB:
                self.computeLB(child)
                
                node.add_to_queue(child)

            node.backpropagate(self.cache)


       