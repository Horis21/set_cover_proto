import pandas as pd
import numpy as np
import gurobipy as grb
import queue
from gurobipy import GRB, quicksum
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from classes.Node import Node
from classes.Cache import Cache
from classes.Cache import HashableDataFrame
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_distances_argmin_min
import gc
import sys

class Solver:
    def __init__(self, sample_size = 200, MIP_gap = 0.05, cover_runs = 1, lower_bound_strategy = 'both'):
        self.sample_size = sample_size
        self.MIP_gap = MIP_gap
        self.cover_runs = cover_runs
        self.lower_bound_strategy = lower_bound_strategy
        self.cache = Cache()

    def get_difference_table(self, df):
        # Split positive and negative instances (column 0 is the label)
        pos = df[df[0] == 1]
        neg = df[df[0] == 0]

        # for every positive instance, and for every negative instance, compute the differences using XOR
        diffs = []
        for i, row1 in pos.iterrows():
            for j, row2 in neg.iterrows():
                dif = np.logical_xor(row1[1:], row2[1:]) # exclude label column
                if sum(dif) == 0: continue # Do not add rows that are identical. We can never split these two instances
                diffs.append(dif)
        
        if len(diffs) == 0:
            print("coaie")
            print(df)
        diff_df = pd.concat(diffs, axis=1).T
        return diff_df

    def find_vertex_cover(self, diff_df, verbose=True):
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
        model.optimize()

        # Retrieve solution
        if model.status == GRB.OPTIMAL:
            selected_subsets = [j for j, var in x.items() if var.X > 0.5]
            return selected_subsets
        else:
            print("No solution found")
            sys.exit(1)

    def get_sample(self, df):
        if df.size == 0:
            print("get sample df  is empty")
        labels = np.unique(df[0].values)
        cluster_size = int(np.floor(self.sample_size / len(labels)))

        sample = pd.DataFrame()
        for label in labels:
            class_points = df[df[0] == label]
            clustered_points = self.apply_clustering(class_points, cluster_size)
            sample = pd.concat([sample, clustered_points])

        if sample.size == 0:
            print("final sample is empty")
            print(df)
        return sample

    def apply_clustering(self, df, cluster_size):
        if df.size == 0:
            print("applying clustering on empty sample")
        if df.shape[0] <= cluster_size:
            return df
        
        # Initialize an empty list to hold the representative samples
        representative_samples = []


        # Apply Agglomerative Clustering with Hamming distance
        #clustering = AgglomerativeClustering(n_clusters=cluster_size, affinity='precomputed', linkage = 'average')
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

    def vertex_cover_features(self, df):
        lower_bound = 0
        counts = np.zeros(df.shape[1] - 1, dtype=int)
        # repeat multiple times
        for i in range(self.cover_runs):
            # Get a random sample from the rows (This could be improved by using clustering)
            sample = self.get_sample(df)
            diff_table = self.get_difference_table(sample)
            features = self.find_vertex_cover(diff_table, verbose=False)
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
        pos = set(tuple(x) for x in df[df[0] == 1].values)
        neg = set(tuple(x) for x in df[df[0] == 0].values)

        ignore_feaures = set()
        features = []

        if len(pos) < len(neg):
            search = pos
            opp = neg
        else:
            search = neg
            opp = pos
        
        for row in search:
            for f in pos_features:
                if f in ignore_feaures: #If we already found a one-off based on this feature continue
                    continue
                new_row = np.copy(row)
                new_row[f+1] = np.logical_xor(1, new_row[f+1]) #Compute the off-by-one feature vector
                new_row[0] = np.logical_xor(1, new_row[0])
                new_row = tuple(new_row)
                #Check in the opposing class
                if opp.__contains__(new_row):
                    features.append(f)
                    ignore_feaures.add(f)
        return features

    #Split data based on feature f
    def split(self, df, f):
        return HashableDataFrame(df[df[f+1] == 0]), HashableDataFrame(df[df[f+1]==1])

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
            features = self.cache.get_possbile_feats(df)
            if features is not None:
                return features
        
            features = set([i for i in range(df.shape[1]-1)])
            features_present = np.zeros(df.shape[1]-1)
            features_always_present = np.ones(df.shape[1]-1)

            for i, row in df.iterrows():
                features_present = np.logical_or(features_present, row[1:])
                features_always_present = np.logical_and(features_always_present, row[1:])

            for i, f in enumerate(features_present):
                if not f: 
                    features.remove(i) #Remove all features that aren't present in any instance

            for i, f in enumerate(features_always_present):
                if f: 
                    features.remove(i) #Remove all features that are present in all instances

            if parent is not None:
                parent_pos_feats = self.cache.get_possbile_feats(parent.df)
                features = set(features) & set(parent_pos_feats)

            cols = set()
            for i, col in enumerate(df.columns[1:]):
                if i not in features:
                    continue #Don't bother with already ignored features

                if col in cols:
                    features.remove(i) #Redundant feature
                else:
                    cols.add(col)
            
            for i, col in enumerate(df.columns[1:]):
                if i not in features:
                    continue #Don't bother with already ignored features

                #Check if its complement feature is present
                complement = np.logical_xor(col,col)
                if complement in cols: #Check if the complement of this feature exists
                    features.remove(i)
                    cols.remove(col) #Remove so that we don't remove the complement as well

            self.cache.put_possible_feats(df, features)
            return features
        
    #Return one_offs, cover_features and vertex cover lower bound    
    def get_features(self, df, parent = None, feature = None):
        one_offs = self.cache.get_one_offs(df)
        if one_offs is None and (self.lower_bound_strategy == 'one-off' or self.lower_bound_strategy == 'both'):
            if parent is None:
                one_offs = self.one_off_features(df)
                self.cache.put_one_offs(df, one_offs)
            else:
                one_offs = self.cache.get_one_offs(parent.df)
                if feature in one_offs:
                    one_offs.remove(feature)
                self.cache.put_one_offs(df, one_offs)
        cover_features = self.cache.get_vertex_cover(df)
        vclb = 0
        if cover_features is None and (self.lower_bound_strategy == 'set-cover' or self.lower_bound_strategy == 'both'):
            cover_features, vclb = self.vertex_cover_features(df)
            self.cache.put_vertex_cover(df, cover_features)
        return one_offs, cover_features, vclb
        

    def computeUB(self, node: Node):
        data = node.df
        if self.cache.get_upper(data) is None:
            cart = self.fast_forward(data) #Compute fast forward upper bound
            ff = cart.get_n_leaves() - 1 if cart is not None else 0

            self.cache.put_upper(data, ff)
            best = self.transformTree(0, cart.tree_, node.parent, node.isLeft)
            self.cache.put_best(data, best)
        node.put_node_upper(self.cache.get_upper(data)) #Add the dataset upper bound to the node upper bound
        node.best = self.cache.get_best(data)
        node.best.parent_feat = node.parent_feat

    def computeLB(self, node: Node):
        data = node.df     
        if self.cache.get_lower(data) is None:
            one_offs, cover_features, vclb = self.get_features(data, node.parent, node.parent_feat)
            if self.lower_bound_strategy == 'both':     
                llb = max(len(one_offs), vclb)
            elif self.lower_bound_strategy == 'one-off':
                llb = len(one_offs)
            elif self.lower_bound_strategy == 'set-cover':
                llb = vclb
            self.cache.put_lower(data, llb) #Add lower bound based on vertex cover and one_offs
        node.put_node_lower(self.cache.get_lower(data))


    def mark_leaf(self, node : Node):
        self.cache.put_lower(node.df, 0)
        node.put_node_lower(0)
        node.put_node_upper(0)
        self.cache.put_upper(node.df, 0)
        self.df = None
        node.mark_ready(self.cache)

    def solve(self, orig_df):
        df = HashableDataFrame(orig_df.copy())
        pq = queue.PriorityQueue()
        explored = 0
        
        root = Node(df)
        #Add the root
        pq.put(root)
        self.computeLB(root)
        self.computeUB(root)
        while not pq.empty():
            node = pq.get()
            if not node.feasible: 
                continue #Skip nodes deemed unfeasible

            data = node.df
            solution = self.cache.get_solution(data) #Check if solution already found
            if solution is not None:
                node.lower = solution.lower
                node.upper = solution.upper
                node.improving = solution.improving
                node.link_and_prune(solution, self.cache)
                node.best = node
                continue
        
            explored += 1 #Only updated explored nodes if not a leaf and not in cache

            need_LB = [] #Store nodes for which computing the LB might be needed
            early_solution = False

            #Search for all features
            pos_features = self.possible_features(data, node.parent)

            one_offs, cover_features, vclb = self.get_features(data)
            for i in pos_features:
                #Split the data based on feature i
                left_df, right_df = self.split(data, i)
                is_one_off_child = True if one_offs is not None and i in one_offs else False
                set_cover_counts = cover_features[i] if cover_features is not None else 0
                left = Node(left_df, i, node, True, is_one_off_child,set_cover_counts)
                right = Node(right_df, i, node, False, is_one_off_child, set_cover_counts)

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

                node.lefts[i] = left
                node.rights[i] = right

                if left.upper + right.upper + 1 == node.lower: #Solution found
                    node.upper = node.lower
                    node.save_best(i)
                    node.link_and_prune(node.best, self.cache)
                    node.mark_ready(self.cache)
                    early_solution = True
                    break #solution found here no need to search further
            
        
            if not early_solution:
                for child in need_LB:
                    self.computeLB(child)
                    
                    pq.put(child)

                node.backpropagate(self.cache)

        size, depth = root.print_solution()
        return size, depth, explored