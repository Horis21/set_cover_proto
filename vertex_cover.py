import pandas as pd
import numpy as np
import gurobipy as grb
import queue
from gurobipy import GRB, quicksum
from sklearn.tree import DecisionTreeClassifier
from classes.Node import Node
from classes.Cache import Cache
import sys
import copy

def get_difference_table(df):
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

    diff_df = pd.concat(diffs, axis=1).T
    return diff_df

def find_vertex_cover(diff_df, verbose=True):
    model = grb.Model("set_cover")
    if not verbose:
        model.Params.LogToConsole = 0
        
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

def vertex_cover_features(df):
    if df.shape[0] <= 200:
        diff_table = get_difference_table(df)
        features = find_vertex_cover(diff_table, verbose=False)
        #print("Require features: ", features)
        return features, len(features)
    else:
        lower_bound = 0
        counts = np.zeros(df.shape[1] - 1, dtype=int)
        # repeat multiple times
        for i in range(10):
            # Get a random sample from the rows (This could be improved by using clustering)
            sample = df.sample(100)
            diff_table = get_difference_table(sample)
            features = find_vertex_cover(diff_table, verbose=False)
            # the lower bound is the maximum of all the lower bounds we have obtained
            lower_bound = max(lower_bound, len(features))
            #print(f"Iteration {i+1} resulting features: {features}")
            # Count the features used. 
            for f in features:
                counts[f] += 1
        
        # print the features in order of importance
        feature_order = np.argsort(counts)[::-1]
        for f in feature_order:
            if counts[f] == 0: break
            #print(f"Feature {f}: required {counts[f]} times")

        # print the final lower bound
        #print(f"Require at least {lower_bound} features")
        return feature_order, lower_bound
    
def possible_features(df, cache : Cache):
    features = cache.get_possbile_feats(df)
    if features is not None:
        return features
    features = set([i for i in range(df.shape[1]-1)])
    features_present = np.zeros(df.shape[1]-1)

    for i, row in df.iterrows():
        features_present = np.logical_or(features_present, row[1:])

    for i, f in enumerate(features_present):
        if f == 0: features.remove(i) #Remove all features that aren't present in any class
    cache.put_possible_feats(df, features)
    return features

    
def one_off_features(df, cache):
    print("Computing one off features for: ", df)
    pos_features = possible_features(df, cache)

    pos = set(tuple(x) for x in df[df[0] == 1].values)
    neg = set(tuple(x) for x in df[df[0] == 0].values)

    ignore_feaures = set()
    features = []
    
    for i, row in df.iterrows():
        row = row.values
        for f in pos_features:
            if f in ignore_feaures: #If we already found a one-off based on this feature continue
                continue
            label = row[0]
            new_row = np.copy(row)
            new_row[f+1] = np.logical_xor(1, new_row[f+1]) #Compute the off-by-one feature vector
            new_row[0] = np.logical_xor(1, new_row[0])
            new_row = tuple(new_row)
            #Check in the opposing class
            if label == 0 and pos.__contains__(new_row):
                features.append(f)
                ignore_feaures.add(f)
            elif neg.__contains__(new_row):
                features.append(f)
                ignore_feaures.add(f)
    print("One-off features: ", features)
    return features

#Split data based on feature f
def split(df, f):
    return df[df[f+1] == 0], df[df[f+1]==1]

#Check if the dataset is pure
def check_leaf(df):
    if df.empty or df[df[0] == 0].size == 0 or df[df[0] == 1].size == 0:
        return True
    else:
        return False


#Fast forward upper bound of a dataset
def fast_forward(df, pos_features):
    if check_leaf(df):
        return 0
    else:
        X = df[df.columns[[f + 1 for f in pos_features]]]
        cart = DecisionTreeClassifier()
        cart.fit(X, df[0])
        return cart.get_n_leaves() - 1
    
#Return one_offs, cover_features and vertex cover lower bound    
def get_features(df, cache : Cache):
    one_offs = cache.get_one_offs(df)
    if one_offs is None:
        one_offs = one_off_features(df, cache)
        cache.put_one_offs(df, one_offs)
    cover_features = cache.get_vertex_cover(df)
    vclb = 0
    if cover_features is None:
        cover_features, vclb = vertex_cover_features(df)
        cache.put_vertex_cover(df, cover_features)
    return one_offs, cover_features, vclb

def backpropagate(node : Node, cache : Cache):
    print("Backpropagating")
    data = node.df
    pos_feats = possible_features(data, cache)
    if cache.get_upper(data) != 0: #Skip leaves
        node.update_local_bounds(pos_feats) #If root still check if maybe some childrent can be pruned
    if node.parent is None:
        return
    
    parent = node.parent
    f = node.parent_feat
    
    #Add bounds for the chid at feature f
    print("Propagating bounds from child")
    parent.add_child_lower(f, node.isLeft, max(cache.get_lower(data), node.lower))
    parent.add_child_upper(f, node.isLeft, min(cache.get_upper(data), node.upper))
   
    #Update local bounds for the parent based on updates on children
    pos_feats = cache.get_possbile_feats(parent.df)
    #parent.update_local_bounds(pos_feats)
    backpropagate(parent,cache)
    
   
def print_solution(root : Node):
    q = queue.Queue()
    q.put(root)
    while not q.empty():
        node = q.get()
        print(node.f)
        if node.left is not None:
            q.put(node.left)
        if node.right is not None:
            q.put(node.right)

def solve(df):
    cache = Cache()
    pq = queue.PriorityQueue()
    pos_features = possible_features(df, cache)
    
    first = Node(df, None, None, None)
    #Add the root
    pq.put((1, first))

    while not pq.empty():
        root = pq.get()[1]
        print("Looking at node: ", root)
        if not root.feasible: 
            print("Node is not feasible, skipping.")
            continue #Skip nodes deemed unfeasible

        data = root.df
        solution =cache.get_solution(data) #Check if solution already found
        if solution is not None:
            solution = copy.deepcopy(solution)
            solution.parent_feat = root.parent_feat
            solution.parent = root.parent 
            if root.isLeft:
                root.parent.lefts[root.parent_feat] = solution
            else:
                root.parent.rights[root.parent_feat] = solution
            root.parent.update_local_bounds(cache.get_possbile_feats(root.parent.df))
            root.parent.check_ready(cache)
            continue
       

        if check_leaf(data):
            print("Leaf node found.")
            cache.put_lower(data, 0)
            root.put_node_lower(0)
            root.put_node_upper(0)
            cache.put_upper(data, 0)
            backpropagate(root, cache) #Backpropagate the bounds for the found leaf node
            root.mark_ready(cache) #Mark solution found for subproblem
            continue
        else:
            cache.put_lower(data, 1) #Lower bound of 1 if it's not a pure node

        pos_features = possible_features(data, cache)
        if cache.get_upper(data) is None:
            ff = fast_forward(data, pos_features) #Compute fast forward upper bound
            cache.put_upper(data, ff)
        root.put_node_upper(cache.get_upper(data)) #Add the dataset upper bound to the node upper bound
        print("Dataset upper bound: ", cache.get_upper(data))

        one_offs, cover_features, vclb = get_features(data, cache)
        llb = max(len(one_offs), vclb)
        print("Putting lower bound from vc or feats")
        print("one offs: ",len(one_offs) )
        print("vclb: ", vclb)
        cache.put_lower(data, llb) #Add lower bound based on vertex cover and one_offs
        root.put_node_lower(cache.get_lower(data))
        print("Dataset lower bound: ", cache.get_lower(data))

        if root.parent is not None:
            pub = root.parent.upper - 1
            #root.put_node_upper(pub) #Add the upper bound coming from the parent just for the node

        #Backpropagate the bounds
        backpropagate(root, cache)

        if not root.feasible: #Stop if bounds are not looking good
            print("Node became infeasible after backpropagation, skipping.")
            continue

        #Search for all features
        for i in pos_features:
            #Split the data based on feature i
            left_df, right_df = split(data, i)
            left = Node(left_df, i, root, True)
            right = Node(right_df, i, root, False)
            root.lefts[i] = left
            root.rights[i] = right

            #Compute priority of the child nodes
            priority = 4
            if i in one_offs:
                priority -= 2  # One_off features is 100% sure to be in the optimal tree
            if i in cover_features:
                priority -= 1  # Vertex cover feature is only highly likely to be present

            pq.put((priority, left))
            pq.put((priority, right))
    print("done")
    print_solution(first)


if __name__ == "__main__":
    #df = pd.read_csv("anneal.csv", sep=" ", header=None)
    #df = pd.read_csv("monk3_bin.csv", sep=" ", header=None)
    df = pd.read_csv("test.csv", sep=" ", header=None)
    #print("vertex_cover_features: ", vertex_cover_features(df))
    #print("of-by-one feature: ", one_off_features(df))

    solve(df)
    
    
    