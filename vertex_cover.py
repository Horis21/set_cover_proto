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
import sys

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
        return best_cover, lower_bound
    

    
def one_off_features(df, cache):
    #print("Computing one off features for: ", df)
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
    #print("One-off features: ", features)
    return features

#Split data based on feature f
def split(df, f):
    return HashableDataFrame(df[df[f+1] == 0]), HashableDataFrame(df[df[f+1]==1])

#Check if the dataset is pure
def check_leaf(df):
    if df.empty or df[df[0] == 0].size == 0 or df[df[0] == 1].size == 0:
        return True
    else:
        return False

def transformTree(df, node_id, tree, parent=None, isLeft = None):
    if tree is None:
        return Node(None, None, parent, isLeft)
    # Create a new node object
    if tree.children_left[node_id] != tree.children_right[node_id]:  # It's a decision node
        feature = tree.feature[node_id]
        new_node = Node(df = df, parent_feat = None, parent=parent, isLeft = isLeft)
        new_node.f = feature
        if parent is not None:
            new_node.parent_feat = parent.f
        
        left_df, right_df = split(df, feature)
        # Recursively create left and right children
        left_child = transformTree(left_df, tree.children_left[node_id],tree ,new_node, True)
        right_child = transformTree(right_df, tree.children_right[node_id],tree, new_node, False)

        left_child.parent_feat = feature
        right_child.parent_feat = feature

        left_child.isLeft = True
        right_child.isLeft = False
        
        # Add children to the current node
        new_node.left = left_child
        new_node.right = right_child

    else:  # It's a leaf node
        new_node = Node(df = df, parent_feat = parent.f, parent= parent, isLeft = isLeft)
    
    return new_node

#Fast forward upper bound of a dataset
def fast_forward(df):
    if check_leaf(df):
        return None
    else:
        X = df[df.columns[1:]]
        cart = DecisionTreeClassifier()
        cart.fit(X, df[0])
        return cart
    
def possible_features(df, cache : Cache):
        features = cache.get_possbile_feats(df)
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
        cache.put_possible_feats(df, features)
        return features
    
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
    

def computeUB(node: Node, cache: Cache):
    data = node.df
    if cache.get_upper(data) is None:
        cart = fast_forward(data) #Compute fast forward upper bound
        ff = cart.get_n_leaves() - 1 if cart is not None else 0


        cache.put_upper(data, ff)
        best = transformTree(node.df, 0, cart.tree_, node.parent, node.isLeft)
        cache.put_best(data, best)

        if node.parent is None:
            print("bound: ", ff)
            print(export_text(cart))
            print("transformed tree for node init: ")
            best.print_solution()

        # print("best for: ", node,"dataset:", node.df, "is:")
        # best.print_solution()
        # print("from transformed from:")
        # print(export_text(cart))

        node.best = best
    node.put_node_upper(cache.get_upper(data)) #Add the dataset upper bound to the node upper bound
    node.best = cache.get_best(data)
    node.best.parent_feat = node.parent_feat

def computeLB(node: Node, cache: Cache):
    data = node.df     
    if cache.get_lower(data) is None:
        one_offs, cover_features, vclb = get_features(data, cache)
        llb = max(len(one_offs), vclb)
        # print("Putting lower bound from vc or feats")
        # print("cover features: ", cover_features)
        # print("one offs: ",len(one_offs) )
        # print("vclb: ", vclb)
        cache.put_lower(data, llb) #Add lower bound based on vertex cover and one_offs
    # print("putting a lower bound")
    node.put_node_lower(cache.get_lower(data))


def mark_leaf(node : Node, cache : Cache):
    cache.put_lower(node.df, 0)
    node.put_node_lower(0)
    node.put_node_upper(0)
    cache.put_upper(node.df, 0)
    node.mark_ready(cache)

def solve(df):
    df = HashableDataFrame(df)
    cache = Cache()
    pq = queue.PriorityQueue()

    #print("max nodes: ", max_nodes)
   
    explored = 0
    
    root = Node(df, None, None, None)
    #Add the root
    pq.put((1, root))
    computeLB(root,cache)
    computeUB(root,cache)
    list = []
    while not pq.empty():
        node = pq.get()[1]
        list.append(node)
        print("Looking at node: ", node)
        if not node.feasible: 
            print("Node is not feasible, skipping.")
            continue #Skip nodes deemed unfeasible

        data = node.df
        solution =cache.get_solution(data) #Check if solution already found
        if solution is not None:
            print("Solution already existing in cache: ", str(solution))
            node.lower = solution.lower
            node.upper = solution.upper
            node.improving = solution.improving
            node.link_and_prune(solution, cache)
            continue
       

        explored += 1 #Only updated explored nodes if not a leaf and not in cache

        
        #print("Dataset upper bound: ", cache.get_upper(data))

        
        #print("Dataset lower bound: ", cache.get_lower(data))
        
        #Get the features needed for computing priority
        one_offs, cover_features, vclb = get_features(data, cache)
        pos_features = possible_features(data, cache)
        #Search for all features
        for i in pos_features:
            #Split the data based on feature i
            left_df, right_df = split(data, i)
           
            left = Node(left_df, i, node, True)
            right = Node(right_df, i, node, False)

            left_flag = check_leaf(left_df)
            right_flag = check_leaf(right_df)

            if left_flag:
                mark_leaf(left, cache)
            else:
                computeUB(left, cache)
                computeLB(left, cache)
            if right_flag:
                mark_leaf(right, cache)
            else: 
                computeUB(right, cache)
                computeLB(right, cache)

            node.lefts[i] = left
            node.rights[i] = right

            if left.lower + right.lower + 1 > node.upper:
                left.feasible = False
                right.feasible = False
                continue

            if right_flag and left_flag:
                continue

            #Compute priority of the child nodes
            priority = 4
            if i in one_offs:
                priority -= 2  # One_off features is 100% sure to be in the optimal tree
            if i in cover_features:
                priority -= 1  # Vertex cover feature is only highly likely to be present

            if not left_flag:
                pq.put((priority, left))
            if not right_flag:
                pq.put((priority, right))
        node.backpropagate(cache)

    print("done")
    print("Only explored:", explored)
    print("final tree:")

    root.print_solution()

if __name__ == "__main__":
    #df = pd.read_csv("anneal.csv", sep=" ", header=None)
    df = pd.read_csv("monk3_bin.csv", sep=" ", header=None)
    #df = pd.read_csv("test.csv", sep=" ", header=None)
    #print("vertex_cover_features: ", vertex_cover_features(df))
    #print("of-by-one feature: ", one_off_features(df))

    solve(df)
    
    
    