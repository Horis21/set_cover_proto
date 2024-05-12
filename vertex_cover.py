import pandas as pd
import numpy as np
import gurobipy as grb
import queue
from gurobipy import GRB, quicksum
from classes.Node import Node
from classes.Cache import Cache
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
        return features
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
        return feature_order
    
def possible_features(df):
    features = set([i for i in range(df.shape[1]-1)])
    features_present = np.zeros(df.shape[1]-1)

    for i, row in df.iterrows():
        features_present = np.logical_or(features_present, row[1:])

    for i, f in enumerate(features_present):
        if f == 0: features.remove(i) #Remove all features that aren't present in any class

    return features

    
def one_off_features(df):
    pos = set(tuple(x) for x in df[df[0] == 1].values)
    neg = set(tuple(x) for x in df[df[0] == 0].values)

    pos_features = possible_features(df)
    ignore_feaures = set()
    features = []
    
    for i, row in df.iterrows():
        row = row.values
        for f in pos_features:
            if f in ignore_feaures: #If we already found a one-off based on this feature continue
                continue
            label = row[0]
            new_row = np.copy(row)
            new_row[f] = np.logical_xor(1, new_row[f]) #Compute the off-by-one feature vector
            new_row[0] = np.logical_xor(1, new_row[0])
            new_row = tuple(new_row)
            #Check in the opposing class
            if label == 0 and pos.__contains__(new_row):
                features.append(f)
                ignore_feaures.add(f)
            elif neg.__contains__(new_row):
                features.append(f)
                ignore_feaures.add(f)
    return features

def solve(df):
    cache = Cache()
    pq = queue.PriorityQueue()
    cache.one_off[df] = one_off_features(df)
    cache.vertex_cover[df] = vertex_cover_features(df)
    for i in range(df.shape[1]-1):
        priority = 4
        if i in cache.one_off[df]:
            priority -= 2
        if i in cache.vertex_cover[df]:
            priority -= 1 

        pq.put((priority,Node(df, i, None)))

    while pq:
        






if __name__ == "__main__":
    #df = pd.read_csv("anneal.csv", sep=" ", header=None)
    df = pd.read_csv("monk3_bin.csv", sep=" ", header=None)
    #df = pd.read_csv("test.csv", sep=" ", header=None)
    #print("vertex_cover_features: ", vertex_cover_features(df))
    #print("of-by-one feature: ", one_off_features(df))

    solve(df)
    
    
    