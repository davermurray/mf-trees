#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import pandas as pd

from spotpy.objectivefunctions import rmse
from spotpy.objectivefunctions import kge
from spotpy.objectivefunctions import correlationcoefficient as r
from spotpy.objectivefunctions import bias
from spotpy.objectivefunctions import nashsutcliffe as nse
from sklearn.model_selection import train_test_split


def metrics(x,y): #x = obs, y = sim
    return [bias(x,y),rmse(x,y),r(x,y),nse(x,y),kge(x,y)]


def NSE(predictions,targets):
    mse = np.mean((predictions - targets) ** 2)
    nse = 1 - (mse / np.var(targets))
    return nse

def trainAndEvaluateModel(MLmodel, samples, targets, pars, test_size, n, rn):
    PredList = []
    metrics_ls = []
    print('Shape of Input Samples: ' + str(samples.shape))
    
    imp_df = pd.DataFrame(index = samples.columns.astype('int'))

    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(samples, targets[rn], test_size=test_size) #random_state=1) #changed the random state to 1 - 2/27/23
        # Create Tree Model Object
        Tree = MLmodel(**pars)
        # Train Decision Tree Classifer
        Tree = Tree.fit(X_train,y_train)

        #Predict the response for test dataset
        y_pred = Tree.predict(X_test)
        PredList.append(y_pred)
        metrics_ls.append(metrics(y_test, y_pred))   
        
        #feature importance
        feat_df = pd.DataFrame(Tree.feature_importances_, index = Tree.feature_names_in_.astype('int')) 
        imp_df = imp_df.merge(feat_df.rename(columns={0:i}), left_index=True, right_index = True)
        
    metric_cols = ['bias','rmse','r','nse','kge']
    metrics_df = pd.DataFrame(metrics_ls, columns = metric_cols).mean()
    
    return imp_df, metrics_df, y_test, y_pred


def evalTree(clf):
    #Evaluating the tree - From Sklearn
    #https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has {n} nodes and has "
          "the following tree structure:\n".format(n=n_nodes))
    for i in range(n_nodes):
        if is_leaves[i]:
            print("{space}node={node} is a leaf node.".format(
                space=node_depth[i] * "\t", node=i))
        else:
            print("{space}node={node} is a split node: "
                  "go to node {left} if X[:, {feature}] <= {threshold} "
                  "else to node {right}.".format(
                      space=node_depth[i] * "\t",
                      node=i,
                      left=children_left[i],
                      feature=feature[i],
                      threshold=threshold[i],
                      right=children_right[i]))
    return




