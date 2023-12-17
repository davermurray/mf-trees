#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from spotpy.objectivefunctions import rmse
from spotpy.objectivefunctions import kge
from spotpy.objectivefunctions import correlationcoefficient as r
from spotpy.objectivefunctions import pbias
from spotpy.objectivefunctions import nashsutcliffe as nse
from spotpy.objectivefunctions import lognashsutcliffe as lognse

from sklearn.model_selection import train_test_split
from itertools import product

#%% Metrics Cell
def metrics(x,y): #x = obs, y = sim
    epsilon=np.mean(x)/100 + 0.1
    return [pbias(x,y),rmse(x,y),r(x,y),nse(x,y),lognse(x,y,epsilon=epsilon),kge(x,y)]

def NSE(predictions,targets):
    mse = np.mean((predictions - targets) ** 2)
    nse = 1 - (mse / np.var(targets))
    return nse

#%% Model Evaluation
def trainAndEvaluateModel(MLmodel, samples, targets, pars, test_size, n, random_state):
    #PredList = []
    metrics_ls = []
    impurity_list = []
    
    print('Shape of Input Samples: ' + str(samples.shape))
    print('Shape of Output Targets: ' + str(targets.shape))
      
    imp_df = pd.DataFrame(index = samples.columns)#.astype('int'))
    #imp_list = []
    
# =============================================================================
#     if targets.shape[0] > 1:
#         samples = samples.to_numpy()
#         targets = targets.to_numpy()
# =============================================================================
        
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(samples, targets, test_size=test_size, random_state=random_state) #changed the random state to 1 - 2/27/23
        # Create Tree Model Object
        if pars == None:
            Tree = MLmodel()
        else:
            Tree = MLmodel(**pars)
        # Train Decision Tree Classifer
        Tree = Tree.fit(X_train,y_train)

        #Predict the response for test dataset
        y_pred = Tree.predict(X_test)
       # PredList.append(y_pred)
        
        #passing multi-level dataframes into the targets(y) returns a dataframe, so need to check for it
        if isinstance(y_test, pd.DataFrame):
           y_pred = pd.DataFrame(y_pred.reshape((y_test.shape[0],y_test.shape[1])), index = y_test.index, columns = y_test.columns)
           for j in y_test.columns: 
               metrics_ls.append(metrics(y_test[j], y_pred[j]))
        else:
            metrics_ls.append(metrics(y_test, y_pred))   
        
        if n > 1:
            impurity_list.append(quickEvalTree(Tree))
        #feature importance
        
        feat_df = pd.DataFrame(Tree.feature_importances_, index = Tree.feature_names_in_, columns=[i])#.astype('int')) 
       # print(feat_df)
        #feat_df = pd.DataFrame(Tree.feature_importances_, columns=[i])#.astype('int')) 

        #imp_df = imp_df.merge(feat_df.rename(columns={0:i}), left_index=True, right_index = True)
        imp_df = imp_df.merge(feat_df, left_index=True, right_index = True)
       # imp_list.append(Tree.feature_importances_)
        
   # imp_df = pd.DataFrame(imp_list) 
    #imp_df.columns = samples.columns
    metric_cols = ['bias','rmse','r','nse','lognse','kge']
    metrics_df = pd.DataFrame(metrics_ls, columns = metric_cols)#.mean()
    
    return imp_df, metrics_df, impurity_list, Tree, y_test, y_pred

#%%
def plotImportance(imp_df,well_heads,well_loc_df, numTS, n,rn, title):
    
    dt_total_imp_df = imp_df.sum(axis=1) / n #sum up all the trees and normalize to 1 
    #Sum up all the stress periods for each well location
    dt_AllTSimp = dt_total_imp_df#.groupby(dt_total_imp_df.index // numTS).sum()
    dt_AllTSimp.index = dt_total_imp_df.index.astype('int')#.groupby(dt_total_imp_df.index // numTS).sum()

    #get indicies of every feature per TS (stressperiod)
# =============================================================================
#     TSIndices = np.arange(0,int(well_heads.columns[-1])+1,numTS)
#     
#     for i in range(numTS):
#         featInTS = np.intersect1d(dt_total_imp_df.index, TSIndices+i)
#         print(featInTS)
#         print("Number of Features used in Stress period " + str(i) + ": " + str(len(featInTS)))
#         print("Sum of Importances in Stress period " + str(i) + ": " + str(dt_total_imp_df.loc[featInTS].sum()))
# =============================================================================
        
    #feature importance Mapping
    wellmesh_dt = np.ndarray((50,50))
    wellmesh_dt[:,:] = -1e30

    
    for k in dt_AllTSimp.index:
            wellmesh_dt[int(well_loc_df.loc[k, 1]), int(well_loc_df.loc[k, 0])] = dt_AllTSimp.loc[k]
    
    #set up the meshgrid   
    kk = np.arange(0,50)
    gg = np.arange(0,50)
    GG, KK = np.meshgrid(gg,kk)
    
    cmap2 = cm.get_cmap("jet_r")#,lut=20)
    cmap2.set_under("k")
    
    vmax = np.max(dt_AllTSimp)
    #vmax = 0.1
    vmin = np.min(dt_AllTSimp)
    
    river = np.ndarray((48,))
    river[:] = 25
    
    plt.figure(figsize=(8,6))
    plt.pcolormesh(KK,GG,wellmesh_dt,vmax = vmax, vmin = vmin, cmap = cmap2, shading='nearest')
    plt.plot(range(1,49),river, color="lightblue", linestyle = 'dashed', alpha = 1, label='River')
    plt.plot(32, 19, marker="o", markersize=12, color="White", linestyle = "None", label="Pumping Well")
    plt.plot(rn, 25, marker="*", markersize=16, color="Green", linestyle = "None",label="Prediction Reach")
    # grid_z0 = griddata(wellmap[:1], wellmap[2], (KK, GG), method='nearest')
    #plt.show()
    #plt.imshow(wellmesh, cmap='RdBu')
    plt.colorbar(label = "Location Importance")
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.title(title)
    plt.legend(loc="lower right")
    
    return wellmesh_dt
    
def wellmesh_correlation(all_well_loc,well_head_df, dist):
    
    all_well_loc_df = pd.DataFrame(all_well_loc, columns = ['row','col']).reset_index()
    all_well_loc_df.set_index(['row','col'], inplace = True)
    
    #create a blank 50x50 mesh grid - not these are literals so only works for this model
    wellmesh_corr = np.ndarray((50,50))
    wellmesh_corr[:][:] = -1e30
    wellmesh_corr_min = wellmesh_corr.copy()
    
    #find matrix of neighbors by provided distance
    tup = (1,0,-1)
    
    if dist > 1:          
        for i in range(2,dist+1):
            tup = tup + (i,-i)
                
    #for i in all_well_loc.index:
    for i in range(all_well_loc.shape[0]):
        #find the location of the well and create the corr list
        center = tuple(all_well_loc[i])
        corr_matrix = []
        
        for search in product(tup,repeat = 2): #creates non repeating combos pairs which creates a grid of neighbors
            neighbor = tuple(np.add(search, (center)))
            if neighbor != center and neighbor in all_well_loc_df.index:
                corr_matrix.append(r(well_head_df[i],well_head_df[all_well_loc_df.loc[neighbor][0]]))
                
        #set the mesh location to equal the average correlation value      
        wellmesh_corr[int(all_well_loc[i, 1]),int(all_well_loc[i, 0])] = np.mean(corr_matrix)
        wellmesh_corr_min[int(all_well_loc[i, 1]),int(all_well_loc[i, 0])] = np.min(corr_matrix)

    return wellmesh_corr, wellmesh_corr_min
 #%%   
def evalTree(clf):
    #Evaluating the tree - From Sklearn
    #https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    impurity = clf.tree_.impurity
    node_samples = clf.tree_.weighted_n_node_samples
    
    split_impurity_improvement = np.zeros(shape=n_nodes, dtype=np.int64)
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
            split_impurity_improvement[i] = round(node_samples[i]/node_samples[0] * (impurity[i]  - (node_samples[children_left[i]]/node_samples[i]) * impurity[children_left[i]] - (node_samples[children_right[i]]/node_samples[i]) * impurity[children_right[i]]),2)
# =============================================================================
#             print("{space}node={node} is a split node (improvement from split: {imp}): ".format(
#                     imp=split_impurity_improvement[i],
#                     node = i,
#                     space=node_depth[i] * "\t" ))
# =============================================================================
            print("{space}node={node} is a split node (improvement from split: {imp}): "
                  "go to node {left} if X[:, {feature}] <= {threshold} "
                  "else to node {right}.".format(
                      imp=split_impurity_improvement[i],
                      space=node_depth[i] * "\t",
                      node=i,
                      left=children_left[i],
                      feature=feature[i],
                      threshold=threshold[i],
                      right=children_right[i]))
            
    return split_impurity_improvement/impurity[0]

def quickEvalTree(clf):
    #Modified from - Evaluating the tree - From Sklearn
    #https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html

    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    impurity = clf.tree_.impurity
    node_samples = clf.tree_.weighted_n_node_samples
    
    split_impurity_improvement = np.zeros(shape=n_nodes, dtype=np.int64)
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

    for i in range(n_nodes):
        if not is_leaves[i]:
            split_impurity_improvement[i] = round(node_samples[i]/node_samples[0] * (impurity[i]  - (node_samples[children_left[i]]/node_samples[i]) * impurity[children_left[i]] - (node_samples[children_right[i]]/node_samples[i]) * impurity[children_right[i]]),2)
        
    return split_impurity_improvement/impurity[0]




