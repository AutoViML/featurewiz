import numpy as np
import pandas as pd
import random
np.random.seed(99)
random.seed(42)
from . import settings
settings.init()
################################################################################
#### The warnings from Sklearn are so annoying that I have to shut it off #######
import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
def warn(*args, **kwargs):
    pass
warnings.warn = warn
import logging
####################################################################################
import pdb
import copy
import time
from sklearn.feature_selection import chi2, mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import SelectKBest
from itertools import combinations
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
#################################################################################################
from collections import defaultdict
from collections import OrderedDict
import time
import networkx as nx # Import networkx for groupwise method
#################################################################################
def left_subtract(l1,l2):
    lst = []
    for i in l1:
        if i not in l2:
            lst.append(i)
    return lst
#################################################################################
def return_dictionary_list(lst_of_tuples):
    """ Returns a dictionary of lists if you send in a list of Tuples"""
    orDict = defaultdict(list)
    # iterating over list of tuples
    for key, val in lst_of_tuples:
        orDict[key].append(val)
    return orDict
################################################################################
def find_remove_duplicates(list_of_values):
    """
    # Removes duplicates from a list to return unique values - USED ONLY ONCE
    """
    output = []
    seen = set()
    for value in list_of_values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output
##################################################################################
def remove_highly_correlated_vars_fast(df, corr_limit): # Keeping original function for fallback
    """Fast method to remove highly correlated vars using just linear correlation."""
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > corr_limit)]
    return to_drop

def FE_remove_variables_using_SULOV_method(df, numvars, modeltype, target,
                                corr_limit = 0.70, verbose=0, dask_xgboost_flag=False,
                                correlation_types = ['pearson'], # New parameter for correlation types
                                adaptive_threshold = False,      # New parameter for adaptive threshold
                                sulov_mode = 'pairwise'):         # New parameter for SULOV mode (pairwise/groupwise)
    """
    FE stands for Feature Engineering - it means this function performs feature engineering
    ###########################################################################################
    #####              SULOV stands for Searching Uncorrelated List Of Variables  #############
    ###########################################################################################
    SULOV method was created by Ram Seshadri in 2018. This highly efficient method removes
        variables that are highly correlated using a series of pair-wise correlation knockout
        rounds. It is extremely fast and hence can work on thousands of variables in less than
        a minute, even on a laptop. You need to send in a list of numeric variables and that's
        all! The method defines high Correlation as anything over 0.70 (absolute) but this can
        be changed. If two variables have absolute correlation higher than this, they will be
        marked, and using a process of elimination, one of them will get knocked out:
    To decide order of variables to keep, we use mutuail information score to select. MIS returns
        a ranked list of these correlated variables: when we select one, we knock out others that
        are highly correlated to it. Then we select next variable to inspect. This continues until
        we knock out all highly correlated variables in each set of variables. Finally we are
        left with only uncorrelated variables that are also highly important in mutual score.
    ###########################################################################################
    ########  YOU MUST INCLUDE THE ABOVE MESSAGE IF YOU COPY SULOV method IN YOUR LIBRARY ##########
    ###########################################################################################
    """
    df = copy.deepcopy(df)
    df_target = df[target]
    df = df[numvars]
    ### for some reason, doing a mass fillna of vars doesn't work! Hence doing it individually!
    null_vars = np.array(numvars)[df.isnull().sum()>0]
    for each_num in null_vars:
        df[each_num] = df[each_num].fillna(0)
    target = copy.deepcopy(target)
    if verbose:
        print('#######################################################################################')
        print('#####  Searching for Uncorrelated List Of Variables (SULOV) in %s features ############' %len(numvars))
        print('#######################################################################################')
    print('Starting SULOV with %d features...' %len(numvars))
    
    # 1. Calculate Correlation Matrices based on correlation_types parameter
    correlation_matrices = {}
    for corr_type in correlation_types:
        correlation_matrices[corr_type] = df.corr(method=corr_type).abs()

    # 2. Adaptive Threshold (if enabled)
    current_corr_threshold = corr_limit
    if adaptive_threshold:
        combined_corr_matrix = pd.concat(correlation_matrices.values()).max(level=0) # Max across all corr types
        upper_triangle_corrs = combined_corr_matrix.where(np.triu(np.ones(combined_corr_matrix.shape),k=1).astype(bool)).stack().sort_values(ascending=False)
        correlation_values = upper_triangle_corrs.values
        current_corr_threshold = np.percentile(correlation_values, 75) # Example: 75th percentile
        print(f"Adaptive Correlation Threshold: {current_corr_threshold:.3f}")

    # 3. Find Correlated Pairs based on all selected correlation types
    correlated_pairs = []
    for i in range(len(df.columns)):
        for j in range(i + 1, len(df.columns)):
            col1 = df.columns[i]
            col2 = df.columns[j]
            is_correlated = False
            for corr_type, corr_matrix in correlation_matrices.items():
                if corr_matrix.loc[col1, col2] >= current_corr_threshold:
                    is_correlated = True
                    break # If correlated by any type, consider them correlated
            if is_correlated:
                correlated_pairs.append((col1, col2))

    # Deterministic sorting of correlated pairs (always applied)
    correlated_pairs.sort()

    if modeltype == 'Regression':
        sel_function = mutual_info_regression
    else:
        sel_function = mutual_info_classif

    if correlated_pairs: # Proceed only if correlated pairs are found
        if isinstance(target, list):
            target = target[0]
        max_feats = len(numvars) # Changed from len(corr_list) to numvars to be more robust

        ##### you must ensure there are no infinite nor null values in corr_list df ##
        df_fit_cols = find_remove_duplicates(sum(correlated_pairs,())) # Unique cols from correlated pairs
        df_fit = df[df_fit_cols]

        ### Now check if there are any NaN values in the dataset #####
        if df_fit.isnull().sum().sum() > 0:
            df_fit = df_fit.dropna()
        else:
            print('    there are no null values in dataset...')

        if df_target.isnull().sum().sum() > 0:
            print('    there are null values in target. Returning with all vars...')
            return numvars
        else:
            print('    there are no null values in target column...')

        ##### Ready to perform fit and find mutual information score ####
        
        try:
            if modeltype == 'Regression':
                fs = mutual_info_regression(df_fit, df_target, n_neighbors=5, discrete_features=False, random_state=42)
            else:
                fs = mutual_info_classif(df_fit, df_target, n_neighbors=5, discrete_features=False, random_state=42)
        except Exception as e:
            print(f'    SelectKBest() function is erroring with: {e}. Returning with all {len(numvars)} variables...')
            return numvars
        
        try:
            #################################################################################
            #######   This is the main section where we use mutual info score to select vars
            #################################################################################
            mutual_info = dict(zip(df_fit_cols,fs)) # Use df_fit_cols as keys
            #### The first variable in list has the highest correlation to the target variable ###
            sorted_by_mutual_info =[key for (key,val) in sorted(mutual_info.items(), key=lambda kv: kv[1],reverse=True)]

            if sulov_mode == 'pairwise':
                #####   Now we select the final list of correlated variables (Pairwise SULOV) ###########
                selected_corr_list = []
                copy_sorted = copy.deepcopy(sorted_by_mutual_info)
                analyzed_pairs = set() # Track analyzed pairs to avoid redundancy

                for col1_sorted in copy_sorted:
                    selected_corr_list.append(col1_sorted)
                    for col2_tup in correlated_pairs:
                        col1_corr, col2_corr = col2_tup
                        pair = tuple(sorted(col2_tup)) # Ensure consistent pair order
                        if col1_sorted == col1_corr and pair not in analyzed_pairs: # Check if current sorted col is part of a correlated pair
                            analyzed_pairs.add(pair)
                            if col2_corr in copy_sorted:
                                copy_sorted.remove(col2_corr)
                        elif col1_sorted == col2_corr and pair not in analyzed_pairs: # Check if current sorted col is part of a correlated pair
                            analyzed_pairs.add(pair)
                            if col1_corr in copy_sorted:
                                copy_sorted.remove(col1_corr)

            elif sulov_mode == 'groupwise':
                #####   Groupwise SULOV ###########
                G = nx.Graph()
                for col in df_fit_cols: # Use df_fit_cols for graph nodes
                    G.add_node(col)
                for col1_g, col2_g in correlated_pairs:
                    G.add_edge(col1_g, col2_g)
                correlated_feature_groups = list(nx.connected_components(G))

                selected_corr_list = []
                features_to_drop_in_group = set()
                for group in correlated_feature_groups:
                    if len(group) > 1:
                        group_mis_scores = {feature: mutual_info.get(feature, 0) for feature in group} # Get MIS for group, default 0 if not found
                        best_feature = max(group_mis_scores, key=group_mis_scores.get) # Feature with max MIS
                        selected_corr_list.append(best_feature) # Keep best feature
                        for feature in group:
                            if feature != best_feature:
                                features_to_drop_in_group.add(feature)
                selected_corr_list = list(set(selected_corr_list)) # Ensure unique selected features
                removed_cols_sulov = list(features_to_drop_in_group) # Features removed in groupwise mode
                final_list_corr_part = selected_corr_list # Renamed for clarity

            else: # Default to original pairwise logic if mode is not recognized
                print(f"Warning: Unknown SULOV mode '{sulov_mode}'. Defaulting to pairwise mode.")
                ##### Original Pairwise SULOV logic (as fallback) ######
                selected_corr_list = []
                copy_sorted = copy.deepcopy(sorted_by_mutual_info)
                copy_pair = dict(return_dictionary_list(correlated_pairs)) # Recreate pair dict if needed
                for each_corr_name in copy_sorted:
                    selected_corr_list.append(each_corr_name)
                    if each_corr_name in copy_pair: # Check if key exists before accessing
                        for each_remove in copy_pair[each_corr_name]:
                            if each_remove in copy_sorted:
                                copy_sorted.remove(each_remove)
                final_list_corr_part = selected_corr_list # Renamed for clarity


            if sulov_mode != 'groupwise': # For pairwise and default modes
                final_list_corr_part = selected_corr_list # Renamed for consistency
                removed_cols_sulov = left_subtract(df_fit_cols, final_list_corr_part) # Calculate removed cols

            ##### Now we combine the uncorrelated list to the selected correlated list above
            rem_col_list = left_subtract(numvars, df_fit_cols) # Uncorrelated columns are those not in df_fit_cols
            final_list = rem_col_list + final_list_corr_part
            removed_cols = left_subtract(numvars, final_list) + removed_cols_sulov # Combine all removed cols

        except Exception as e:
            print(f'    SULOV Method crashing due to: {e}')
            #### Dropping highly correlated Features fast using simple linear correlation ###
            removed_cols = remove_highly_correlated_vars_fast(df,corr_limit)
            final_list = left_subtract(numvars, removed_cols)

        if len(removed_cols) > 0:
            if verbose:
                print(f'    Removing ({len(removed_cols)}) highly correlated variables:')
                if len(removed_cols) <= 30:
                    print(f'    {removed_cols}')
                if len(final_list) <= 30:
                    print(f'    Following ({len(final_list)}) vars selected: {final_list}')

        ##############    D R A W   C O R R E L A T I O N   N E T W O R K ##################
        selected = copy.deepcopy(final_list)
        if verbose and len(selected) <= 1000 and correlated_pairs: # Draw only if correlated pairs exist
            try:
                #### Now start building the graph ###################
                gf = nx.Graph()
                ### the mutual info score gives the size of the bubble ###
                multiplier = 2100
                for each in sorted_by_mutual_info: # Use sorted_by_mutual_info for node order
                    if each in mutual_info: # Check if mutual_info exists for the node
                         gf.add_node(each, size=int(max(1,mutual_info[each]*multiplier)))

                ######### This is where you calculate the size of each node to draw
                sizes = [mutual_info.get(x,0)*multiplier for x in list(gf.nodes())] # Use .get with default 0 for robustness

                corr = df[df_fit_cols].corr() # Use df_fit_cols for correlation calculation
                high_corr = corr[abs(corr)>current_corr_threshold] # Use adaptive/original threshold
                combos = combinations(df_fit_cols,2) # Use df_fit_cols for combinations

                ### this gives the strength of correlation between 2 nodes ##
                multiplier_edge = 20 # Renamed to avoid confusion
                for (var1, var2) in combos:
                    if np.isnan(high_corr.loc[var1,var2]):
                        pass
                    else:
                        gf.add_edge(var1, var2,weight=multiplier_edge*high_corr.loc[var1,var2])

                ######## Now start building the networkx graph ##########################
                widths = nx.get_edge_attributes(gf, 'weight')
                nodelist = gf.nodes()
                cols = 5
                height_size = 5
                width_size = 15
                rows = int(len(df_fit_cols)/cols) # Use df_fit_cols length
                if rows < 1:
                    rows = 1
                plt.figure(figsize=(width_size,min(20,height_size*rows)))
                pos = nx.shell_layout(gf)
                nx.draw_networkx_nodes(gf,pos,
                                       nodelist=nodelist,
                                       node_size=sizes,
                                       node_color='blue',
                                       alpha=0.5)
                nx.draw_networkx_edges(gf,pos,
                                       edgelist = widths.keys(),
                                       width=list(widths.values()),
                                       edge_color='lightblue',
                                       alpha=0.6)
                pos_higher = {}
                x_off = 0.04  # offset on the x axis
                y_off = 0.04  # offset on the y axis
                for k, v in pos.items():
                    pos_higher[k] = (v[0]+x_off, v[1]+y_off)

                labels_dict = {} # Create labels dictionary
                for x in nodelist:
                    if x in selected:
                        labels_dict[x] = x+' (selected)'
                    else:
                        labels_dict[x] = x+' (removed)'

                nx.draw_networkx_labels(gf, pos=pos_higher,
                                    labels = labels_dict,
                                    font_color='black')

                plt.box(True)
                plt.title("""In SULOV, we repeatedly remove features with lower mutual info scores among highly correlated pairs (see figure),
                            SULOV selects the feature with higher mutual info score related to target when choosing between a pair. """, fontsize=10)
                plt.suptitle('How SULOV Method Works by Removing Highly Correlated Features', fontsize=20,y=1.03)
                red_patch = mpatches.Patch(color='blue', label='Bigger circle denotes higher mutual info score with target')
                blue_patch = mpatches.Patch(color='lightblue', label='Thicker line denotes higher correlation between two variables')
                plt.legend(handles=[red_patch, blue_patch],loc='best')
                plt.show();
                #####    N E T W O R K     D I A G R A M    C O M P L E T E   #################
                return final_list
            except Exception as e:
                print(f'    Networkx library visualization crashing due to {e}')
                print(f'Completed SULOV. {len(final_list)} features selected')
                return final_list
        else:
            print(f'Completed SULOV. {len(final_list)} features selected')
        return final_list
    print(f'Completed SULOV. All {len(numvars)} features selected')
    return numvars
###################################################################################