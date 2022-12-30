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
def FE_remove_variables_using_SULOV_method(df, numvars, modeltype, target,
                                corr_limit = 0.70,verbose=0, dask_xgboost_flag=False):
    """
    FE stands for Feature Engineering - it means this function performs feature engineering
    ###########################################################################################
    #####              SULOV stands for Searching Uncorrelated List Of Variables  #############
    This highly efficient method removes variables that are highly correlated using a series of
    pair-wise correlation knockout rounds. It is extremely fast and hence can work on thousands
    of variables in less than a minute, even on a laptop. You need to send in a list of numeric
    variables and that's all! The method defines high Correlation as anything over 0.70 (absolute)
    but this can be changed. If two variables have absolute correlation higher than this, they
    will be marked, and using a process of elimination, one of them will get knocked out:
    To decide order of variables to keep, we use mutuail information score to select. MIS returns
    a ranked list of these correlated variables: when we select one, we knock out others
    that it is correlated to. Then we select next var. This way we knock out correlated variables.
    Finally we are left with uncorrelated variables that are also highly important in mutual score.
    ########  YOU MUST INCLUDE THE ABOVE MESSAGE IF YOU COPY THIS CODE IN YOUR LIBRARY ##########
    """
    df = copy.deepcopy(df)
    df_target = df[target]
    df = df[numvars]
    ### for some reason, doing a mass fillna of vars doesn't work! Hence doing it individually!
    null_vars = np.array(numvars)[df.isnull().sum()>0]
    for each_num in null_vars:
        df[each_num] = df[each_num].fillna(0)
    target = copy.deepcopy(target)

    print('#######################################################################################')
    print('#####  Searching for Uncorrelated List Of Variables (SULOV) in %s features ############' %len(numvars))
    print('#######################################################################################')
    ### This is a shorter version of getting unduplicated and highly correlated vars ##
    #correlation_dataframe = df.corr().abs().unstack().sort_values().drop_duplicates()
    ### This change was suggested by such3r on GitHub issues. Added Dec 30, 2022 ###
    correlation_dataframe = df.corr().abs().unstack().sort_values().round(7).drop_duplicates()
    corrdf = pd.DataFrame(correlation_dataframe[:].reset_index())
    corrdf.columns = ['var1','var2','coeff']
    corrdf1 = corrdf[corrdf['coeff']>=corr_limit]
    ### Make sure that the same var is not correlated to itself! ###
    corrdf1 = corrdf1[corrdf1['var1'] != corrdf1['var2']]
    correlated_pair = list(zip(corrdf1['var1'].values.tolist(),corrdf1['var2'].values.tolist()))
    corr_pair_dict = dict(return_dictionary_list(correlated_pair))
    corr_list = find_remove_duplicates(corrdf1['var1'].values.tolist()+corrdf1['var2'].values.tolist())
    keys_in_dict = list(corr_pair_dict.keys())
    reverse_correlated_pair = [(y,x) for (x,y) in correlated_pair]
    reverse_corr_pair_dict = dict(return_dictionary_list(reverse_correlated_pair))
    #### corr_pair_dict is used later to make the network diagram to see which vars are correlated to which
    for key, val in reverse_corr_pair_dict.items():
        if key in keys_in_dict:
            if len(key) > 1:
                corr_pair_dict[key] += val
        else:
            corr_pair_dict[key] = val
    
    ###### This is for ordering the variables in the highest to lowest importance to target ###
    if len(corr_list) == 0:
        final_list = list(correlation_dataframe)
        print('Selecting all (%d) variables since none of numeric vars are highly correlated...' %len(numvars))
        return numvars
    else:
        if isinstance(target, list):
            target = target[0]
        max_feats = len(corr_list)
        if modeltype == 'Regression':
            sel_function = mutual_info_regression
            #fs = SelectKBest(score_func=sel_function, k=max_feats)
        else:
            sel_function = mutual_info_classif
            #fs = SelectKBest(score_func=sel_function, k=max_feats)
        ##### you must ensure there are no infinite nor null values in corr_list df ##
        df_fit = df[corr_list]
        ### Now check if there are any NaN values in the dataset #####
        
        if df_fit.isnull().sum().sum() > 0:
            df_fit = df_fit.dropna()
        else:
            print('    there are no null values in dataset...')
        ##### Reduce memory usage and find mutual information score ####       
        #try:
        #    df_fit = reduce_mem_usage(df_fit)
        #except:
        #    print('Reduce memory erroring. Continuing...')
        ##### Ready to perform fit and find mutual information score ####
        
        try:
            #fs.fit(df_fit, df_target)
            if modeltype == 'Regression':
                fs = mutual_info_regression(df_fit, df_target, n_neighbors=5, discrete_features=False, random_state=42)
            else:
                fs = mutual_info_classif(df_fit, df_target, n_neighbors=5, discrete_features=False, random_state=42)
        except:
            print('    SelectKBest() function is erroring. Returning with all %s variables...' %len(numvars))
            return numvars
        try:
            #################################################################################
            #######   This is the main section where we use mutual info score to select vars        
            #################################################################################
            #mutual_info = dict(zip(corr_list,fs.scores_))
            mutual_info = dict(zip(corr_list,fs))
            #### The first variable in list has the highest correlation to the target variable ###
            sorted_by_mutual_info =[key for (key,val) in sorted(mutual_info.items(), key=lambda kv: kv[1],reverse=True)]
            #####   Now we select the final list of correlated variables ###########
            selected_corr_list = []
            #### You have to make multiple copies of this sorted list since it is iterated many times ####
            orig_sorted = copy.deepcopy(sorted_by_mutual_info)
            copy_sorted = copy.deepcopy(sorted_by_mutual_info)
            copy_pair = copy.deepcopy(corr_pair_dict)
            #### select each variable by the highest mutual info and see what vars are correlated to it
            for each_corr_name in copy_sorted:
                ### add the selected var to the selected_corr_list
                selected_corr_list.append(each_corr_name)
                for each_remove in copy_pair[each_corr_name]:
                    #### Now remove each variable that is highly correlated to the selected variable
                    if each_remove in copy_sorted:
                        copy_sorted.remove(each_remove)
            ##### Now we combine the uncorrelated list to the selected correlated list above
            rem_col_list = left_subtract(numvars,corr_list)
            final_list = rem_col_list + selected_corr_list
            removed_cols = left_subtract(numvars, final_list)
        except Exception as e:
            print('    SULOV Method crashing due to %s' %e)
            #### Dropping highly correlated Features fast using simple linear correlation ###
            removed_cols = remove_highly_correlated_vars_fast(df,corr_limit)
            final_list = left_subtract(numvars, removed_cols)
        if len(removed_cols) > 0:
            print('    Removing (%d) highly correlated variables:' %(len(removed_cols)))
            if len(removed_cols) <= 30:
                print('    %s' %removed_cols)
            if len(final_list) <= 30:
                print('    Following (%d) vars selected: %s' %(len(final_list),final_list))
        ##############    D R A W   C O R R E L A T I O N   N E T W O R K ##################
        selected = copy.deepcopy(final_list)
        if verbose:
            try:
                import networkx as nx
                #### Now start building the graph ###################
                gf = nx.Graph()
                ### the mutual info score gives the size of the bubble ###
                multiplier = 2100
                for each in orig_sorted:
                    gf.add_node(each, size=int(max(1,mutual_info[each]*multiplier)))
                ######### This is where you calculate the size of each node to draw
                sizes = [mutual_info[x]*multiplier for x in list(gf.nodes())]
                ####  The sizes of the bubbles for each node is determined by its mutual information score value
                corr = df_fit.corr()
                high_corr = corr[abs(corr)>corr_limit]
                ## high_corr is the dataframe of a few variables that are highly correlated to each other
                combos = combinations(corr_list,2)
                ### this gives the strength of correlation between 2 nodes ##
                multiplier = 20
                for (var1, var2) in combos:
                    if np.isnan(high_corr.loc[var1,var2]):
                        pass
                    else:
                        gf.add_edge(var1, var2,weight=multiplier*high_corr.loc[var1,var2])
                ######## Now start building the networkx graph ##########################
                widths = nx.get_edge_attributes(gf, 'weight')
                nodelist = gf.nodes()
                cols = 5
                height_size = 5
                width_size = 15
                rows = int(len(corr_list)/cols)
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
                if len(selected) == 0:
                    nx.draw_networkx_labels(gf, pos=pos_higher,
                                        labels=dict(zip(nodelist,nodelist)),
                                        font_color='black')
                else:
                    nx.draw_networkx_labels(gf, pos=pos_higher,
                                        labels = dict(zip(nodelist,[x+' (selected)' if x in selected else x+' (removed)' for x in nodelist])),
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
                print('    Networkx library visualization crashing due to %s' %e)
                print('Completed SULOV. %d features selected' %len(final_list))
        else:
            print('Completed SULOV. %d features selected' %len(final_list))
        return final_list
###################################################################################
