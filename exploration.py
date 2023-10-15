# %%
import plotly.graph_objects as go
# import dash
import pylab as plt

import numpy as np
import pandas as pd
pd.options.plotting.backend = "plotly"

import plotly.express as px
from itertools import permutations

from hilbertcurve.hilbertcurve import HilbertCurve
import TSB
from jarowinkler import jaro_similarity
import Levenshtein
import math
import itertools
import re


#* conclusions:
# - the first city doesn't matter
# - the second city can either be x or shape-x 
#   => can only search into the first half of possibilities for city 2
# 

def generate_exploration0(my_list:list, start=1, n=int(10e3)):
    n_cities = len(my_list)
    perms = itertools.permutations(my_list[1:])
    n = int(math.factorial(n_cities-1)/2) #expected number of lists
    sigs_to_explore = []
    for sig in perms:
        if len(sigs_to_explore) >= n:
            break
        if sig <= sig[::-1]:
            sigs_to_explore.append([my_list[0]]+list(sig))
    return sigs_to_explore

def generate_exploration1000(my_list:list, start=1, n=int(10e3)):
    n = min(len(my_list)-1, n)
    start = max(1,start)
    # first_el, rest = my_list[:1], my_list[1:]
    translations = [
        my_list[:1]+my_list[i:]+my_list[1:i]
        for i in range(start, start+n)
    ]
    
    print( translations)
    swaps = []
    for trans in translations:
        for i in range(1,n):
            j=i+1
            swap = trans.copy()
            swap[i], swap[j] = swap[j], swap[i]
            swaps.append(swap)
    
    return translations + swaps

def swap_distance(s1, s2, swap_counter=0):
    if len(s1)==0:
        return 0
    s1, s1_to_s2, s2 = list(s1), list(s1), list(s2)
    j = s1.index(s2[0])
    swap = j!=0
    s1_to_s2[0],s1_to_s2[j] = s1_to_s2[j],s1_to_s2[0]
    
    # print(s1, s2, j, s1_to_s2, swap)
    return swap + swap_distance(s1_to_s2[1:], s2[1:], swap_counter)

def scores_from_sigs(sigs, ref: pd.DataFrame):
    return [TSB.dtot(ref.loc[list(sig)]) for sig in sigs]

def is_same(sig1, sig2):
    assert sig1[0] == sig2[0]
    return sig1[1:] == sig2[1:] or sig1[1:] == sig2[1:][::-1]

def matrix_apply_to_df(f, arrays_1, arrays_2=None):
    arrays_2 = arrays_1 if arrays_2 is None else arrays_2
    matrix= [
        [
            f(arr1, arr2) 
            for arr1 in arrays_1
        ]
        for arr2 in arrays_2
    ]
    
    return pd.DataFrame(
    matrix,

    columns=[''.join(map(str, l)) for l in arrays_1],
    index=[''.join(map(str, l)) for l in arrays_2]
)
    
def get_best_info(scores, sigs):
    
    best_score = np.min(scores)
    best_path = np.array(sigs)[np.isclose(np.min(scores) - scores, 0, 10e-5)]
    # print(f'Best score: {best_score},\n paths: \n {best_path}')
    return best_score, best_path

def settup_exploration(exploration_generation_function, n_cities):
    ref_sig = list(range(n_cities))
    all_sigs = [ref_sig[:1]+list(sig) for sig in itertools.permutations(ref_sig[1:])]
    exp_sigs = exploration_generation_function(ref_sig)
    return all_sigs, exp_sigs
#%% ----------------

# %%
def generate_refs(sig):
    refs = []
    sig = np.array(sig)
    indices = np.arange(sig.shape[0])
    for i in range(1,4):
        is_multiple_of_i = indices%i==0
        refs.append(sig[is_multiple_of_i].tolist()+sig[~is_multiple_of_i].tolist())
    return refs
        
n_cities = 6
LV = TSB.crea(n_cities)
exp_sigs = generate_exploration0(range(n_cities))
refs = [LV.index.tolist()]
exp_scores = scores_from_sigs(exp_sigs, LV)

df = matrix_apply_to_df(swap_distance, refs, exp_sigs)
df['swap_dist_from_ref'] = df.min(axis=1)
px.imshow(matrix_apply_to_df(swap_distance, exp_sigs, exp_sigs)).show()
df.iloc[(df['swap_dist_from_ref']==df['swap_dist_from_ref'].max()).values,:]
# f=is_same
# dissimatrix = matrix_apply_to_df(f, all_sigs[:n])
# px.imshow(dissimatrix, title=f.__name__).show()


# %%
# score_matrix = np.broadcast_to(exp_scores, shape=[len(exp_scores)]*2)
# px.imshow(
#     pd.DataFrame(
#         score_matrix-score_matrix.T,
#         index=[''.join(map(str, l)) for l in sigs_to_explore],
#         columns= [''.join(map(str, l)) for l in sigs_to_explore]
#     ),
#     title=f'Scores, here:{min([TSB.dtot(LV.loc[sig_bis]) for sig_bis in sigs_to_explore])} all:{min(all_scores)}'
# ).show()

# check solution in explore
# print(get_best_info(all_scores, all_sigs))
# print(get_best_info(sigs_score, sigs_to_explore))

# np.unique(sigs_score, return_counts=1)
# %%
# [TSB.plot_path(LV.loc[sig]).show() for sig in  np.array(all_sigs)[np.min(all_scores) == all_scores]]
# %%
# ? goal:
# find a way to generate the minimum amount of path to covert all
# relevent paths and test whether there is no discrepency several times

def test(exploration_generation_function, number_of_tests=100, n_cities = 5):
    all_sigs, exp_sigs = settup_exploration(exploration_generation_function, n_cities)

    redundant = np.any(matrix_apply_to_df(is_same, exp_sigs).sum()-1)
    expexted_optimal = int(math.factorial(n_cities-1)/2)
    
    print(f'Redundant {redundant}, {len(exp_sigs)=}, {expexted_optimal=}')
    
    did_not_cover_mem = []
    for i in range(number_of_tests):
        LV = TSB.crea(n_cities)
        
        all_scores = scores_from_sigs(all_sigs, LV)
        exp_scores = scores_from_sigs(exp_sigs, LV)
        
        all_best_score, all_best_paths = get_best_info(all_scores, all_sigs)
        exp_best_score, exp_best_paths = get_best_info(exp_scores, exp_sigs)
        
        best_score_in_exploration_space = np.isclose(all_best_score,exp_best_score,10e-6)
        if not best_score_in_exploration_space:
            did_not_cover_mem.append((LV, best_score_in_exploration_space))
            print("-----")
        if not i%10:
            print(i)

    return did_not_cover_mem
  
t = test(generate_exploration0, 100, 7)
# %%
