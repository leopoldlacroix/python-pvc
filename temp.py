# %% import random
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


class Neighbor_proposition:
    def _choose_sorted(array):
        return np.sort(np.random.choice(array, size=2, replace=False))
        
    def inverse_between_two_nodes(LV:pd.DataFrame) -> pd.DataFrame:
        new_index = list(range(LV.shape[0]))
        node_i, node_j = Neighbor_proposition._choose_sorted(new_index)
        start, inverted, end = new_index[:node_i+1], new_index[node_j:node_i:-1], new_index[node_j+1:]
        new_index = start+inverted+end
        # print(node_i, node_j, start, inverted, end, new_index, indices)
        return LV.iloc[new_index]
    
    def swap_two_cities(LV:pd.DataFrame, restrain):
        new_index = LV.index.tolist()
        
        node_i, node_j = Neighbor_proposition._choose_sorted(new_index)
        new_index[node_i], new_index[node_j]=new_index[node_j], new_index[node_i]
        
        while restrain and new_index[1:] >= new_index[1:][::-1]:
            node_i, node_j = Neighbor_proposition._choose_sorted(new_index)
            new_index[node_i], new_index[node_j]=new_index[node_j], new_index[node_i]
        
        return LV.loc[new_index]
    
    def insert(LV:pd.DataFrame):
        new_index:list = LV.index.tolist()
        i, to = np.random.choice(new_index), np.random.randint(0, len(new_index)-1)
        node_i = new_index.pop(i)
        new_index.insert(to, node_i)
        return LV.loc[new_index]
        
    def insert_subroute(LV:pd.DataFrame):
        new_index = LV.index.tolist()
        node_i, node_j = Neighbor_proposition._choose_sorted(new_index)
        is_subroute = np.logical_and(new_index>=node_i, new_index<=node_j)
        new_i = np.random.choice(range(np.sum(~is_subroute)+1))
        new_index = np.insert(new_index[~is_subroute],new_i, new_index[is_subroute])
        print(node_i, node_j, is_subroute, new_i, new_index)
        return LV.loc[new_index]
    
    def complete_random(LV:pd.DataFrame, restrain):
        "Shuffles cities (except the first one)"
        random_i_s = np.random.choice(LV.index[1:], LV.shape[0]-1, replace=False)
        new_index = [LV.index[0]]+random_i_s.tolist()
        if restrain and new_index[1:] >= new_index[1:][::-1]:
            random_i_s = np.random.choice(LV.index[1:], LV.shape[0]-1, replace=False)
            new_index = [LV.index[0]]+random_i_s.tolist()
        return LV.loc[new_index,:]


#_______________________________methode 7: Annealing algorithm ________________________________________________

class Annealing:
    def __init__(self, ref_temp = 1, same_score_max_count = 500, restrain=False) -> None:
        self.ref_temp=ref_temp
        self.restrain=restrain
        self.same_score_max_count = same_score_max_count
    # -- heuristic/performance
    # quicly normalizing the score
    def simulate_estimators(self, LV:pd.DataFrame):
        sample_ds = []
        for _ in range(100):
            sample_ds.append(TSB.dtot(Neighbor_proposition.complete_random(LV, self.restrain)))
        self.mean_est, self.std_est = np.mean(sample_ds), np.std(sample_ds)
        
    
    def score_qnormalized(self, LV: pd.DataFrame):
        """
        The ``lower`` the score the ``better``
        Quicly normalized at random when finding a good solution should be below zero. 
        """

        return (TSB.dtot(LV)-self.mean_est)/self.std_est
    
    def acceptance_probability(diff, temp, temp_ref):
        """The lower the score diff the greeater the result"""
        return np.exp(-diff-1/ (temp_ref/temp))
    
    def start(self, LV:pd.DataFrame):
        self.simulate_estimators(LV) #used to keep furthest solutions find a way to precompute them and check if no to ressource extensive

        solution_scores_neighbor_scores_acceptance_probabilities  = []
        global_solution, current_solution = LV.copy(), LV.copy()
        global_score, current_score = self.score_qnormalized(global_solution), self.score_qnormalized(current_solution)
        same_score_diff,current_temp = 0, 1
        
        while same_score_diff < self.same_score_max_count:
            neighbor_solution = Neighbor_proposition.swap_two_cities(current_solution, self.restrain)
            
            # the lower the better the neighbor
            neighbor_score = self.score_qnormalized(neighbor_solution)
            acceptance_prob = Annealing.acceptance_probability(current_score-neighbor_score, current_temp, self.ref_temp)

            if neighbor_score < current_score:
                current_solution = neighbor_solution
                current_score = self.score_qnormalized(current_solution)
                same_score_diff = 0
                
                if neighbor_score < global_score:
                    global_solution = neighbor_solution
                    global_score = self.score_qnormalized(global_solution)
            
            # if the new solution is not better, accept it with a probability of e^(-score/temp)
            else:
                if np.random.uniform(0, 1) <= acceptance_prob:
                    current_solution = neighbor_solution
                    current_score = self.score_qnormalized(current_solution)
                    same_score_diff = 0
                else:
                    same_score_diff+=1
                    
            current_temp+=1
            if current_temp%1000==0:
                print(current_temp, global_score, current_score, acceptance_prob, TSB.dtot(global_solution))
            solution_scores_neighbor_scores_acceptance_probabilities.append([self.score_qnormalized(current_solution),self.score_qnormalized(neighbor_solution), acceptance_prob])
        
        return global_solution, pd.DataFrame(solution_scores_neighbor_scores_acceptance_probabilities, columns=["solution_scores","neighbor_scores","acceptance_probabilities"])

def swap_distance(s1, s2, swap_counter=0):
    if len(s1)==0:
        return 0
    s1, s1_to_s2, s2 = list(s1), list(s1), list(s2)
    j = s1.index(s2[0])
    swap = j!=0
    s1_to_s2[0],s1_to_s2[j] = s1_to_s2[j],s1_to_s2[0]
    
    # print(s1, s2, j, s1_to_s2, swap)
    return swap + swap_distance(s1_to_s2[1:], s2[1:], swap_counter)

def matrix_apply_to_df(f, arrays_1, arrays_2=None):
    arrays_2 = arrays_1 if arrays_2 is None else arrays_2
    matrix= [
        [
            f(arr1, arr2) 
            for arr1 in arrays_1
        ]
        for arr2 in arrays_2
    ]
    
    return matrix
# %%
temp, max_count=0.2,200
LV = TSB.crea(20)
ref_sigs = [LV.index.tolist()]
candidate_sigs = [LV.index.tolist()] + [Neighbor_proposition.complete_random(LV, restrain=True).index.tolist() for i in range(100-1)]
highest_dist = np.inf
while len(ref_sigs)<=8 and highest_dist >=3:
    matrix_distances_to_refs = matrix_apply_to_df(swap_distance, ref_sigs, candidate_sigs)
    min_distance_to_refs = np.min(matrix_distances_to_refs, axis=1)
    print(max(min_distance_to_refs))
    ref_sigs.append(candidate_sigs[np.argmax(min_distance_to_refs)])

scores_1 = []
for ref_sig in ref_sigs:
    ref = LV.loc[ref_sig]
    solution, record = Annealing(temp,max_count, True).start(ref)
    scores_1.append(TSB.dtot(solution))
    # TSB.plot_path(solution).show()

scores_2 = []
for i in range(8):
    solution, record = Annealing(temp,max_count, False).start(ref)
    scores_2.append(TSB.dtot(solution))
    # TSB.plot_path(solution, strategy_name='unrestrained and not prestarted').show()

# %%
record['score_diffs'] = record['solution_scores'] - record['neighbor_scores']
record.plot(y=['score_diffs', "acceptance_probabilities"])
# %%
# %%
#_____________________________methode0: le meilleur!________________________________________________

    
# def m0_all(LV:pd.DataFrame):
#     "tests all possibilities"
#     indexes_perms = pd.Series(permutations(LV.index))
#     scores = indexes_perms.apply(lambda idx: dtot(LV.loc[idx,:]))
#     i_best_perm = np.argmin(scores)
#     return LV.loc[indexes_perms[i_best_perm], :], None


# #_______________________________methode1: proche en proche________________________________________________

# def distance(LV: pd.DataFrame, V: pd.Series):
#     """ Calcul la distance entre les villes (LV) et la ville V"""
#     return ((LV - V)**2).sum(axis=1).rename('distance')



# def m1_naive(LV: pd.DataFrame):
#     "va de proche en proche"
#     best_i_left, i_path, i_left = 0, [], list(range(LV.shape[0]))
#     for _ in range(LV.shape[0]-1):
#         i_path.append(best_i_left)
#         i_left.remove(i_path[-1])

#         CV = LV.iloc[i_path[-1],:]
#         d_left = distance(LV.iloc[i_left], CV)

#         best_i_left = i_left[d_left.argmin()]

#     assert len(i_left) == 1
#     i_path+=i_left

#     return LV.iloc[i_path], None




# #_______________________________methodeA: Hilbert curve ________________________________________________
# # %%
# def mirored_hilber_curve_points(iteration = 2, dim = 2):
#     if iteration == 9:
#         return np.load('mhcp9.npy')
#     if iteration >9:
#         raise Exception('Need to enhance code for more iterations!')
    
#     n_points = 4**iteration
#     hilbert_curve_index = list(range(n_points))
#     hilb_curve = np.array(HilbertCurve(n = dim, p = iteration).points_from_distances(hilbert_curve_index))/(2**iteration-1)
#     hilb_curve = pd.DataFrame(hilb_curve)
    
#     lower_part = hilb_curve[1] < 0.5
#     hilb_curve.loc[lower_part,[0]] = (hilb_curve.loc[~lower_part, 0]).values.reshape((n_points//2,-1))
#     hilb_curve.loc[lower_part,[1]] = (1 - hilb_curve.loc[~lower_part, 1]).values.reshape((n_points//2,-1))

#     n_to_miror = n_points//4
#     hilb_curve.iloc[0:n_to_miror,[0]] = hilb_curve.iloc[0:n_to_miror,[0]][::-1]
#     hilb_curve.iloc[-n_to_miror:,[0]] = hilb_curve.iloc[-n_to_miror:,[0]][::-1]

#     return hilb_curve.values

# def get_closest_hilbert_curve_index_points(iteration, LV, dim = 2):
#     """returns list of points of the curve of order n"""
#     hc_points = mirored_hilber_curve_points(iteration)
    
#     closest_p_index = LV.apply(
#         lambda row: ((row[['x', 'y']].values - hc_points)**2).sum(axis = 1).argmin(),
#         axis = 1
#     )
#     return closest_p_index

# # %%
# def m2_hilb(LV: pd.DataFrame, iteration = 1):
#     LV_normalized = (LV-LV.min())/(LV.max()-LV.min())
#     hc_indexes_col = "true_hc_index"
#     LV_normalized[hc_indexes_col] = 0
    
#     LV_normalized[hc_indexes_col] = get_closest_hilbert_curve_index_points(iteration, LV_normalized)
#     for iteration in range(iteration+1, 9):
#         LV_normalized_hc_index_value_count = LV_normalized[hc_indexes_col].value_counts()
#         are_one_ones = LV_normalized_hc_index_value_count == 1
#         print(f"One_ones: {are_one_ones.sum()/are_one_ones.shape[0]}")
#         if all(are_one_ones):
#             print(f"break at iteration {iteration-1}")
#             break
        
#         LV_normalized[hc_indexes_col] = get_closest_hilbert_curve_index_points(iteration, LV_normalized)

#     # bottom indices to upper indices
#     # sort according to upper -indices
#     LV[hc_indexes_col] = LV_normalized[hc_indexes_col]
#     return LV.sort_values(hc_indexes_col).set_index(hc_indexes_col), iteration-1


# #_______________________________methode2: cadrillage________________________________________________

        
# def GD(M,Va):
#     "renvoie deux listes de villes: plus a gauche/droite de la Va"
#     M=list(M)
#     M.append(Va)
#     M.sort()
#     iVa=M.index(Va)
#     M.remove(Va)
#     (G,D)=(M[0:iVa],M[iVa:len(M)])
#     return (G,D)


# def BH(M,Va):
#     "renvoie deux listes de villes: plus en bas/haut de la Va"
#     M.append(Va)
#     for elt in M:
#         elt.reverse()
#     M.sort()
#     for elt in M:
#         elt.reverse()
#     iVa=M.index(Va)
#     M.remove(Va)
#     (B,H)=(M[0:iVa],M[iVa:len(M)])
#     return (B,H)

    
# ##def nbVHBGD(LV):
# ##    BGHDV=[]
# ##    for elt in LV:
# ##        LV1=list(LV)
# ##        LV1.remove(elt)
# ##        (G,D)=GD(LV1,elt)
# ##        (BG,HG)=BH(G,elt)
# ##        (BD,HD)=BH(D,elt)
# ##        BGHDV.append(((len(BG),len(HG),len(HD),len(BD)),elt))
# ##        BGHDV.sort()
# ##    return BGHDV


    

# ##def M21(LV):
# ##    L=LV[0:1]
# ##    M=LV[1:len(LV)]
# ##    BGHDV=NBVcadri(LV)
# ##    
# ##    while len(L)!=len(LV) and M!=[]:
# ##        r

# def m2(LV):
#     "methode cadriallge"
#     L=[[0,0]]
#     M=[LV[i] for i in range(1,len(LV))]
    
#     while len(L)!=len(LV) and M!=[]:
#         Va=L[-1]
#         (G,D)=GD(M,Va)
#         (BG,HG)=BH(G,Va)
#         (BD,HD)=BH(D,Va)
#         RV=[]
#         for elt in [BG,HG,BD,HD]:
#             if elt!=[]:
#                 RV.append((len(elt),elt))
#         RV.sort()
#         (I,i) = (0,0)
#         while i<len(RV) and RV[0][0]==RV[i][0]:
#             i+=1
#             I+=1
#         LVP=[RV[i][1][j] for i in range(I) for j in range(len(RV[i][1]))]
#         Vp=VP(LVP,Va)
#         L.append(Vp)
#         M.remove(Vp)
#     L.append([0,0])
#     return L

                   



# #_______________________________methode3: genetique________________________________________________


# def CA(L):
#     "opere un changement aleatoir dans la liste de villes"
#     if len(L)<4:
#         return L
#     M=list(L)
#     i=random.randint(1,len(M)-2)
#     j=i
#     while i==j:
#         j=random.randint(1,len(M)-2)
#     (M[i],M[j])=(M[j],M[i])
#     return M


# def m3(LV):
#     "methode genetique"
#     II=5*len(LV)
#     E=[m1(LV),m2(LV),m4(LV),m5(LV),moym6(LV),medm6(LV)]*3
#     I,g = 0,0
#     DL=[(dtot(l),l) for l in E]
    
#     while I<II:
#         g+=1
# ##        if g%500==0:
# ##            print(g)
# ##            if I==0:
# ##                print([DL[i][0] for i in range(4)])
#         for i in range(len(E)):
#             L=E[i]
#             for n in range(4):
#                 l=list(L)
#                 for nbc in range(n): 
#                     l=CA(l)
#                 DL.append((dtot(l),l))
        
#         DL.sort()
#         E=[DL[i][1] for i in range(len(E))]
#         if int(DL[0][0])!=int(DL[1][0]) or int(DL[1][0])!=int(DL[2][0]):
#             if I!=0:
# ##                print('I:',I)
# ##                print([DL[i][0] for i in range(3)])
# ##                print('')
#                 I=0
                
#         elif g>20 and int(DL[0][0])==int(DL[1][0])==int(DL[2][0]):
#             I+=1
# ##            if I==1:
# ##                print('dS:',dtot(E[0]))
# ##                print('')

#     L=E[0]
#     return L
            
                
# def TNG(n):
#     "test l'evolution au fil des generation (M3)"
#     LV=crea(n)
#     I=0
#     E=[m1(LV)]*10
#     AR=[m1(LV)]
#     g=0
#     while I!=2000:
#         g+=1
#         DL=[(dtot(l),l) for l in E]
        
#         if g%500==0:
#             print(g)
            
#         for i in range(len(E)):
#             L=E[i]
#             for n in range(4):
#                 l=CA(L)
#                 DL.append((dtot(l),l))
#         DL.sort()
#         E=[DL[i][1] for i in range(len(E))]
        
#         if E[0]!=E[1] or E[1]!=E[2]:
#             if I!=0:
#                 print('I:',I)
#                 print('')
#                 I=0
                
#         elif g>20 and E[0]==E[1]==E[2]:
#             I+=1
#             S=E[0]
#             if I==1:
#                 print('dS:',dtot(S))
#                 print('')
# ##            if I==2000:
# ##                if S not in AR:
# ##                    AR.append(S)
# ##                return AR
#         NR=list(E[0])        
#         if g%400==0 and NR!=AR[-1]:
#             print('nr')
#             AR.append(list(E[0]))
#     if E[0] not in AR:
#         AR.append(E[0])
        
#     return AR




# #_______________________________methode4: barycentre________________________________________________

# def barycentre(LV):
#     "renvoie le barycentre de LV"
#     B=[0,0]
#     for elt in LV:
#         B[0]+=elt[0]/len(LV)
#         B[1]+=elt[1]/len(LV)
#     return B

# def CDP(o,ca):
#     """renvoie le coef dir de la perpendiculaire a la droite (o,ca)
#     passant par ca"""
#     if ca[1] == o[1]:
#         cdp=2**10
#     else:
#         cdp=-(ca[0]-o[0])/(ca[1]-o[1])
#         cdp=min(cdp,2**10)
#     return cdp

# def VSM4(B,Va,LVR):
#     "selectionne la ville suivante pour la M4"
#     cdp=CDP(B,Va)
# ##    print('cdp',cdp)
#     LVS=[]
#     if B[1]>cdp*(B[0]-Va[0])+Va[1]:
# ##        print('S')
#         for V in LVR:
#             if V[1]<=cdp*(V[0]-Va[0])+Va[1]:
#                 LVS.append(V)
# ##        print('lvs',LVS)
#         if LVS:
#             VS=VP(LVS,Va)
#         else:
#             VS=VP(LVR,Va)
            
#     else:
# ##        print('I')
#         for V in LVR:
#             if V[1]>=cdp*(V[0]-Va[0])+Va[1]:
#                 LVS.append(V)
#         if LVS:
#             VS=VP(LVS,Va)
#         else:
#             VS=VP(LVR,Va)
#     return VS


# def m4(LV):
#     "methode barycentre"
#     L=LV[0:1]
#     M=LV[1:len(LV)]
#     B=barycentre(LV)
#     while len(L)!=len(LV) and M!=[]:
#         Vs=VSM4(B,L[-1],M)
#         L.append(Vs)
#         M.remove(Vs)
#     L.append([0,0])
#     return L



# #________________________methode5: groupes de villes_________________________________



# ###############################methodes appropriées pour m5###########################



# def M5m0(LV):
#     LT=ArL(LV[1:len(LV)])
# ##    print(LT)
#     LT=[LV[0:1]+L+LV[0:1] for L in LT]
#     LDT=[(dtot(L),L) for L in LT]
# ##    print(LDT)
#     LDT.sort()
#     L=LDT[0][1]
#     return LDT[0][1][0:len(LV)]

# def M5m1(LV):
#     "va de proche en proche"
#     L=LV[0:1]
#     M=LV[1:len(LV)]
    
#     while len(L)!=len(LV) and M!=[]:
#         Va=L[-1]
#         vp=VP(M,Va)
#         L.append(vp)
#         M.remove(vp)
#     return L

# def M5m2(LV):
#     "methode cadriallge"
#     L=LV[0:1]
#     M=LV[1:len(LV)]
    
#     while len(L)!=len(LV) and M!=[]:
#         Va=L[-1]
#         (G,D)=GD(M,Va)
#         (BG,HG)=BH(G,Va)
#         (BD,HD)=BH(D,Va)
#         RV=[]
#         for elt in [BG,HG,BD,HD]:
#             if elt!=[]:
#                 RV.append((len(elt),elt))
#         RV.sort()
#         (I,i) = (0,0)
#         while i<len(RV) and RV[0][0]==RV[i][0]:
#             i+=1
#             I+=1
#         LVP=[RV[i][1][j] for i in range(I) for j in range(len(RV[i][1]))]
#         Vp=VP(LVP,Va)
#         L.append(Vp)
#         M.remove(Vp)
#     return L


# def M5m4(LV):
#     "methode barycentre"
#     L=LV[0:1]
#     M=LV[1:len(LV)]
#     B=barycentre(LV)
#     while len(L)!=len(LV) and M!=[]:
#         Vs=VSM4(B,L[-1],M)
#         L.append(Vs)
#         M.remove(Vs)
#     return L




# def M5moym6(LV):
#     "villes pas affectées"
#     M2=LV[0:len(LV)]
#     "villes en bout de chaine"
#     M1=[]
#     "villes encadrées par deux villes"
#     M0=[]
#     LCV=[]
#     df=(0,0)
#     while len(M0)!=len(LV) and M2+M1!=[]:
#         DV=DmoyV(M2+M1)
#         VL=DV[-1][1]
        
#         if len(M1)==2 and M2==[]:
#             M1=[]
#             M0+=M1
#         elif VL in M1:
#             M1.remove(VL)
#             if M1+M2!=[]:
#                 VLinM1(M2,M1,M0,LCV,VL)
                
#         else:
#             M2.remove(VL)
#             VLinM2(M2,M1,M0,LCV,VL)
#     L=LCV[0]
#     return L

# def M5medm6(LV):
#     "villes pas affectées"
#     M2=LV[0:len(LV)]
#     "villes en bout de chaine"
#     M1=[]
#     "villes encadrées par deux villes"
#     M0=[]
#     LCV=[]
#     while len(M0)!=len(LV) and M2+M1!=[]:
#         DV=DmedV(M2+M1)
#         VL=DV[-1][1]
        
#         if len(M1)==2 and M2==[]:
#             M1=[]
#             M0+=M1
#         elif VL in M1:
#             M1.remove(VL)
#             if M1+M2!=[]:
#                 VLinM1(M2,M1,M0,LCV,VL)
                
#         else:
#             M2.remove(VL)
#             VLinM2(M2,M1,M0,LCV,VL)
#     L=LCV[0]
#     return L


# #####################################################################################

# def gdV(LV):
#     """créée les groupes de villes (plus proche jusqua boucler)"""
#     M=list(LV)
#     LCV=[]
#     while M!=[]:
#         DV=DmoyV(M)
#         CV=[DV[-1][1]]
#         RV=list(M)
#         RV.remove(CV[0])
        
#         if RV:
#             vp=VP(RV,CV[-1])
#             while vp not in CV:
#                 if len(CV)>=2:
#                     RV.append(CV[-2])
#                 CV.append(vp)
#                 RV.remove(CV[-1])
#                 if RV:
#                     vp=VP(RV,CV[-1])

#         LCV.append(CV)
#         for V in CV:
#             M.remove(V)
#     return LCV
        

# def AireCoins(LChV):
#     "renvoie une liste de couples d'aire et couins des rectangles trié par aire"
#     Lrect=[]
#     LCV=list(LChV)
#     for CV in LCV:
#         CV.sort()
# ##        print('1ere derniere ville de CV,x',CV[0],CV[-1])
#         x=(CV[0][0],CV[-1][0])
# ##        print('x',x)
#         l=abs(x[0]-x[1])
#         for V in CV:
#             V.reverse()
#         CV.sort()
#         for V in CV:
#             V.reverse()
# ##        print('1ere derniere ville de CV,y',CV[0],CV[-1])
#         y=(CV[0][1],CV[-1][1])
# ####        print('y',y)
#         L=abs(y[0]-y[1])
#         Lrect.append((l*L,[(x[0],x[1]),(y[0],y[1])]))
#         Lrect.sort()
#     return Lrect




# def chevauchement(rect1,rect2):
#     "precise si les rectangles se chevauchent"
#     ncc=0
#     for xy in range(2):
#         for mM in range(2):
#             if rect2[xy][0]<=rect1[xy][mM]<=rect2[xy][1]:
#                 ncc+=1
#             if rect1[xy][0]<=rect2[xy][mM]<=rect1[xy][1]:
#                 ncc+=1
#     if ncc>=3:
#         return True
#     else:
#         return False
            
            
    

# def fusrect(LAC):
#     """renvoie une la liste des aires et coins en fusionnant les rectangles qui
#     chevauches"""
#     i=-1
#     while i<len(LAC)-1:
#         i+=1
#         j=i
# ##        print('LAC=',LAC)
# ##        print('')
# ##        input('1')
#         while j<len(LAC):
#             if j!=i :
# ##                print('ij',i,j)
# ##                print('LAC=',LAC)
# ##                print('')
#                 rect1,rect2=LAC[i][1],LAC[j][1]
# ##                print('R1,R2',rect1,rect2)
# ##                input()
#                 if chevauchement(rect1,rect2):
# ##                    print('____________________________________________________')
# ##                    print('')
# ##                    print('ij:',i,j)
# ##                    print('LAC=',LAC)
# ##                    print('LAC[i,j]',LAC[i],LAC[j])
# ##                    print('')
#                     if j<i:
#                         LAC.remove(LAC[i])
#                         LAC.remove(LAC[j])
#                     else:
#                         LAC.remove(LAC[j])
#                         LAC.remove(LAC[i])
#                     x=(min(rect1[0][0],rect2[0][0]),max(rect1[0][1],rect2[0][1]))
#                     y=(min(rect1[1][0],rect2[1][0]),max(rect1[1][1],rect2[1][1]))
#                     A=abs(x[0]-x[1])*abs(y[0]-y[1])
#                     LAC.append((A,[x,y]))
# ##                    print('LAC=',LAC)
# ##                    print('')
#                     (i,j)=(0,0)
# ##                    input('2')
#             j+=1
            
#     LAC.sort()
#     return LAC

# def meilconca(CV0,CV1):
#     "concatenne deux listes de ville de la 'meilleure' facon posible"
#     LDCV=[]
#     for r0 in range(2):
#         CV0.reverse()
#         for r1 in range(2):
#             CV1.reverse()
#             for i in range(len(CV0)):
#                 CV0=CV0[1:len(CV0)]+CV0[0:1]
#                 for j in range(len(CV1)):
#                     CV1=CV1[1:len(CV1)]+CV1[0:1]
#                     LDCV.append(CV0+CV1+CV0[0:1])
        
#     LDCV=[(dtot(l),l) for l in LDCV]
#     LDCV.sort()
#     return LDCV[0][1][0:-1]
                                 
            

     

# def concaprk(LCV):
#     """concatenne tout les parcours de la LCV pour renvoyer
#     une seule chaine de ville opti"""
#     LCV=[list(CV) for CV in LCV]
# ##    print('LCV',LCV)
#     while len(LCV)!=1:
# ##        print('LCV',LCV)
# ##        input('______________________')
# ##        print('LCV[0]',LCV[0])
# ##        print('LCV[1]',LCV[1])
# ##        print('')
#         LCV[0]=meilconca(LCV[0],LCV[1])
#         LCV.remove(LCV[1])
# ##        print('LDCV',LDCV)
# ##        print('')
# ##        print('LCV',LCV)
# ##        input('______________________')
#     return LCV[0]
        
        
        
        
        
    


# def m5(LV):
#     """methode des groupe de villes"""
# ##    print('LV=',LV)
# ##    print('')
#     LCV=gdV(LV)
# ##    print('LCV=',LCV)
# ##    print('')
#     LAC=AireCoins(LCV)
# ##    print('LAC=',LAC)
# ##    print('')
#     LAC=fusrect(LAC)
# ##    print('LAC=',LAC)
#     LAC.reverse()
#     M=list(LV)
# ##    print(M==LV)
# ##    print('')
#     LCV=[]
#     for elt in LAC:
# ##        print('M=',M)
# ##        print(elt)
# ##        input()
#         coins=elt[1]
#         CV=[]
#         for V in LV:
# ##            print('V',V)
# ##            print('coins',coins)
# ##            print('xX',coins[0][0],coins[0][1])
# ##            input()
#             if coins[0][0]<=V[0]<=coins[0][1]:
# ##                print('yY',coins[1][0],coins[1][1])
# ##                input()
#                 if coins[1][0]<=V[1]<=coins[1][1]:
# ##                    print('V',V)
#                     CV.append(V)
#                     M.remove(V)
#         if CV!=[]:
#             LCV.append(CV)
# ##    print('LCV=',LCV)
# ##    print('M:',M)
# ##    input()
#     Lcv=[]
#     for CV in LCV:
#         if len(CV)<4:
#             Lcv.append(CV)
#         elif len(CV)<8:
# ##            print('CV=',CV)
#             L=M5m0(CV)
#             Lcv.append(M5m0(CV))
#         elif len(CV)<70:
#             L=[M5medm6(CV),M5moym6(CV)]
# ##            print('L',L)
#             L=[(dtot(l),l) for l in L]
#             L.sort()
#             Lcv.append(L[0][1])
#         else:
#             L=[M5m1(CV),M5m2(CV),M5m4(CV)]
#             L=[(dtot(l),l) for l in L]
#             L.sort()
#             Lcv.append(L[0][1])
# ##    print('LCV',Lcv)
# ##    print('')
#     DCRCV=[]
    
#     for i in range(len(Lcv)):
#         CoorRecti=[(LAC[i][1][0][0]+LAC[i][1][0][1])/2,(LAC[i][1][1][0]+LAC[i][1][1][1])/2]
#         DCRCV.append((CoorRecti,Lcv[i]))
#     LCR=[CRCV[0] for CRCV in DCRCV]
#     DCRCV={tuple(CRCV[0]):tuple(CRCV[1]) for CRCV in DCRCV}
# ##    print('LCR',LCR)
#     if len(LCR)>3:
#         if len(LCR)<9:
#             LCR=M5m0(LCR)
#         else:
#             L=[M5medm6(LCR),M5moym6(LCR),M5m1(LCR),M5m2(LCR),M5m4(LCR)]
#             L=[(dtot(l),l) for l in L]
#             L.sort()
#             LCR=L[0][1]
# ##    print("DCRCV=",DCRCV)
# ##    print('')
# ##    print('LCR=',LCR)
#     LCV=[DCRCV[tuple(CR)] for CR in LCR]
# ##    print('LCV',LCV)
#     LCV=concaprk(LCV)
#     LCV=LCV+LCV[0:1]
#     return LCV



# def m5bis(LV):
# ##    print('LV=',LV)
# ##    print('')
#     LCV=gdV(LV)
# ##    print('LCV=',LCV)
# ##    print('')
#     LAC=AireCoins(LCV)
# ##    print('LAC=',LAC)
# ##    print('')
#     LAC.reverse()
#     M=list(LV)
# ##    print(M==LV)
# ##    print('')
#     LCV=[]
#     for elt in LAC:
# ##        print('M=',M)
# ##        print(elt)
# ##        input()
#         coins=elt[1]
#         CV=[]
#         for V in LV:
# ##            print('V',V)
# ##            print('coins',coins)
# ##            print('xX',coins[0][0],coins[0][1])
# ##            input()
#             if coins[0][0]<=V[0]<=coins[0][1]:
# ##                print('yY',coins[1][0],coins[1][1])
# ##                input()
#                 if coins[1][0]<=V[1]<=coins[1][1]:
#                     if V in M:
#                         CV.append(V)
#                         M.remove(V)
#         if CV!=[]:
#             LCV.append(CV)
# ##    print('LCV=',LCV)
# ##    print('M:',M)
# ##    input()
#     Lcv=[]
#     for CV in LCV:
#         if len(CV)<4:
#             Lcv.append(CV)
#         elif len(CV)<8:
# ##            print('CV=',CV)
#             L=M5m0(CV)
#             Lcv.append(M5m0(CV))
#         elif len(CV)<70:
#             L=[M5medm6(CV),M5moym6(CV)]
# ##            print('L',L)
#             L=[(dtot(l),l) for l in L]
#             L.sort()
#             Lcv.append(L[0][1])
#         else:
#             L=[M5m1(CV),M5m2(CV),M5m4(CV)]
#             L=[(dtot(l),l) for l in L]
#             L.sort()
#             Lcv.append(L[0][1])
# ##    print('LCV',Lcv)
# ##    print('')
#     DCRCV=[]
    
#     for i in range(len(Lcv)):
#         CoorRecti=[(LAC[i][1][0][0]+LAC[i][1][0][1])/2,(LAC[i][1][1][0]+LAC[i][1][1][1])/2]
#         DCRCV.append((CoorRecti,Lcv[i]))
#     LCR=[CRCV[0] for CRCV in DCRCV]
#     DCRCV={tuple(CRCV[0]):tuple(CRCV[1]) for CRCV in DCRCV}
# ##    print('LCR',LCR)
#     if len(LCR)>3:
#         if len(LCR)<9:
#             LCR=M5m0(LCR)
#         else:
#             L=[M5medm6(LCR),M5moym6(LCR),M5m1(LCR),M5m2(LCR),M5m4(LCR)]
#             L=[(dtot(l),l) for l in L]
#             L.sort()
#             LCR=L[0][1]
# ##    print("DCRCV=",DCRCV)
# ##    print('')
# ##    print('LCR=',LCR)
#     LCV=[DCRCV[tuple(CR)] for CR in LCR]
# ##    print('LCV',LCV)
#     LCV=concaprk(LCV)
#     LCV=LCV+LCV[0:1]
#     return LCV


    




# #________________________methode6: villes lointaines 1rst________________________________________________


# def DmoyV(Lv):
#     "renvoie la distance moyenne de chaque ville par rap aux autres"
#     LV=list(Lv)
#     DV=[]
#     for i in range(len(LV)):
#         D=0
#         for j in range(len(LV)):
#             if i!=j:
#                 D+=((LV[i][0]-LV[j][0])**2+(LV[i][1]-LV[j][1])**2)**0.5
#         DV.append((D,LV[i]))
#     DV.sort()
#     return DV
        
            
    
# def DmedV(LV):
#     "renvoie la distance medianne de chaque ville par rap aux autres"
#     DV=[]
#     for i in range(len(LV)):
#         D=[]
#         for j in range(len(LV)):
#             if i!=j:
#                 D.append(((LV[i][0]-LV[j][0])**2+(LV[i][1]-LV[j][1])**2)**0.5)
#         DV.append((D[int(len(D)/2)],LV[i]))
#     DV.sort()
#     return DV

# def VLinM1(M2,M1,M0,LCV,VL):
    
# ##  dans quelle chaine? -> on met la ville a la fin de la chaine
#     for CV in LCV:
#         if CV[-1]==VL:
#             CV1=CV
#             LCV.remove(CV)
#         elif CV[0]==VL:
#             CV.reverse()
#             CV1=CV
#             LCV.remove(CV)
            
#     MCV1=M2+M1
#     vp=VP(MCV1,VL)
#     while vp in CV1:
#         MCV1.remove(vp)
#         vp=VP(MCV1,VL)
# ##  vp en bout de chaine?    
#     if vp in M1:         
# ##      dans quelle chaine? -> on met la ville au debut de la chaine
#         for CV in LCV:
#             if CV[0]==vp:
#                 CV2=CV
#                 LCV.remove(CV)
#             elif CV[-1]==vp:
#                 CV.reverse()
#                 CV2=CV
#                 LCV.remove(CV)
        

#         LCV.append(CV1+CV2)
#         M1.remove(vp)
#         M0.append(vp)
        
# ##      vp pas affectée
#     else:
#         LCV.append(CV1+[vp])
#         M2.remove(vp)
#         M1.append(vp)
#         M0.append(VL)
    

# def VLinM2(M2,M1,M0,LCV,VL):
#     if VL in M2:
#         M2.remove(VL)
#     elif VL in M1:
#         M1.remove(VL)
#     vp=VP(M2+M1,VL)
# ##  vp en bout de chaine?
#     if vp in M1:
# ##      dans quelle chaine? -> on ajoute VL a coté de vp en bout de la chaine
#         for i in range(len(LCV)):
#             if LCV[i][0]==vp:
#                 LCV[i]=[VL]+LCV[i]
#             elif LCV[i][-1]==vp:
#                 LCV[i]=LCV[i]+[VL]
#         M1.remove(vp)
#         M0.append(vp)
#         M1.append(VL)
#     else:
#         CV=[VL,vp]
#         LCV.append(CV)
#         M2.remove(vp)
#         M1.append(vp)
#         M1.append(VL)
        
            
    
        



    
# def medm6(LV):
#     "villes pas affectées"
#     M2=LV[0:len(LV)]
#     "villes en bout de chaine"
#     M1=[]
#     "villes encadrées par deux villes"
#     M0=[]
#     LCV=[]
#     while len(M0)!=len(LV) and M2+M1!=[]:
#         DV=DmedV(M2+M1)
#         VL=DV[-1][1]
        
#         if len(M1)==2 and M2==[]:
#             M1=[]
#             M0+=M1
#         elif VL in M1:
#             M1.remove(VL)
#             if M1+M2!=[]:
#                 VLinM1(M2,M1,M0,LCV,VL)
                
#         else:
#             M2.remove(VL)
#             VLinM2(M2,M1,M0,LCV,VL)
#     L=LCV[0]
#     L=L[L.index([0,0]):len(L)]+L[0:L.index([0,0])]+[[0,0]]
#     return L


# def moym6(LV):
#     "villes pas affectées"
#     M2=LV[0:len(LV)]
#     "villes en bout de chaine"
#     M1=[]
#     "villes encadrées par deux villes"
#     M0=[]
#     LCV=[]
#     while len(M0)!=len(LV) and M2+M1!=[]:
#         DV=DmoyV(M2+M1)
#         VL=DV[-1][1]
        
#         if len(M1)==2 and M2==[]:
#             M1=[]
#             M0+=M1
#         elif VL in M1:
#             M1.remove(VL)
#             if M1+M2!=[]:
#                 VLinM1(M2,M1,M0,LCV,VL)
                
#         else:
#             M2.remove(VL)
#             VLinM2(M2,M1,M0,LCV,VL)
#     L=LCV[0]
#     L=L[L.index([0,0]):len(L)]+L[0:L.index([0,0])]+[[0,0]]
#     return L

