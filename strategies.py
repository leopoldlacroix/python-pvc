# %% import random
import time

import numpy as np
import pandas as pd

from itertools import permutations


import plotly.express as px


def dtot(LV : pd.DataFrame):
    "Distance tot d'un parcours formule: sqrt(somme((Xi-X'i)**2)). \n referme le chemin!!"
    circuit = pd.concat([LV,LV[0:1]])
    d = (circuit.diff()**2).sum()
    return d.x + d.y

def crea(n:int):
    "créer n villes"
    return pd.DataFrame(
        np.random.random(size=(n,2)),
        columns=['x', 'y']
    )
LV = crea(6)

def plot_path(path: pd.DataFrame, elapsed = 0):

    circuit = pd.concat([path,path[0:1]])
    circuit = circuit.reset_index(names=["id"])
    score = dtot(circuit)
    fig = px.line(data_frame = circuit, x="x", y="y", text="id", title=f'Score: {round(score, 4)} \t elapsed {round(elapsed, 4)}')
    fig.update_traces(textposition='top left')
    fig.show()

def tst_strategy(strategy):
    start = time.time()
    path = strategy(LV)
    end = time.time()

    elapsed = end - start
    plot_path(path, elapsed)
    


plot_path(LV)



#_____________________________ strats!________________________________________________


        

#_____________________________methode0: le meilleur!________________________________________________

    
def m0(LV:pd.DataFrame):
    "tests all possibilities"
    indexes_perms = pd.Series(permutations(LV.index))
    scores = indexes_perms.apply(lambda idx: dtot(LV.loc[idx,:]))
    i_best_perm = np.argmin(scores)
    return LV.loc[indexes_perms[i_best_perm], :]


tst_strategy(m0)


#_______________________________methode1: proche en proche________________________________________________

def distance(LV: pd.DataFrame, V: pd.Series):
    """ Calcul la distance entre les villes (LV) et la ville V"""
    return ((LV - V)**2).sum(axis=1).rename('distance')



def m1(LV: pd.DataFrame):
    "va de proche en proche"
    best_i_left, i_path, i_left = 0, [], list(range(LV.shape[0]))
    for _ in range(LV.shape[0]-1):
        i_path.append(best_i_left)
        i_left.remove(i_path[-1])

        CV = LV.iloc[i_path[-1],:]
        d_left = distance(LV.iloc[i_left], CV)

        best_i_left = i_left[d_left.argmin()]

    assert len(i_left) == 1
    i_path+=i_left

    return LV.iloc[i_path]

tst_strategy(m1)

#_______________________________methodeA: Hilbert curve ________________________________________________
# %%
def hilbert_curve(order):
    n = 4**order
    n_digits =int(np.sqrt(n))

    bin_rep = np.vectorize(lambda x: bin(x)[2:].rjust(n_digits, '0'))(np.arange(n))

    hilbert_df = pd.Series(bin_rep).str.extract(r"(\d\d)"*int(n_digits//2), expand = True)
    hilbert_df.columns = hilbert_df.columns[::-1]

    sort_index = hilbert_df.replace({"00":0, "01":1, "11":2, "10":3})
    sort_index = sort_index.apply(lambda x: x*4**x.name).sum(axis=1)

    hilbert_df = hilbert_df.applymap(lambda x: np.array(tuple(x), dtype=int))
    hilbert_df = hilbert_df.apply(lambda s: s*(2**s.name))

    hilbert_df = hilbert_df.sum(axis=1).to_frame()
    hilbert_df['id'] = bin_rep
    hilbert_df['index'] = sort_index
    hilbert_df = hilbert_df.sort_values("index")
    hilbert_df[['x', 'y']] = hilbert_df[0].tolist()
    return hilbert_df

px.line(data_frame = hilbert_curve(1), x = "x", y = "y").show()
px.line(data_frame = hilbert_curve(2), x = "x", y = "y").show()
px.line(data_frame = hilbert_curve(3), x = "x", y = "y").show()
# %%

# first hilb
# 
# index attributed NAN
# while not all hilb point:
    # index attributed * 4 (4 times more points)
    # hilb order n+1 
    # distance to each hilb
    # if one/one => attribue index of hilb point
# return sorted array with hilb attributed index 





#_______________________________methode2: cadrillage________________________________________________

        
def GD(M,Va):
    "renvoie deux listes de villes: plus a gauche/droite de la Va"
    M=list(M)
    M.append(Va)
    M.sort()
    iVa=M.index(Va)
    M.remove(Va)
    (G,D)=(M[0:iVa],M[iVa:len(M)])
    return (G,D)


def BH(M,Va):
    "renvoie deux listes de villes: plus en bas/haut de la Va"
    M.append(Va)
    for elt in M:
        elt.reverse()
    M.sort()
    for elt in M:
        elt.reverse()
    iVa=M.index(Va)
    M.remove(Va)
    (B,H)=(M[0:iVa],M[iVa:len(M)])
    return (B,H)

    
##def nbVHBGD(LV):
##    BGHDV=[]
##    for elt in LV:
##        LV1=list(LV)
##        LV1.remove(elt)
##        (G,D)=GD(LV1,elt)
##        (BG,HG)=BH(G,elt)
##        (BD,HD)=BH(D,elt)
##        BGHDV.append(((len(BG),len(HG),len(HD),len(BD)),elt))
##        BGHDV.sort()
##    return BGHDV


    

##def M21(LV):
##    L=LV[0:1]
##    M=LV[1:len(LV)]
##    BGHDV=NBVcadri(LV)
##    
##    while len(L)!=len(LV) and M!=[]:
##        r

def m2(LV):
    "methode cadriallge"
    L=[[0,0]]
    M=[LV[i] for i in range(1,len(LV))]
    
    while len(L)!=len(LV) and M!=[]:
        Va=L[-1]
        (G,D)=GD(M,Va)
        (BG,HG)=BH(G,Va)
        (BD,HD)=BH(D,Va)
        RV=[]
        for elt in [BG,HG,BD,HD]:
            if elt!=[]:
                RV.append((len(elt),elt))
        RV.sort()
        (I,i) = (0,0)
        while i<len(RV) and RV[0][0]==RV[i][0]:
            i+=1
            I+=1
        LVP=[RV[i][1][j] for i in range(I) for j in range(len(RV[i][1]))]
        Vp=VP(LVP,Va)
        L.append(Vp)
        M.remove(Vp)
    L.append([0,0])
    return L

                   



#_______________________________methode3: genetique________________________________________________


def CA(L):
    "opere un changement aleatoir dans la liste de villes"
    if len(L)<4:
        return L
    M=list(L)
    i=random.randint(1,len(M)-2)
    j=i
    while i==j:
        j=random.randint(1,len(M)-2)
    (M[i],M[j])=(M[j],M[i])
    return M


def m3(LV):
    "methode genetique"
    II=5*len(LV)
    E=[m1(LV),m2(LV),m4(LV),m5(LV),moym6(LV),medm6(LV)]*3
    I,g = 0,0
    DL=[(dtot(l),l) for l in E]
    
    while I<II:
        g+=1
##        if g%500==0:
##            print(g)
##            if I==0:
##                print([DL[i][0] for i in range(4)])
        for i in range(len(E)):
            L=E[i]
            for n in range(4):
                l=list(L)
                for nbc in range(n): 
                    l=CA(l)
                DL.append((dtot(l),l))
        
        DL.sort()
        E=[DL[i][1] for i in range(len(E))]
        if int(DL[0][0])!=int(DL[1][0]) or int(DL[1][0])!=int(DL[2][0]):
            if I!=0:
##                print('I:',I)
##                print([DL[i][0] for i in range(3)])
##                print('')
                I=0
                
        elif g>20 and int(DL[0][0])==int(DL[1][0])==int(DL[2][0]):
            I+=1
##            if I==1:
##                print('dS:',dtot(E[0]))
##                print('')

    L=E[0]
    return L
            
                
def TNG(n):
    "test l'evolution au fil des generation (M3)"
    LV=crea(n)
    I=0
    E=[m1(LV)]*10
    AR=[m1(LV)]
    g=0
    while I!=2000:
        g+=1
        DL=[(dtot(l),l) for l in E]
        
        if g%500==0:
            print(g)
            
        for i in range(len(E)):
            L=E[i]
            for n in range(4):
                l=CA(L)
                DL.append((dtot(l),l))
        DL.sort()
        E=[DL[i][1] for i in range(len(E))]
        
        if E[0]!=E[1] or E[1]!=E[2]:
            if I!=0:
                print('I:',I)
                print('')
                I=0
                
        elif g>20 and E[0]==E[1]==E[2]:
            I+=1
            S=E[0]
            if I==1:
                print('dS:',dtot(S))
                print('')
##            if I==2000:
##                if S not in AR:
##                    AR.append(S)
##                return AR
        NR=list(E[0])        
        if g%400==0 and NR!=AR[-1]:
            print('nr')
            AR.append(list(E[0]))
    if E[0] not in AR:
        AR.append(E[0])
        
    return AR




#_______________________________methode4: barycentre________________________________________________

def barycentre(LV):
    "renvoie le barycentre de LV"
    B=[0,0]
    for elt in LV:
        B[0]+=elt[0]/len(LV)
        B[1]+=elt[1]/len(LV)
    return B

def CDP(o,ca):
    """renvoie le coef dir de la perpendiculaire a la droite (o,ca)
    passant par ca"""
    if ca[1] == o[1]:
        cdp=2**10
    else:
        cdp=-(ca[0]-o[0])/(ca[1]-o[1])
        cdp=min(cdp,2**10)
    return cdp

def VSM4(B,Va,LVR):
    "selectionne la ville suivante pour la M4"
    cdp=CDP(B,Va)
##    print('cdp',cdp)
    LVS=[]
    if B[1]>cdp*(B[0]-Va[0])+Va[1]:
##        print('S')
        for V in LVR:
            if V[1]<=cdp*(V[0]-Va[0])+Va[1]:
                LVS.append(V)
##        print('lvs',LVS)
        if LVS:
            VS=VP(LVS,Va)
        else:
            VS=VP(LVR,Va)
            
    else:
##        print('I')
        for V in LVR:
            if V[1]>=cdp*(V[0]-Va[0])+Va[1]:
                LVS.append(V)
        if LVS:
            VS=VP(LVS,Va)
        else:
            VS=VP(LVR,Va)
    return VS


def m4(LV):
    "methode barycentre"
    L=LV[0:1]
    M=LV[1:len(LV)]
    B=barycentre(LV)
    while len(L)!=len(LV) and M!=[]:
        Vs=VSM4(B,L[-1],M)
        L.append(Vs)
        M.remove(Vs)
    L.append([0,0])
    return L



#________________________methode5: groupes de villes_________________________________



###############################methodes appropriées pour m5###########################



def M5m0(LV):
    LT=ArL(LV[1:len(LV)])
##    print(LT)
    LT=[LV[0:1]+L+LV[0:1] for L in LT]
    LDT=[(dtot(L),L) for L in LT]
##    print(LDT)
    LDT.sort()
    L=LDT[0][1]
    return LDT[0][1][0:len(LV)]

def M5m1(LV):
    "va de proche en proche"
    L=LV[0:1]
    M=LV[1:len(LV)]
    
    while len(L)!=len(LV) and M!=[]:
        Va=L[-1]
        vp=VP(M,Va)
        L.append(vp)
        M.remove(vp)
    return L

def M5m2(LV):
    "methode cadriallge"
    L=LV[0:1]
    M=LV[1:len(LV)]
    
    while len(L)!=len(LV) and M!=[]:
        Va=L[-1]
        (G,D)=GD(M,Va)
        (BG,HG)=BH(G,Va)
        (BD,HD)=BH(D,Va)
        RV=[]
        for elt in [BG,HG,BD,HD]:
            if elt!=[]:
                RV.append((len(elt),elt))
        RV.sort()
        (I,i) = (0,0)
        while i<len(RV) and RV[0][0]==RV[i][0]:
            i+=1
            I+=1
        LVP=[RV[i][1][j] for i in range(I) for j in range(len(RV[i][1]))]
        Vp=VP(LVP,Va)
        L.append(Vp)
        M.remove(Vp)
    return L


def M5m4(LV):
    "methode barycentre"
    L=LV[0:1]
    M=LV[1:len(LV)]
    B=barycentre(LV)
    while len(L)!=len(LV) and M!=[]:
        Vs=VSM4(B,L[-1],M)
        L.append(Vs)
        M.remove(Vs)
    return L




def M5moym6(LV):
    "villes pas affectées"
    M2=LV[0:len(LV)]
    "villes en bout de chaine"
    M1=[]
    "villes encadrées par deux villes"
    M0=[]
    LCV=[]
    df=(0,0)
    while len(M0)!=len(LV) and M2+M1!=[]:
        DV=DmoyV(M2+M1)
        VL=DV[-1][1]
        
        if len(M1)==2 and M2==[]:
            M1=[]
            M0+=M1
        elif VL in M1:
            M1.remove(VL)
            if M1+M2!=[]:
                VLinM1(M2,M1,M0,LCV,VL)
                
        else:
            M2.remove(VL)
            VLinM2(M2,M1,M0,LCV,VL)
    L=LCV[0]
    return L

def M5medm6(LV):
    "villes pas affectées"
    M2=LV[0:len(LV)]
    "villes en bout de chaine"
    M1=[]
    "villes encadrées par deux villes"
    M0=[]
    LCV=[]
    while len(M0)!=len(LV) and M2+M1!=[]:
        DV=DmedV(M2+M1)
        VL=DV[-1][1]
        
        if len(M1)==2 and M2==[]:
            M1=[]
            M0+=M1
        elif VL in M1:
            M1.remove(VL)
            if M1+M2!=[]:
                VLinM1(M2,M1,M0,LCV,VL)
                
        else:
            M2.remove(VL)
            VLinM2(M2,M1,M0,LCV,VL)
    L=LCV[0]
    return L


#####################################################################################

def gdV(LV):
    """créée les groupes de villes (plus proche jusqua boucler)"""
    M=list(LV)
    LCV=[]
    while M!=[]:
        DV=DmoyV(M)
        CV=[DV[-1][1]]
        RV=list(M)
        RV.remove(CV[0])
        
        if RV:
            vp=VP(RV,CV[-1])
            while vp not in CV:
                if len(CV)>=2:
                    RV.append(CV[-2])
                CV.append(vp)
                RV.remove(CV[-1])
                if RV:
                    vp=VP(RV,CV[-1])

        LCV.append(CV)
        for V in CV:
            M.remove(V)
    return LCV
        

def AireCoins(LChV):
    "renvoie une liste de couples d'aire et couins des rectangles trié par aire"
    Lrect=[]
    LCV=list(LChV)
    for CV in LCV:
        CV.sort()
##        print('1ere derniere ville de CV,x',CV[0],CV[-1])
        x=(CV[0][0],CV[-1][0])
##        print('x',x)
        l=abs(x[0]-x[1])
        for V in CV:
            V.reverse()
        CV.sort()
        for V in CV:
            V.reverse()
##        print('1ere derniere ville de CV,y',CV[0],CV[-1])
        y=(CV[0][1],CV[-1][1])
####        print('y',y)
        L=abs(y[0]-y[1])
        Lrect.append((l*L,[(x[0],x[1]),(y[0],y[1])]))
        Lrect.sort()
    return Lrect




def chevauchement(rect1,rect2):
    "precise si les rectangles se chevauchent"
    ncc=0
    for xy in range(2):
        for mM in range(2):
            if rect2[xy][0]<=rect1[xy][mM]<=rect2[xy][1]:
                ncc+=1
            if rect1[xy][0]<=rect2[xy][mM]<=rect1[xy][1]:
                ncc+=1
    if ncc>=3:
        return True
    else:
        return False
            
            
    

def fusrect(LAC):
    """renvoie une la liste des aires et coins en fusionnant les rectangles qui
    chevauches"""
    i=-1
    while i<len(LAC)-1:
        i+=1
        j=i
##        print('LAC=',LAC)
##        print('')
##        input('1')
        while j<len(LAC):
            if j!=i :
##                print('ij',i,j)
##                print('LAC=',LAC)
##                print('')
                rect1,rect2=LAC[i][1],LAC[j][1]
##                print('R1,R2',rect1,rect2)
##                input()
                if chevauchement(rect1,rect2):
##                    print('____________________________________________________')
##                    print('')
##                    print('ij:',i,j)
##                    print('LAC=',LAC)
##                    print('LAC[i,j]',LAC[i],LAC[j])
##                    print('')
                    if j<i:
                        LAC.remove(LAC[i])
                        LAC.remove(LAC[j])
                    else:
                        LAC.remove(LAC[j])
                        LAC.remove(LAC[i])
                    x=(min(rect1[0][0],rect2[0][0]),max(rect1[0][1],rect2[0][1]))
                    y=(min(rect1[1][0],rect2[1][0]),max(rect1[1][1],rect2[1][1]))
                    A=abs(x[0]-x[1])*abs(y[0]-y[1])
                    LAC.append((A,[x,y]))
##                    print('LAC=',LAC)
##                    print('')
                    (i,j)=(0,0)
##                    input('2')
            j+=1
            
    LAC.sort()
    return LAC

def meilconca(CV0,CV1):
    "concatenne deux listes de ville de la 'meilleure' facon posible"
    LDCV=[]
    for r0 in range(2):
        CV0.reverse()
        for r1 in range(2):
            CV1.reverse()
            for i in range(len(CV0)):
                CV0=CV0[1:len(CV0)]+CV0[0:1]
                for j in range(len(CV1)):
                    CV1=CV1[1:len(CV1)]+CV1[0:1]
                    LDCV.append(CV0+CV1+CV0[0:1])
        
    LDCV=[(dtot(l),l) for l in LDCV]
    LDCV.sort()
    return LDCV[0][1][0:-1]
                                 
            

     

def concaprk(LCV):
    """concatenne tout les parcours de la LCV pour renvoyer
    une seule chaine de ville opti"""
    LCV=[list(CV) for CV in LCV]
##    print('LCV',LCV)
    while len(LCV)!=1:
##        print('LCV',LCV)
##        input('______________________')
##        print('LCV[0]',LCV[0])
##        print('LCV[1]',LCV[1])
##        print('')
        LCV[0]=meilconca(LCV[0],LCV[1])
        LCV.remove(LCV[1])
##        print('LDCV',LDCV)
##        print('')
##        print('LCV',LCV)
##        input('______________________')
    return LCV[0]
        
        
        
        
        
    


def m5(LV):
    """methode des groupe de villes"""
##    print('LV=',LV)
##    print('')
    LCV=gdV(LV)
##    print('LCV=',LCV)
##    print('')
    LAC=AireCoins(LCV)
##    print('LAC=',LAC)
##    print('')
    LAC=fusrect(LAC)
##    print('LAC=',LAC)
    LAC.reverse()
    M=list(LV)
##    print(M==LV)
##    print('')
    LCV=[]
    for elt in LAC:
##        print('M=',M)
##        print(elt)
##        input()
        coins=elt[1]
        CV=[]
        for V in LV:
##            print('V',V)
##            print('coins',coins)
##            print('xX',coins[0][0],coins[0][1])
##            input()
            if coins[0][0]<=V[0]<=coins[0][1]:
##                print('yY',coins[1][0],coins[1][1])
##                input()
                if coins[1][0]<=V[1]<=coins[1][1]:
##                    print('V',V)
                    CV.append(V)
                    M.remove(V)
        if CV!=[]:
            LCV.append(CV)
##    print('LCV=',LCV)
##    print('M:',M)
##    input()
    Lcv=[]
    for CV in LCV:
        if len(CV)<4:
            Lcv.append(CV)
        elif len(CV)<8:
##            print('CV=',CV)
            L=M5m0(CV)
            Lcv.append(M5m0(CV))
        elif len(CV)<70:
            L=[M5medm6(CV),M5moym6(CV)]
##            print('L',L)
            L=[(dtot(l),l) for l in L]
            L.sort()
            Lcv.append(L[0][1])
        else:
            L=[M5m1(CV),M5m2(CV),M5m4(CV)]
            L=[(dtot(l),l) for l in L]
            L.sort()
            Lcv.append(L[0][1])
##    print('LCV',Lcv)
##    print('')
    DCRCV=[]
    
    for i in range(len(Lcv)):
        CoorRecti=[(LAC[i][1][0][0]+LAC[i][1][0][1])/2,(LAC[i][1][1][0]+LAC[i][1][1][1])/2]
        DCRCV.append((CoorRecti,Lcv[i]))
    LCR=[CRCV[0] for CRCV in DCRCV]
    DCRCV={tuple(CRCV[0]):tuple(CRCV[1]) for CRCV in DCRCV}
##    print('LCR',LCR)
    if len(LCR)>3:
        if len(LCR)<9:
            LCR=M5m0(LCR)
        else:
            L=[M5medm6(LCR),M5moym6(LCR),M5m1(LCR),M5m2(LCR),M5m4(LCR)]
            L=[(dtot(l),l) for l in L]
            L.sort()
            LCR=L[0][1]
##    print("DCRCV=",DCRCV)
##    print('')
##    print('LCR=',LCR)
    LCV=[DCRCV[tuple(CR)] for CR in LCR]
##    print('LCV',LCV)
    LCV=concaprk(LCV)
    LCV=LCV+LCV[0:1]
    return LCV



def m5bis(LV):
##    print('LV=',LV)
##    print('')
    LCV=gdV(LV)
##    print('LCV=',LCV)
##    print('')
    LAC=AireCoins(LCV)
##    print('LAC=',LAC)
##    print('')
    LAC.reverse()
    M=list(LV)
##    print(M==LV)
##    print('')
    LCV=[]
    for elt in LAC:
##        print('M=',M)
##        print(elt)
##        input()
        coins=elt[1]
        CV=[]
        for V in LV:
##            print('V',V)
##            print('coins',coins)
##            print('xX',coins[0][0],coins[0][1])
##            input()
            if coins[0][0]<=V[0]<=coins[0][1]:
##                print('yY',coins[1][0],coins[1][1])
##                input()
                if coins[1][0]<=V[1]<=coins[1][1]:
                    if V in M:
                        CV.append(V)
                        M.remove(V)
        if CV!=[]:
            LCV.append(CV)
##    print('LCV=',LCV)
##    print('M:',M)
##    input()
    Lcv=[]
    for CV in LCV:
        if len(CV)<4:
            Lcv.append(CV)
        elif len(CV)<8:
##            print('CV=',CV)
            L=M5m0(CV)
            Lcv.append(M5m0(CV))
        elif len(CV)<70:
            L=[M5medm6(CV),M5moym6(CV)]
##            print('L',L)
            L=[(dtot(l),l) for l in L]
            L.sort()
            Lcv.append(L[0][1])
        else:
            L=[M5m1(CV),M5m2(CV),M5m4(CV)]
            L=[(dtot(l),l) for l in L]
            L.sort()
            Lcv.append(L[0][1])
##    print('LCV',Lcv)
##    print('')
    DCRCV=[]
    
    for i in range(len(Lcv)):
        CoorRecti=[(LAC[i][1][0][0]+LAC[i][1][0][1])/2,(LAC[i][1][1][0]+LAC[i][1][1][1])/2]
        DCRCV.append((CoorRecti,Lcv[i]))
    LCR=[CRCV[0] for CRCV in DCRCV]
    DCRCV={tuple(CRCV[0]):tuple(CRCV[1]) for CRCV in DCRCV}
##    print('LCR',LCR)
    if len(LCR)>3:
        if len(LCR)<9:
            LCR=M5m0(LCR)
        else:
            L=[M5medm6(LCR),M5moym6(LCR),M5m1(LCR),M5m2(LCR),M5m4(LCR)]
            L=[(dtot(l),l) for l in L]
            L.sort()
            LCR=L[0][1]
##    print("DCRCV=",DCRCV)
##    print('')
##    print('LCR=',LCR)
    LCV=[DCRCV[tuple(CR)] for CR in LCR]
##    print('LCV',LCV)
    LCV=concaprk(LCV)
    LCV=LCV+LCV[0:1]
    return LCV


    




#________________________methode6: villes lointaines 1rst________________________________________________


def DmoyV(Lv):
    "renvoie la distance moyenne de chaque ville par rap aux autres"
    LV=list(Lv)
    DV=[]
    for i in range(len(LV)):
        D=0
        for j in range(len(LV)):
            if i!=j:
                D+=((LV[i][0]-LV[j][0])**2+(LV[i][1]-LV[j][1])**2)**0.5
        DV.append((D,LV[i]))
    DV.sort()
    return DV
        
            
    
def DmedV(LV):
    "renvoie la distance medianne de chaque ville par rap aux autres"
    DV=[]
    for i in range(len(LV)):
        D=[]
        for j in range(len(LV)):
            if i!=j:
                D.append(((LV[i][0]-LV[j][0])**2+(LV[i][1]-LV[j][1])**2)**0.5)
        DV.append((D[int(len(D)/2)],LV[i]))
    DV.sort()
    return DV

def VLinM1(M2,M1,M0,LCV,VL):
    
##  dans quelle chaine? -> on met la ville a la fin de la chaine
    for CV in LCV:
        if CV[-1]==VL:
            CV1=CV
            LCV.remove(CV)
        elif CV[0]==VL:
            CV.reverse()
            CV1=CV
            LCV.remove(CV)
            
    MCV1=M2+M1
    vp=VP(MCV1,VL)
    while vp in CV1:
        MCV1.remove(vp)
        vp=VP(MCV1,VL)
##  vp en bout de chaine?    
    if vp in M1:         
##      dans quelle chaine? -> on met la ville au debut de la chaine
        for CV in LCV:
            if CV[0]==vp:
                CV2=CV
                LCV.remove(CV)
            elif CV[-1]==vp:
                CV.reverse()
                CV2=CV
                LCV.remove(CV)
        

        LCV.append(CV1+CV2)
        M1.remove(vp)
        M0.append(vp)
        
##      vp pas affectée
    else:
        LCV.append(CV1+[vp])
        M2.remove(vp)
        M1.append(vp)
        M0.append(VL)
    

def VLinM2(M2,M1,M0,LCV,VL):
    if VL in M2:
        M2.remove(VL)
    elif VL in M1:
        M1.remove(VL)
    vp=VP(M2+M1,VL)
##  vp en bout de chaine?
    if vp in M1:
##      dans quelle chaine? -> on ajoute VL a coté de vp en bout de la chaine
        for i in range(len(LCV)):
            if LCV[i][0]==vp:
                LCV[i]=[VL]+LCV[i]
            elif LCV[i][-1]==vp:
                LCV[i]=LCV[i]+[VL]
        M1.remove(vp)
        M0.append(vp)
        M1.append(VL)
    else:
        CV=[VL,vp]
        LCV.append(CV)
        M2.remove(vp)
        M1.append(vp)
        M1.append(VL)
        
            
    
        



    
def medm6(LV):
    "villes pas affectées"
    M2=LV[0:len(LV)]
    "villes en bout de chaine"
    M1=[]
    "villes encadrées par deux villes"
    M0=[]
    LCV=[]
    while len(M0)!=len(LV) and M2+M1!=[]:
        DV=DmedV(M2+M1)
        VL=DV[-1][1]
        
        if len(M1)==2 and M2==[]:
            M1=[]
            M0+=M1
        elif VL in M1:
            M1.remove(VL)
            if M1+M2!=[]:
                VLinM1(M2,M1,M0,LCV,VL)
                
        else:
            M2.remove(VL)
            VLinM2(M2,M1,M0,LCV,VL)
    L=LCV[0]
    L=L[L.index([0,0]):len(L)]+L[0:L.index([0,0])]+[[0,0]]
    return L


def moym6(LV):
    "villes pas affectées"
    M2=LV[0:len(LV)]
    "villes en bout de chaine"
    M1=[]
    "villes encadrées par deux villes"
    M0=[]
    LCV=[]
    while len(M0)!=len(LV) and M2+M1!=[]:
        DV=DmoyV(M2+M1)
        VL=DV[-1][1]
        
        if len(M1)==2 and M2==[]:
            M1=[]
            M0+=M1
        elif VL in M1:
            M1.remove(VL)
            if M1+M2!=[]:
                VLinM1(M2,M1,M0,LCV,VL)
                
        else:
            M2.remove(VL)
            VLinM2(M2,M1,M0,LCV,VL)
    L=LCV[0]
    L=L[L.index([0,0]):len(L)]+L[0:L.index([0,0])]+[[0,0]]
    return L




##-------------------------------------test--------------------------------------



##def VLinM1(M2,M1,M0,LCV,VL):
##    print('-----------------VLinM1---------------')
##    print('')
##    
####  dans quelle chaine? -> on met la ville a la fin de la chaine
##    for CV in LCV:
##        if CV[-1]==VL:
##            CV1=CV
##            LCV.remove(CV)
##        elif CV[0]==VL:
##            CV.reverse()
##            CV1=CV
##            LCV.remove(CV)
##            
##    MCV1=M2+M1
##    vp=VP(MCV1,VL)
##    while vp in CV1:
##        MCV1.remove(vp)
##        vp=VP(MCV1,VL)
##    print('vp=',vp)
####  vp en bout de chaine?    
##    if vp in M1:         
####      dans quelle chaine? -> on met la ville au debut de la chaine
##        for CV in LCV:
##            if CV[0]==vp:
##                CV2=CV
##                LCV.remove(CV)
##            elif CV[-1]==vp:
##                CV.reverse()
##                CV2=CV
##                LCV.remove(CV)
##        
##
##        LCV.append(CV1+CV2)
##        M1.remove(vp)
##        M0.append(vp)
##        
####      vp pas affectée
##    else:
##        LCV.append(CV1+[vp])
##        M2.remove(vp)
##        M1.append(vp)
##        M0.append(VL)
##    print('--------------------------------------')
##    
##
##def VLinM2(M2,M1,M0,LCV,VL):
##    if VL in M2:
##        M2.remove(VL)
##    elif VL in M1:
##        M1.remove(VL)
##    print('---------------VLinM2-----------------')
##    vp=VP(M2+M1,VL)
####  vp en bout de chaine?
##    print('vp=',vp)
##    if vp in M1:
##        print('vpinM1')
####      dans quelle chaine? -> on ajoute VL a coté de vp en bout de la chaine
##        for i in range(len(LCV)):
##            print(LCV[i])
##            input()
##            if LCV[i][0]==vp:
##                LCV[i]=[VL]+LCV[i]
##            elif LCV[i][-1]==vp:
##                LCV[i]=LCV[i]+[VL]
##        print('LCV=',LCV)
##        M1.remove(vp)
##        M0.append(vp)
##        M1.append(VL)
##    else:
##        CV=[VL,vp]
##        LCV.append(CV)
##        M2.remove(vp)
##        M1.append(vp)
##        M1.append(VL)
##    print('--------------------------------------')
##        
##            
##    
##        
##
##
##
##    
##def Meth6(LV):
##    print('LV=',LV)
##    "villes pas affectées"
##    M2=LV[0:len(LV)]
##    "villes en bout de chaine"
##    M1=[]
##    "villes encadrées par deux villes"
##    M0=[]
##    LCV=[]
##    df=(0,0)
##    while len(M0)!=len(LV) and M2+M1!=[]:
##        print('M2=',M2)
##        print('M1=',M1)
##        print('M0=',M0)
##        print('')
##        print('LCV=',LCV)
##        input('')
##        DV=DVtri(M2+M1)
##        print('DV=',DV)
##        print('VL=',DV[-1][1])
##        print('')
##        VL=DV[-1][1]
##        
##        if len(M1)==2 and M2==[]:
##            M1=[]
##            M0+=M1
##        elif VL in M1:
##            M1.remove(VL)
##            if M1+M2!=[]:
##                VLinM1(M2,M1,M0,LCV,VL)
##                
##        else:
##            M2.remove(VL)
##            VLinM2(M2,M1,M0,LCV,VL)
##    L=LCV[0]
##    L=L[L.index([0,0]):len(L)]+L[0:L.index([0,0])]+[[0,0]]
##    print('L=',L)
##    return L
                
            
        
#________________________________methode7: cercle________________________________________________


# def DmaxVV(LV):
#     "renvoie la distance max de chaque ville par rap aux autres"
#     DV=[]
#     for i in range(len(LV)):
#         D=[]
#         for j in range(len(LV)):
#             D.append(((LV[i][0]-LV[j][0])**2+(LV[i][1]-LV[j][1])**2)**0.5)
#         DV.append((D[-1],LV[i]))
#     DV.sort()
#     LV=[DV[i][1] for i in range(len(DV))]
#     LV.reverse()
#     LV=LV[1:len(LV)]+LV[0:1]
#     return LV



# def VLD(L,Va):
#     "donne la ville la plus lointaine de Va et la distance qui les separent"
#     D=[]
#     for elt in L:
#         D.append(((elt[0]-Va[0])**2+(elt[1]-Va[1])**2)**0.5)
        
#     idmax=D.index(max(D))
#     return (L[idmax],max(D))       


# def V_d_V(LV):
#     "renvoie un couple composé des deux villes les plus lointaines de la liste LV"
#     LDV=[]
#     for V in LV:
#         (Vl,d)=VLD(LV,V)
#         LDV.append((d,(V,Vl)))
#     LDV.sort()
#     return LDV[-1][1]


# def CVM8(MDR,Va,Vl):
#     print('________________')
#     MDR=list(MDR)
#     CV=[]
    

#     while MDR!=[]:
#         rect=[(min(Va[0],Vl[0]),max(Va[0],Vl[0])),(min(Va[1],Vl[1]),max(Va[1],Vl[1]))]
        
#         print('rect',rect)
#         print('')
#         input(('MDR',MDR))
#         turtle.up()
#         turtle.goto(Va)
#         input('va')
#         turtle.goto(Vl)
#         input('vl')
#         turtle.goto((rect[0][0],rect[1][0]))
#         turtle.down()
#         turtle.goto((rect[0][0],rect[1][1]))
#         turtle.goto((rect[0][1],rect[1][1]))
#         turtle.goto((rect[0][1],rect[1][0]))
#         turtle.goto((rect[0][0],rect[1][0]))
#         turtle.up()
#         if len(MDR)==1:
#             CV+=MDR
#             MDR=[]

#         else:
#             print('rect',rect)
#             print('')
#             input(('MDR',MDR))
#             turtle.up()
#             turtle.goto(Va)
#             input('va')
#             turtle.goto(Vl)
#             input('vl')
#             turtle.goto((rect[0][0],rect[1][0]))
#             turtle.down()
#             turtle.goto((rect[0][0],rect[1][1]))
#             turtle.goto((rect[0][1],rect[1][1]))
#             turtle.goto((rect[0][1],rect[1][0]))
#             turtle.goto((rect[0][0],rect[1][0]))
#             turtle.up()

#             MDRbis=[] 
#             for V in MDR:
#                 turtle.goto(V)
#                 if rect[0][0]<V[0]<rect[0][1] and rect[1][0]<V[1]<rect[1][1]:
#                     MDRbis.append(V)
#             MDRbis.append(Vl)
#             Vabis=VP(MDRbis,Va)
#             CV+=CVM8(MDRbis,Vabis,Vl)
#             CV.insert(0,Va)
            
#             print('CV',CV)
#             input(('MDR',MDR))
            
#             for V in CV:
#                 if V in MDR:
#                     MDR.remove(V)
                
#             input(('MDR',MDR))
            
#             turtle.color('green')
#             for V in CV:
#                 turtle.goto(V)
#                 turtle.down()
#             turtle.up()
#             turtle.color('black')
#     input('----------------')    
#     return CV
    
            
        

# ##def m8(LV):
# ##    print('LV=',LV)
# ##    
# ##    (Va,Vl)=V_d_V(LV) #les deux villes les plus lointaines (e) elles de LV en prennant Va pr 1ere ville
# ##    
# ##    print('va,vl',Va,Vl)
# ##    LV=LV[LV.index(Vl):len(LV)]+LV[0:LV.index(Vl)] #on reordonne LV pour mettre Vl en 1er
# ##    LV.remove(Va)
# ##    LV.append(Va)
# ##    
# ##    turtle.speed('fastest')
# ##    print('LV=',LV)
# ##    turtle.setup(800,800,0,0)
# ##    turtle.up()
# ##    for V in LV :
# ##        turtle.goto(V)
# ##        turtle.dot()
# ##        
# ##    L=[]
# ##    M=LV
# ##
# ##    print('M',M)
# ##    input()
# ##    LCV=[] #Liste de Chaine de Villes
# ##    while M!=[]:
# ##        #tant qu'il manque des villes dans le trajet
# ##        if LCV:
# ##            Va=LCV[-1][-1]
# ##        
# ####        (Vl,R)=VLD(M,Va)
# ##        rect=[(min(Va[0],Vl[0]),max(Va[0],Vl[0])),(min(Va[1],Vl[1]),max(Va[1],Vl[1]))]
# ##        MDR=[]
# ##        for V in M:
# ##            turtle.goto(V)
# ##            if rect[0][0]<V[0]<rect[0][1] and rect[1][0]<V[1]<rect[1][1]:
# ##                MDR.append(V)
# ##        MDR.append(Vl)
# ##            
# ##        CV=CVM8(MDR,Va,Vl)        
# ##        LCV.append(CV)
# ##        for V in CV:
# ##            if V!=Va:
# ##                M.remove(V)
# ##        
# ##        print('M',M)
# ##        input(('LCV',LCV))
# ##        
# ##        turtle.color('red')
# ##        for V in L:
# ##            turtle.goto(V)
# ##            turtle.down()
# ##        turtle.up()
# ##        turtle.color('black')
# ##        
# ##    
# ##    input(('L:',L))
# ##    turtle.color('red')
# ##    for V in L:
# ##        turtle.goto(V)
# ##        turtle.down()
# ##    turtle.up()
# ##    print('----------')
# ##    input('fini????')
# ##    turtle.bye()
# ##    
# ##    return LCV


# ##def CVcercle(M):
# ##    input('____________')
# ##    print('Mbis')
# ##    for V in M :
# ##        turtle.goto(V)
# ##        
# ##    CV=[]
# ##    M=list(M)
# ##    if len(M)==1:
# ##        print('-------------')
# ##        return M
# ##    
# ##    Va=M[-1]
# ##    (Vl,R)=VLD(M,Va)
# ##    I=((Va[0]+Vl[0])/2,(Va[1]+Vl[1])/2)
# ##    R=R/2
# ##    
# ##    turtle.goto(Va)
# ##    input('Va')
# ##    turtle.goto(Vl)
# ##    input('Vl')
# ##    turtle.goto((I[0],I[1]-R))
# ##    turtle.down()
# ##    turtle.circle(R)
# ##    turtle.up()
# ##    input()
# ##    
# ##    Mbis=[]
# ##    for V in M:
# ##        if dtot([I,V])<=R and V!=Va and V!=Vl:
# ##            Mbis.append(V)
# ##    Mbis.append(Vl)
# ##    
# ##    CV=CVcercle(Mbis)
# ##    CV.reverse()
# ##    CV=CV+[Va]
# ##    
# ##    print('cv',CV)
# ##    for V in CV :
# ##        turtle.goto(V)
# ##        turtle.down()
# ##    turtle.up()
# ##    print('------')
# ##    input()
# ##    
# ##    return CV




# def m7(LV):
#     LV.sort()
    
#     print('LV=',LV)
    
#     (Va,Vl)=V_d_V(LV)
#     print(Va,Vl)
#     LV=LV[LV.index(Vl):len(LV)]+LV[0:LV.index(Vl)]
#     LV.remove(Va)
#     LV.append(Va)
    
#     turtle.speed('fastest')
#     print('LV=',LV)
#     turtle.setup(800,800,0,0)
#     turtle.up()
#     for V in LV :
#         turtle.goto(V)
#         turtle.dot()
        
#     L=LV[0:1]
#     M=LV[1:len(LV)]

#     print('M',M)
#     input()
    
#     LCV=[]
#     while M!=[]:
        
#         input('____________')

#         Va=L[-1]
#         (Vl,R)=VLD(M,Va)
#         I=((Va[0]+Vl[0])/2,(Va[1]+Vl[1])/2)
#         R=R/2
        
#         turtle.goto(Va)
#         input('Va')
#         turtle.goto(Vl)
#         input('Vl')
#         turtle.goto((I[0],I[1]-R))
#         turtle.down()
#         turtle.circle(R)
#         turtle.up()
#         input()
        
#         MDC=[]
#         MEC=[]
#         for V in M:
#             if dtot([I,V])<=R:
#                 if V!=Va:
#                     MDC.append(V)
#             else:
#                 MEC.append(V)

#         if Va in MEC:
#             input('VA IN MEC PB')
                
#         MDC.append()
#         CV=CVcercle(MEC,MDC,Vl)
        
#         print('CV',CV)
#         print('M',M)
        
#         for V in CV:
#             M.remove(V)
#         L+=CV

#         print('M',M)
#         input(('L',L))
#         turtle.color('red')
#         for V in L:
#             turtle.goto(V)
#             turtle.down()
#         turtle.up()
#         turtle.color('black')
        
#     L.append(L[0])
    
#     input(('L:',L))
#     turtle.color('red')
#     for V in L:
#         turtle.goto(V)
#         turtle.down()
#     turtle.up()
#     print('----------')
#     input('fini????')
#     turtle.bye()
    
#     return L


        
# def CVcercle(MEC,MDC,Va):
#     input('____________')
#     print('MDC',MDC,len(MDC))
#     for V in MsVa :
#         turtle.goto(V)
        
#     MDC=list(MDC)    
#     CV=[]
#     tour=-1
#     while MDC!=[]:
#         print('MDC',MDC, len(MDC))
#         if len(MDC)==1:
#             CV+=MDC
#             MDC=[]
#         else:
#             if CV:
#                 Va=CV[-1]
            
#             (Vl,R)=VLD(MDC,Va)
#             I=((Va[0]+Vl[0])/2,(Va[1]+Vl[1])/2)
#             R=R/2
                
#             turtle.goto(Va)
#             input('Va')
#             turtle.goto(Vl)
#             input('Vl')
#             turtle.goto((I[0],I[1]-R))
#             turtle.down()
#             turtle.circle(R)
#             turtle.up()
#             input()
            
#             MECbis=[]
#             MDCbis=[]
#             for V in MEC+MDC:
#                 if dtot([I,V])<=R:
#                     if V!=Va:
#                         MDCbis.append(V)
#                     else:
#                         MECbis.append(V)
                        
#             if Va in MECbis:
#                 input('VA IN MEC PB')
            
#             CV=CVcercle(MEC,MDCbis,Vl)
#             CV.reverse()
#             CV=CV+[Va]
#             for V in CV:
#                 if V in MDC:
#                     MDC.remove(V)
            
#             print('cv',CV)
#             for V in CV :
#                 turtle.goto(V)
#                 turtle.down()
#             turtle.up()
#             input()
        
#     print('------')
#     input()
    
#     return CV    
    
    
            


# #________________________resultat/moyenne/graph/comp________________________________________________

        
# def graph_spe(LL):
#     "fait le chemins de toutes les listes contenues dans LL en mm tps"
#     D=[dtot(L) for L in LL]
#     turtle.setup(700,700,0,0)
#     LT=[]
#     for n in range(len(LL)):
#         LT.append(turtle.Turtle())
#     for i in range(len(LT)):
#         LT[i].color(random.random(),random.random(),random.random())
#     for i in range(len(LL[0])):
#         for iT in range(len(LT)):
#             LT[iT].dot(5,'red')
#             LT[iT].goto(LL[iT][i])
#     turtle.title((D))   

 

# def graph(L):
#     d=0
#     turtle.setup(700,700,0,0)
#     turtle.color(random.random(),random.random(),random.random())
#     turtle.up()
#     for elt in L:
#         turtle.goto(elt)
#         turtle.down()
#         turtle.dot(5,'red')
#         d+=turtle.distance(tuple(elt))
#         turtle.title(('distance totale:',d))



# def comp(n,TR,*methodes):
#     LV=crea(n)
#     L=[]
#     T=[]
#     methodes=list([*methodes])
#     for M in methodes:
#         t=time.perf_counter()
#         L.append(M(LV))
#         T.append(time.perf_counter()-t)
#     D=[dtot(M) for M in L]
#     if TR==1:
#         print('T:',T)
#         print('D:',D)
#         turtle.setup(700,700)
#         T=turtle.Turtle()
#         for rep in range(len(L)):
#             T.color(random.random(),random.random(),random.random())
#             T.up()
#             for i in range(len(L[0])):
#                 T.goto(L[rep][i])
#                 T.dot(5,0,0,0)
#                 T.down()
#                 turtle.title(D)
#         input('r')
#         turtle.bye()
#     else:
#         return D




# def moycomp(N,n,*methodes):
#     "moy/N,n villes, mettre que 2 methodes"
#     T=time.perf_counter()
#     methodes=[*methodes]
#     if len(methodes)==2:
#         Moy=[0]*3
#     else:
#         Moy=[0]*len(methodes)
#     for rep in range(N):
#         R=comp(n,0,*methodes)
#         t=time.perf_counter()
#         if t-T>10:
#             T=t
#             print(rep)
#             print([' '+str(meth)[10:12]+' ' for meth in methodes])
#             print([float(str(elt/rep)[0:6]) for elt in Moy])
#             print('')
#         dmini=min(R)
#         if len(methodes)==2:
#             if R[0]==R[1]:
#                 Moy[2]+=1
#             else:
#                 for i in range(len(R)):
#                     if R[i]==dmini:
#                         Moy[i]+=1
#         else:
#             for i in range(len(R)):
#                 if R[i]==dmini:
#                     Moy[i]+=1
        
#     return [elt/N for elt in Moy]
        

# def comptracers(LV,methode):
#     methode
#     nb=0
#     if True:
#         vitesse=input('vitesse: ')
        
#         while True:
#             turtle.speed(vitesse)
#             nb+=1
#             L=methode(LV)
#             print('dm',dtot(L))
#             print('nb',nb)
#             graph(L)
#             input()
#             turtle.clearscreen()
#             LV=crea(len(LV))
# ##    else:
# ##        while True:
# ##            nb+=1
# ##            L0=m0(LV)
# ##            L=methode(LV)
# ##            if dtot(L)!=dtot(L0):
# ##                if dtot(L)>dtot(L0):
# ##                    print('dm,dm0',dtot(L),dtot(L0))
# ##                    print('nb',nb)
# ##                    gr=input('graph?')
# ##                    if gr=='':
# ##                        print('LV=',LV)
# ##                        graph(L)
# ##                        graph(L0)
# ##                        input()
# ##                        turtle.clearscreen()
# ##
# ##                else:
# ##                    print('nb',nb)
# ##                    print('LV=',LV)
# ##                    turtle.title('WTF')
# ##                    input()
# ##                    graph(methode(LV))
# ##                    graph(m0(LV))
# ##                    
# ##                    input()
# ##                    turtle.bye()
# ##            LV=crea(len(LV))

# %%
