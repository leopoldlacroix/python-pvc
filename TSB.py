import plotly.graph_objects as go
import time

import numpy as np
import pandas as pd
import plotly.express as px
from itertools import permutations

from hilbertcurve.hilbertcurve import HilbertCurve



def dtot(cities : pd.DataFrame):
    "Distance tot d'un parcours formule: sqrt(somme((Xi-X'i)**2)). \n referme le chemin!!"
    circuit = cities[["x", "y"]]
    circuit = pd.concat([circuit,circuit[0:1]])
    d = np.sqrt((circuit[["x", "y"]].diff()**2).sum(axis=1)).sum()
    return d

def crea(n:int):
    "créer n villes"
    return pd.DataFrame(
        np.random.random(size=(n,2)),
        columns=['x', 'y']
    )

def plot_path(path: pd.DataFrame, elapsed = 0, strategy_name = ''):

    score = dtot(path)
    circuit = pd.concat([path, path[0:1]]).reset_index(names=["id"])
    title = f'{strategy_name} Score: {round(score, 4)} \t elapsed {round(elapsed, 4)}'
    title += '' if path.shape[0]>8 else "\t"+"".join(path.index.astype(str))
    fig = px.line(
        data_frame = circuit, 
        x="x", y="y", text="id",
          title=title)
    fig.update_traces(textposition='top left')
    return fig

def test_strategy(strategy, n=6):
    cities=crea(n)
    start = time.time()
    path, *debug = strategy(cities)
    end = time.time()

    elapsed = end - start
    fig = plot_path(path, elapsed, strategy.__name__).show()
    return fig, debug

def test_strategies(strategies: list):
    for i in range(1, len(strategies)):
        test_strategy(strategies[i])

    # datas = test_strategy(strategies[0])[0].data
    # for i in range(1, len(strategies)):
    #     datas += test_strategy(strategies[i])[0].data
    # fig = go.Figure(data = datas)
    # fig.show()



cities6 = crea(6)
cities10 = crea(10)
cities18 = crea(18)