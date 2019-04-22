import operator
import sys
from itertools import chain, combinations

from graph import Graph

G = Graph([[1], [0, 2, 3], [1, 4], [1, 4], [2, 3, 5], [4]])
K = [0, 2, 5]


def powerset(X, fr, to):
    S = list(X)
    return chain.from_iterable(combinations(S, r) for r in range(fr, to))


def subsets(X, fr, to):
    return list(map(list, powerset(X, fr, to)))


def dijkstra(G, s, w):
    Q = []
    dist = {}
    for v in G.verticies():
        dist[v] = sys.maxsize
        Q.append(v)
    dist[s] = 0

    while len(Q) > 0:
        u = min(Q, key=lambda x: dist[x])
        Q.remove(u)

        for v in G.neighbours(u):
            alt = dist[u] + w[(u, v)]
            if alt < dist[v]:
                dist[v] = alt

    return dist


def set_dynamic(G, K, w=None):
    if w is None:
        w = {}
        for v in G.verticies():
            for u in G.neighbours(v):
                w[(v, u)] = 1

    new_K = []
    s_G = len(G.verticies())
    for k in K:
        t = s_G + len(new_K)
        new_K.append(t)
        G.add_vert(t, [k])
        w[(t, k)] = w[(k, t)] = 1

    K = new_K

    dist = {}
    for v in G.verticies():
        dist[v] = dijkstra(G, v, w)

    G_K = [i for i in G.verticies() if i not in K]
    T = {}

    for t in K:
        for v in G_K:
            T[(frozenset([t]), v)] = dist[t][v]

    for size in range(2, len(K) + 1):
        sub = subsets(K, size, size + 1)

        for D in sub:
            D = frozenset(D)
            for v in G_K:
                sub_D = subsets(D, 1, len(D))
                mini = sys.maxsize
                for u in G_K:
                    for D_prim in sub_D:
                        D_prim = frozenset(D_prim)
                        alt = T[(D_prim, u)] + T[(frozenset([i for i in D if i not in D_prim]), u)] + dist[v][u]
                        if alt < mini:
                            mini = alt
                T[(D, v)] = mini

    v = min(G_K, key=lambda v: T[(frozenset(K), v)])
    return T[(frozenset(K), v)] - len(K)
