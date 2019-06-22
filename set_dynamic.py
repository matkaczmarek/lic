import networkx as nx
import sys
from itertools import chain, combinations


def powerset(X, fr, to):
    S = list(X)
    return chain.from_iterable(combinations(S, r) for r in range(fr, to))


def subsets(X, fr, to):
    return list(map(frozenset, powerset(X, fr, to)))


def make_terminals_degree_one(G, K):
    for x in K:
        p = len(G.nodes)
        G.add_node(p)
        e_add, e_remove = [], []

        for y in G.neighbors(x):
            e_remove.append((x, y))
            e_add.append((p, y))

        G.remove_edges_from(e_remove)
        G.add_edges_from(e_add)
        G.add_edge(x, p)


def set_dynamic(G: nx.Graph, K: list):
    G = nx.convert_node_labels_to_integers(G, label_attribute='old')
    K = [i for i in G.nodes if G.node[i]['old'] in K]

    make_terminals_degree_one(G, K)

    dist = dict(nx.shortest_path_length(G))
    G_K = [x for x in G.nodes if x not in K]

    T = dict()

    for t in K:
        for v in G_K:
            T[(frozenset([t]), v)] = dist[t][v]

    for size in range(2, len(K) + 1):
        sub = subsets(K, size, size + 1)

        for D in sub:
            for v in G_K:
                sub_D = subsets(D, 1, len(D))
                mini = sys.maxsize

                for u in G_K:
                    for D_prim in sub_D:
                        mini = min(mini,
                                   T[(D_prim, u)] + T[(frozenset([i for i in D if i not in D_prim]), u)] + dist[v][u])

                T[(D, v)] = mini

    v = min(G_K, key=lambda v: T[(frozenset(K), v)])
    return T[(frozenset(K), v)] - len(K)
