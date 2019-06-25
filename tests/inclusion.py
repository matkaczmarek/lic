from itertools import chain, combinations


def powerset(X):
    S = list(X)
    return chain.from_iterable(combinations(S, r) for r in range(len(S) + 1))


def subsets(X):
    return list(map(list, powerset(X)))


def dynamic(G, X, s, l):
    if s in X:
        return 0

    G_prim = G.subgraph_without(X)

    b = {}
    for a in G_prim.verticies():
        b[a] = [] + [1] * (l + 1)

    for j in range(1, l + 1):
        for a in G_prim.verticies():
            for t in G_prim.neighbours(a):
                for j2 in range(j):
                    b[a][j] += b[a][j - 1 - j2] * b[t][j2]

    return b[s][l]


def count_steiner_tree(G, K, l):

    if G.edge_number() < l:
        l = G.edge_number()

    s = K[0]
    out = 0
    for X in subsets(K):
        out += (-1) ** len(X) * dynamic(G, X, s, l)
    return out != 0
