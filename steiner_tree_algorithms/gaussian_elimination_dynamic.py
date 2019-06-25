import itertools
import sys
import networkx as nx
import python_algorithms

LEAF_NODE, INTRODUCE_VERTEX_NODE, INTRODUCE_EDGE_NODE = "Leaf node", "Introduce vertex node", "Introduce edge node"
FORGET_NODE, JOIN_NODE = "Forget node", "Join node"
bag = 'bag'

def build_subgraph_Gf(T: nx.DiGraph, v: int, nodes: set, edges: set, labels: dict):

    for x in T.node[v][bag]:
        nodes.add(x)

    if labels[v][0] == INTRODUCE_EDGE_NODE:
        edges.add(labels[v][1])

    for u in T.neighbors(v):
        build_subgraph_Gf(T, u, nodes, edges, labels)

def join(p1, p2):
    new_P = []
    Temp = nx.MultiGraph()
    for P_i in p1:
        Temp.add_path([i for i in P_i])

    for P_i in p2:
        Temp.add_path([i for i in P_i])

    for c in nx.connected_components(Temp):
        new_P.append(frozenset(c))

    return frozenset(new_P)


def glue(u, v, p):
    new_p = []  # making new partition with u, v glued

    for p_i in p:
        if frozenset([u, v]).issubset(p_i):
            p_u = p_v = p_i
            continue
        if frozenset([u]).issubset(p_i):
            p_u = p_i
            continue
        if frozenset([v]).issubset(p_i):
            p_v = p_i
            continue
        new_p.append(p_i)

    new_p.append(p_u.union(p_v))
    return frozenset(new_p)


def cuts(U, v):
    out = []
    for i in range(1, len(U) + 1):
        out.extend(
            [(frozenset(part), frozenset(U.difference(part))) for part in itertools.combinations(U, i) if v in part])
    return out


def reduce(A, U):
    if len(A) <= 2 ** len(U):
        return A

    #sort matrix by weight
    A = sorted(A, key=lambda x: x[1])
    v = next(iter(U))

    cut_matrix = {}
    all_cuts = cuts(U, v)

    for p, _ in A:
        for cut in all_cuts:
            cut_matrix[(p, cut)] = int(all((p_i.issubset(cut[0]) or p_i.issubset(cut[1]) for p_i in p)))

    #out will be our basis for cut_matrix
    out = []
    for i in range(len(A)):
        if len(out) == 2 ** len(U):
            break

        has_one = False
        p = A[i][0]
        for cut in all_cuts:
            if cut_matrix[(p, cut)] == 1:
                has_one = True
                for j in range(i + 1, len(A)):
                    q = A[j][0]
                    if cut_matrix[(q, cut)] == 0:
                        continue
                    for cut_prim in all_cuts:
                        cut_matrix[(q, cut_prim)] += cut_matrix[(p, cut_prim)]
                        cut_matrix[(q, cut_prim)] %= 2
                break

        if has_one:
            out.append(A[i])

    return set(out)


def memoization(t: int, C: dict, T: nx.DiGraph, X: frozenset, labels: dict, G: nx.Graph, K: list):
    if (t, X) in C.keys():
        return C[(t, X)]

    if labels[t][0] == INTRODUCE_VERTEX_NODE:
        y = [i for i in T.neighbors(t)][0]

        # labels[t] is tuple (name, v)
        v = labels[t][1]
        if v in X:
            C[(t, X)] = {(p.union(frozenset([frozenset([v])])), w) for p, w in
                         memoization(y, C, T, X.difference([v]), labels, G, K)}  # add singleton {v} to p
        elif v not in X and v not in K:
            C[(t, X)] = memoization(y, C, T, X, labels, G, K)
        else:  # v not in X but v in K
            return {}

    elif labels[t][0] == INTRODUCE_EDGE_NODE:
        y = [i for i in T.neighbors(t)][0]
        # labels[t] is tuple (name, (u, v))
        u, v = labels[t][1]

        if u not in X or v not in X:
            C[(t, X)] = memoization(y, C, T, X, labels, G, K)
        else:
            rmc = {}
            for p, w in memoization(y, C, T, X, labels, G, K):
                if p in rmc:
                    rmc[p] = min(rmc[p], w)
                else:
                    rmc[p] = w

            for p, w in memoization(y, C, T, X, labels, G, K):
                if all(not frozenset([v]).issubset(p_i) for p_i in p):
                    p = p.union(frozenset([frozenset([v])]))

                if all(not frozenset([u]).issubset(p_i) for p_i in p):
                    p = p.union(frozenset([frozenset([u])]))

                p_prim = glue(u, v, p)

                if p_prim in rmc:
                    rmc[p_prim] = min(rmc[p_prim], w + 1)
                else:
                    rmc[p_prim] = w + 1
            C[(t, X)] = set([(k, rmc[k]) for k in rmc])

    elif labels[t][0] == FORGET_NODE:
        y = [i for i in T.neighbors(t)][0]

        # labels[t] is tuple (name, v)
        v = labels[t][1]

        rmc = {}
        if v not in K:
            for p, w in memoization(y, C, T, X.difference([v]), labels, G, K):
                if p in rmc:
                    rmc[p] = min(rmc[p], w)
                else:
                    rmc[p] = w

        for p, w in memoization(y, C, T, X.union([v]), labels, G, K):
            if frozenset([v]) in p:
                continue

            if all(not frozenset([v]).issubset(p_i) for p_i in p):
                continue

            new_p = []
            for p_i in p:
                new_p.append(p_i.difference([v]))

            p = frozenset(new_p)

            if p in rmc:
                rmc[p] = min(rmc[p], w)
            else:
                rmc[p] = w

        C[(t, X)] = set([(k, rmc[k]) for k in rmc])

    elif labels[t][0] == JOIN_NODE:
        y = [i for i in T.neighbors(t)][0]
        z = [i for i in T.neighbors(t)][1]

        rmc = {}
        for p1, w1 in memoization(y, C, T, X, labels, G, K):
            for p2, w2 in memoization(z, C, T, X, labels, G, K):
                p_prim = join(p1, p2)
                if p_prim in rmc:
                    rmc[p_prim] = min(rmc[p_prim], w1 + w2)
                else:
                    rmc[p_prim] = w1 + w2

        C[(t, X)] = set([(k, rmc[k]) for k in rmc])

    elif labels[t][0] == LEAF_NODE:
        return {(frozenset([]), 0)}

    C[(t, X)] = reduce(C[(t, X)], X)

    return C[(t, X)]


def gaussian_elim_dynamic(T: nx.DiGraph, root: int, labels: dict, K: list, G: nx.Graph):
    y = [i for i in T.neighbors(root)][0]

    C = {}
    memoization(y, C, T, T.node[y][bag], labels, G, K)

    return [w for _, w in C[ (y, T.node[y][bag]) ]][0]
