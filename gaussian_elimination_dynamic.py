import sys
import networkx as nx
import python_algorithms

LEAF_NODE, INTRODUCE_VERTEX_NODE, INTRODUCE_EDGE_NODE = "Leaf node", "Introduce vertex node", "Introduce edge node"
FORGET_NODE, JOIN_NODE = "Forget node", "Join node"
bag = 'bag'


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
            break
        if frozenset([u]).issubset(p_i):
            p_u = p_i
            continue
        if frozenset([v]).issubset(p_i):
            p_v = p_i
            continue
        new_p.append(p_i)

    new_p.append(p_u.union(p_v))
    return frozenset(new_p)


def memoization(t, C, T, X, labels, G, K):
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

                p = glue(u, v, p)
                if p in rmc:
                    rmc[p] = min(rmc[p], w + 1)
                else:
                    rmc[p] = w + 1
            C[(t, X)] = set([(k, rmc[k]) for k in rmc])

    elif labels[t][0] == FORGET_NODE:
        y = [i for i in T.neighbors(t)][0]

        # labels[t] is tuple (name, v)
        v = labels[t][1]

        rmc = {}
        for p, w in memoization(y, C, T, X.difference([v]), labels, G, K):
            if p in rmc:
                rmc[p] = min(rmc[p], w)
            else:
                rmc[p] = w

        for p, w in memoization(y, C, T, X.union([v]), labels, G, K):
            if frozenset([v]) in p:
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

    return C[(t, X)]


def gaussian_elim_dynamic(T, root, labels, K, G):
    y = [i for i in T.neighbors(root)][0]

    C = {}
    memoization(root, C, T, frozenset([]), labels, G, K)
    return [w for _, w in C[(0, frozenset({0}))]][0]



T = {27: [0]
    , 0: [1]
    , 1: [2]
    , 2: [3]
    , 3: [4]
    , 4: [5]
    , 5: [6]
    , 6: [7]
    , 7: [8]
    , 8: [9]
    , 9: [10]
    , 10: [11, 19]
    , 11: [12]
    , 12: [13]
    , 13: [14]
    , 14: [15]
    , 15: [16]
    , 16: [17]
    , 17: [18]
    , 18: [29]
    , 29: []
    , 19: [20]
    , 20: [21]
    , 21: [22]
    , 22: [23]
    , 23: [24]
    , 24: [25]
    , 25: [26]
    , 26: [28]
    , 28: [] }

bags = {27: frozenset([])
    , 0: frozenset([0])
    , 1: frozenset([0, 1])
    , 2: frozenset([0, 1])
    , 3: frozenset([1])
    , 4: frozenset([1, 2])
    , 5: frozenset([1, 2])
    , 6: frozenset([3, 1, 2])
    , 7: frozenset([3, 1, 2])
    , 8: frozenset([1, 2, 3, 5])
    , 9: frozenset([1, 2, 3, 5])
    , 10: frozenset([1, 2, 3, 5])
    , 11: frozenset([1, 2, 3, 5])
    , 12: frozenset([2, 3, 5])
    , 13: frozenset([2, 5])
    , 14: frozenset([2, 4, 5])
    , 15: frozenset([2, 4, 5])
    , 16: frozenset([4, 5])
    , 17: frozenset([4, 5])
    , 18: frozenset([5])
    , 29: frozenset([])
    , 19: frozenset([1, 2, 3, 5])
    , 20: frozenset([2, 3, 5])
    , 21: frozenset([3, 5])
    , 22: frozenset([3, 5, 6])
    , 23: frozenset([3, 5, 6])
    , 24: frozenset([5, 6])
    , 25: frozenset([5, 6])
    , 26: frozenset([6])
    , 28: frozenset([])    }

labels = {27: (FORGET_NODE, 0)
    , 0: (FORGET_NODE, 1)
    , 1: (INTRODUCE_EDGE_NODE, (0, 1))
    , 2: (INTRODUCE_VERTEX_NODE, 0)
    , 3: (FORGET_NODE, 2)
    , 4: (INTRODUCE_EDGE_NODE, (1, 2))
    , 5: (FORGET_NODE, 3)
    , 6: (INTRODUCE_EDGE_NODE, (1, 3))
    , 7: (FORGET_NODE, 5)
    , 8: (INTRODUCE_EDGE_NODE, (2, 5))
    , 9: (INTRODUCE_EDGE_NODE, (3, 5))
    , 10: (JOIN_NODE, 0)
    , 11: (INTRODUCE_VERTEX_NODE, 1)
    , 12: (INTRODUCE_VERTEX_NODE, 3)
    , 13: (FORGET_NODE, 4)
    , 14: (INTRODUCE_EDGE_NODE, (2, 4))
    , 15: (INTRODUCE_VERTEX_NODE, 2)
    , 16: (INTRODUCE_EDGE_NODE, (4, 5))
    , 17: (INTRODUCE_VERTEX_NODE, 4)
    , 18: (INTRODUCE_VERTEX_NODE, 5)
    , 29: (LEAF_NODE, 0)
    , 19: (INTRODUCE_VERTEX_NODE, 1)
    , 20: (INTRODUCE_VERTEX_NODE, 2)
    , 21: (FORGET_NODE, 6)
    , 22: (INTRODUCE_EDGE_NODE, (3, 6))
    , 23: (INTRODUCE_VERTEX_NODE, 3)
    , 24: (INTRODUCE_EDGE_NODE, (5, 6))
    , 25: (INTRODUCE_VERTEX_NODE, 5)
    , 26: (INTRODUCE_VERTEX_NODE, 6)
    , 28: (LEAF_NODE, 0)
          }
edges_to_add = []
Tree = nx.DiGraph()
for x in T.keys():
    Tree.add_node(x, bag=bags[x])
    # print(x, Tree.node[x][bag])
    for y in T[x]:
        edges_to_add.append((x, y))
Tree.add_edges_from(edges_to_add)
K = [0, 4, 5, 6]

print(gaussian_elim_dynamic(Tree, 27, labels, K, T))
