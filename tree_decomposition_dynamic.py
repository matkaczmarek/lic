import sys

import networkx as nx
from generate_test import tree_decomposition

LEAF_NODE, INTRODUCE_VERTEX_NODE, INTRODUCE_EDGE_NODE = "Leaf node", "Introduce vertex node", "Introduce edge node"
FORGET_NODE, JOIN_NODE = "Forget node", "Join node"


def partition(collection):
    if len(collection) == 1:
        yield frozenset([collection])
        return

    first = collection[0]
    for smaller in partition(collection[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1:]

        yield frozenset([[first]] + smaller)


def relabel(T):
    mapping = {}
    i = 0
    for x in T.nodes:
        mapping[i] = x
        i += 1

    return nx.relabel_nodes(T, {v: k for k, v in mapping.items()}), mapping


def memoisation(t, C, X, P, mapping, labels):
    if (t, X, P) in C.keys():
        return C[(t, X, P)]

    if labels[t][0] == INTRODUCE_VERTEX_NODE:
        t_prim = t.neighbours()[0]
        # labels[t] is tuple (name, v)
        v = labels[t][1]
        if v in X and frozenset(v) in P:
            C[(t, X, P)] = memoisation(t_prim, C, X.remove(v), P.remove(frozenset(v)), mapping, labels)
        else:
            C[(t, X, P)] = memoisation(t_prim, C, X, P, mapping, labels)

    elif labels[t][0] == INTRODUCE_EDGE_NODE:
        t_prim = t.neighbours()[0]
        minimum = sys.maxsize
        for P_prim in partition(X):
            if minimum > memoisation(t_prim, C, X, P_prim, mapping, labels):
                minimum = memoisation(t_prim, C, X, P_prim, mapping, labels)

        C[(t, X, P)] = min(minimum, memoisation(t_prim, C, X, P, mapping, labels))

    elif labels[t][0] == FORGET_NODE:
        t_prim = t.neighbours()[0]
        w = labels[t][1]

        minimum = sys.maxsize
        for P_prim in partition(X):
            if minimum > memoisation(t_prim, C, X.union(w), P_prim, mapping, labels):
                minimum = memoisation(t_prim, C, X.union(w), P_prim, mapping, labels)

        C[(t, X, P)] = min(minimum, memoisation(t_prim, C, X, P, mapping, labels))
    elif labels[t][0] == JOIN_NODE:
        t1 = labels[t][1][0]
        t2 = labels[t][1][1]
        #TODO acyclic merge
    elif labels[t][0] == LEAF_NODE:
        if X == frozenset([]):
            C[(t, X, P)] = sys.maxsize
        else:
            C[(t, X, P)] = 0

    return C[(t, X, P)]


def tree_decomp_dynamic(G, T, root, labels, K, l):
    u = K[0]
    T, mapping = relabel(T)

    # verticies in T are frozensets
    # make T directed
    # add u to every bag in T

    edges_to_add = []
    for x in T.nodes:
        if T.degree[x] == 1:
            # mapping is map
            if u not in mapping[x]:
                new_node = len(mapping)
                labels[new_node] = (LEAF_NODE, u)

                # len(mapping[x]) == 1
                labels[x] = (INTRODUCE_VERTEX_NODE, mapping[x][0])

                edges_to_add.append((new_node, x))
                mapping[new_node] = frozenset([u])

        mapping[x] = mapping[x].union(mapping[x], frozenset([u]))

    T.add_edges_from(edges_to_add)

    C = {}

    return memoisation(root, C, frozenset(u), frozenset([frozenset(u)]), mapping, labels)
