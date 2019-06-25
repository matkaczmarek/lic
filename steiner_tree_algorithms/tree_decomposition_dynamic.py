import copy
import sys
import networkx as nx
import python_algorithms

LEAF_NODE, INTRODUCE_VERTEX_NODE, INTRODUCE_EDGE_NODE = "Leaf node", "Introduce vertex node", "Introduce edge node"
FORGET_NODE, JOIN_NODE = "Forget node", "Join node"
bag = 'bag'


def partition_bis(collection):
    collection = list(collection)
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in partition_bis(collection[1:]):

        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1:]

        yield [[first]] + smaller


def partition(collection):
    out = []
    for x in partition_bis(collection):
        out.append(frozenset([frozenset(i) for i in x]))

    return frozenset(out)


def build_subgraph_Gf(T: nx.DiGraph, v: int, nodes: set, edges: set, labels: dict):

    for x in T.node[v][bag]:
        nodes.add(x)

    if labels[v][0] == INTRODUCE_EDGE_NODE:
        edges.add(labels[v][1])

    for u in T.neighbors(v):
        build_subgraph_Gf(T, u, nodes, edges, labels)


def is_acyclic_merge(P1, P2, P):
    new_P = []
    Temp = nx.MultiGraph()
    for P_i in P1:
        Temp.add_path([i for i in P_i])

    for P_i in P2:
        Temp.add_path([i for i in P_i])

    for c in nx.connected_components(Temp):
        new_P.append(frozenset(c))

    if frozenset(new_P).issubset(P) and P.issubset(frozenset(new_P)):
        try:
            nx.find_cycle(Temp, orientation='ignore')
            return False
        except:
            return True

    return False


def memoization(t: int, T: nx.DiGraph, C: dict, X: frozenset, P: frozenset, labels: dict, G: nx.Graph, K: list):
    if (t, X, P) in C.keys():
        return C[(t, X, P)]

    if not V_t.intersection(K).issubset(V_F):
         C[(t, X, P)] = sys.maxsize
         return C[(t, X, P)]

    if labels[t][0] == INTRODUCE_VERTEX_NODE:
        t_prim = [i for i in T.neighbors(t)][0]

        # labels[t] is tuple (name, v)
        v = labels[t][1]
        if (v in X and frozenset([v]) not in P) or (v in K and v not in X):
            C[(t, X, P)] = sys.maxsize
        elif v in X and frozenset([v]) in P:
            C[(t, X, P)] = memoization(t_prim, T, C, X.difference([v])
                                       , frozenset([x for x in P if x != frozenset([v])])
                                       , labels, G, K)
        else:
            C[(t, X, P)] = memoization(t_prim, T, C, X, P, labels, G, K)

    elif labels[t][0] == INTRODUCE_EDGE_NODE:
        t_prim = [i for i in T.neighbors(t)][0]
        # labels[t] is tuple (name, (u, v))
        u, v = labels[t][1]

        # if for all (u, v) not in P_i then u, v are in separate P_i
        if u not in X or v not in X or all(not frozenset([u, v]).issubset(P_i) for P_i in P):
            C[(t, X, P)] = memoization(t_prim, T, C, X, P, labels, G, K)
        else:
            minimum = sys.maxsize
            for P_i in P:
                # (u, v) is not in separate P_i
                if not frozenset([u, v]).issubset(P_i):
                    continue

                Temp_outer = P.difference(frozenset([P_i]))
                for inner_part in partition(P_i):

                    # check if uv in separate blocks that can be merged to P
                    # maybe this or is not necessary in cyclic version
                    if len(inner_part) != 2 or any(frozenset([u, v]).issubset(inner) for inner in inner_part):
                        continue

                    Temp = frozenset([inner for inner in Temp_outer] + [inner for inner in inner_part])
                    minimum = min(minimum, memoization(t_prim, T, C, X, Temp, labels, G, K) + 1)
                break

            C[(t, X, P)] = min(minimum, memoization(t_prim, T, C, X, P, labels, G, K))

    elif labels[t][0] == FORGET_NODE:
        t_prim = [i for i in T.neighbors(t)][0]
        w = labels[t][1]
        P_prim = frozenset(P)

        minimum = sys.maxsize
        for P_i in P_prim:
            # add w to P_i
            Temp = P_prim.difference(frozenset([P_i]))
            P_i = P_i.union([w])
            Temp = frozenset([inner for inner in Temp] + [P_i])

            minimum = min(minimum, memoization(t_prim, T, C, X.union([w]), Temp, labels, G, K))

        if w not in K:
            C[(t, X, P)] = min(minimum, memoization(t_prim, T, C, X, P, labels, G, K))
                               #memoization(t_prim, T, C, X.union([w]), P.union([frozenset([w])]), labels, G, K))
        else:
            C[(t, X, P)] = minimum

    elif labels[t][0] == JOIN_NODE:

        t1 = [i for i in T.neighbors(t)][0]
        t2 = [i for i in T.neighbors(t)][1]

        minimum = sys.maxsize

        for P1 in partition(X):
            for P2 in partition(X):
                if is_acyclic_merge(P1, P2, P):
                    minimum = min(minimum,
                                  memoization(t1, T, C, X, P1, labels, G, K) + memoization(t2, T, C, X, P2,
                                                                                            labels, G, K))

        C[(t, X, P)] = minimum

    elif labels[t][0] == LEAF_NODE:

        if X == frozenset([]):
            C[(t, X, P)] = sys.maxsize
        else:
            C[(t, X, P)] = 0

    return C[(t, X, P)]


def remove_unnecessary_bags(T, labels, u, root):
    all_nodes = list(T.nodes())
    for x in all_nodes:
        if labels[x] == (INTRODUCE_VERTEX_NODE, u) or labels[x] == (FORGET_NODE, u):
            for m in T.predecessors(x):
                for n in T.successors(x):
                    T.add_edge(m, n)

            if x == root:
                root = [n for n in T.successors(x)][0]

            T.remove_node(x)
    return T, root


def add_terminal_to_decomposition(T, u, root, labels):
    # add u to every bag in T
    edges_to_add = []
    all_nodes = list(T.nodes())
    for x in all_nodes:
        if T.degree[x] == 1 or x == root:
            if u not in T.node[x][bag] and len(T.node[x][bag]) != 0:
                new_node = T.number_of_nodes()  # len(mapping)
                if x == root:
                    labels[new_node] = (FORGET_NODE, list(T.node[x][bag])[0])  # T.node[x][bag][0] = {v}

                    root = new_node
                    edges_to_add.append((new_node, x))
                else:
                    labels[new_node] = (LEAF_NODE, u)

                    # len(mapping[x]) == 1
                    labels[x] = (INTRODUCE_VERTEX_NODE, list(T.node[x][bag])[0])

                    edges_to_add.append((x, new_node))
                T.add_node(new_node, bag=frozenset([u]))

        T.node[x][bag] = T.node[x][bag].union(T.node[x][bag], frozenset([u]))

    T.add_edges_from(edges_to_add)

    return T, u, root, labels


def tree_decomp_dynamic(T: nx.DiGraph, root: int, labels: dict, K: list, G: nx.Graph):
    u = K[0]  # one of terminals

    T, u, root, labels = add_terminal_to_decomposition(T, u, root, labels)
    T, root = remove_unnecessary_bags(T, labels, u, root)

    C = {}

    return memoization(root, T, C, frozenset([u]), frozenset([frozenset([u])]), labels, G, K)
