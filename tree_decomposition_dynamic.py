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

    # V_F = frozenset([x for x in G.nodes if x not in T.node[t][bag].difference(X)])
    # V_t = frozenset([x for x in G.nodes])
    #
    # if not V_t.intersection(K).issubset(V_F):
    #     C[(t, X, P)] = sys.maxsize
    #     return C[(t, X, P)]
    #
    # V_F = G.subgraph(V_F)
    # components = [c for c in nx.connected_components(V_F)]
    # if len(components) > len(P):
    #     C[(t, X, P)] = sys.maxsize
    #     return C[(t, X, P)]
    #
    # for p in P:
    #     Temp = [x for x in V_F.nodes if x not in X.difference(p)]
    #     components = [c for c in nx.connected_components(V_F.subgraph(Temp))]
    #     if not any(p == frozenset(c.intersection(T.node[t][bag])) for c in components):
    #         #print("WRONG", X, P, T.node[t][bag], components)
    #         C[(t, X, P)] = sys.maxsize
    #         return C[(t, X, P)]

    if labels[t][0] == INTRODUCE_VERTEX_NODE:
        t_prim = [i for i in T.neighbors(t)][0]

        # labels[t] is tuple (name, v)
        v = labels[t][1]
        if (v in X and frozenset([v]) not in P) or (v in K and v not in X):
            C[(t, X, P)] = sys.maxsize
        elif v in X and frozenset([v]) in P:
            G.remove_node(v)
            C[(t, X, P)] = memoization(t_prim, T, C, X.difference([v])
                                       , frozenset([x for x in P if x != frozenset([v])])
                                       , labels, G, K)
            G.add_node(v)
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

        nodes, edges = set(), set()
        build_subgraph_Gf(T, t1, nodes, edges, labels)
        G1 = nx.Graph()
        G1.add_nodes_from(nodes)
        G1.add_edges_from(edges)

        nodes, edges = set(), set()
        build_subgraph_Gf(T, t2, nodes, edges, labels)
        G2 = nx.Graph()
        G2.add_nodes_from(nodes)
        G2.add_edges_from(edges)

        for P1 in partition(X):
            for P2 in partition(X):
                if is_acyclic_merge(P1, P2, P):
                    minimum = min(minimum,
                                  memoization(t1, T, C, X, P1, labels, G1, K) + memoization(t2, T, C, X, P2,
                                                                                            labels, G2, K))

        C[(t, X, P)] = minimum

    elif labels[t][0] == LEAF_NODE:

        if X == frozenset([]):
            C[(t, X, P)] = sys.maxsize
        else:
            C[(t, X, P)] = 0

    #print(t, X, P, labels[t], C[(t, X, P)])

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
        # print(x, T.degree[x], [i for i in T.neighbors(x)])
        if T.degree[x] == 1 or x == root:
            if u not in T.node[x][bag] and len(T.node[x][bag]) != 0:
                new_node = T.number_of_nodes()  # len(mapping)
                if x == root:
                    labels[new_node] = (FORGET_NODE, list(T.node[x][bag])[0])  # T.node[x][bag][0] = {v}

                    root = new_node
                    edges_to_add.append((new_node, x))
                else:
                    # print("DODAJE")
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
    , 28: []}

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
    , 28: frozenset([])}

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

G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 4), (2, 5), (3, 5), (3, 6), (4, 5), (5, 6)])

edges_to_add = []
Tree = nx.DiGraph()
for x in T.keys():
    Tree.add_node(x, bag=bags[x])
    # print(x, Tree.node[x][bag])
    for y in T[x]:
        edges_to_add.append((x, y))
Tree.add_edges_from(edges_to_add)
K = [4, 5, 6]
#print(tree_decomp_dynamic(Tree, 27, labels, K, G))
# nx.draw(Tree, with_labels=True)
