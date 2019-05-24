import sys
import networkx as nx

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


def relabel(T):
    mapping = {}
    i = 0
    for x in T.nodes:
        mapping[i] = x
        i += 1

    return nx.relabel_nodes(T, {v: k for k, v in mapping.items()}), mapping


def is_acyclic_merge(P1, P2, P):
    new_P = []
    for P_i in P1:
        new_p = frozenset([])
        for P_prim_i in P2:
            if len(P_i.intersection(P_prim_i)) != 0:
                new_p = new_p.union(P_i.union(P_prim_i))

        new_P.append(new_p)

    return frozenset(new_P).issubset(P) and P.issubset(frozenset(new_P))


def memoisation(t, T, C, X, P, labels):
    if (t, X, P) in C.keys():
        return C[(t, X, P)]

    if labels[t][0] == INTRODUCE_VERTEX_NODE:
        t_prim = [i for i in T.neighbors(t)][0]

        # labels[t] is tuple (name, v)
        v = labels[t][1]
        if (v in K and v in X and frozenset([v]) not in P) or (v in K and v not in X):
            C[(t, X, P)] = sys.maxsize
        elif v in K and v in X and frozenset([v]) in P:
            C[(t, X, P)] = memoisation(t_prim, T, C, X.difference([v])
                                       , frozenset([x for x in P if x != frozenset([v])])
                                       , labels)
        else:
            C[(t, X, P)] = memoisation(t_prim, T, C, X, P, labels)

    elif labels[t][0] == INTRODUCE_EDGE_NODE:
        t_prim = [i for i in T.neighbors(t)][0]
        # labels[t] is tuple (name, (u, v))
        u, v = labels[t][1]

        # if for all (u, v) not in P_i then u, v are in separate P_i
        if u not in X or v not in X or all(not frozenset([u, v]).issubset(P_i) for P_i in P):
            C[(t, X, P)] = memoisation(t_prim, T, C, X, P, labels)
        else:
            minimum = sys.maxsize
            for P_i in P:
                # (u, v) is not in separate P_i
                if not frozenset([u, v]).issubset(P_i):
                    continue

                Temp_outer = P.difference(frozenset([P_i]))
                for inner_part in partition(P_i):

                    # check if uv in separate blocks that can be merged to P
                    if len(inner_part) != 2 or any(frozenset([u, v]).issubset(inner) for inner in inner_part):
                        continue

                    Temp = frozenset([inner for inner in Temp_outer] + [inner for inner in inner_part])
                    minimum = min(minimum, memoisation(t_prim, T, C, X, Temp, labels) + 1)
                break

            C[(t, X, P)] = min(minimum, memoisation(t_prim, T, C, X, P, labels))
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

            minimum = min(minimum, memoisation(t_prim, T, C, X.union([w]), Temp, labels))

        C[(t, X, P)] = min(minimum, memoisation(t_prim, T, C, X, P, labels))
    elif labels[t][0] == JOIN_NODE:

        t1 = [i for i in T.neighbors(t)][0]
        t2 = [i for i in T.neighbors(t)][1]

        minimum = sys.maxsize

        for P1 in partition(X):
            for P2 in partition(X):
                if is_acyclic_merge(P1, P2, P):
                    minimum = min(minimum,
                                  memoisation(t1, T, C, X, P1, labels) + memoisation(t2, T, C, X, P2,
                                                                                     labels))

        C[(t, X, P)] = minimum

    elif labels[t][0] == LEAF_NODE:

        if X == frozenset([]):
            C[(t, X, P)] = sys.maxsize
        else:
            C[(t, X, P)] = 0

    return C[(t, X, P)]


def tree_decomp_dynamic(G, T, root, labels, K, l):
    u = K[0]
    # T, mapping = relabel(T)

    # verticies in T are frozensets
    # make T directed

    # add u to every bag in T
    edges_to_add = []
    iter = list(T.nodes())
    for x in iter:
        # print(x, T.degree[x], [i for i in T.neighbors(x)])
        if T.degree[x] == 1 or x == root:
            # print(x)
            # mapping is map
            # if u not in mapping[x]:
            if u not in T.node[x][bag]:
                new_node = T.number_of_nodes()  # len(mapping)
                if x == root:
                    labels[new_node] = (FORGET_NODE, list(T.node[x][bag])[0])  # T.node[x][bag][0] = {v}
                    root = new_node
                else:
                    # print("DODAJE")
                    labels[new_node] = (LEAF_NODE, u)

                    # len(mapping[x]) == 1
                    labels[x] = (INTRODUCE_VERTEX_NODE, list(T.node[x][bag])[0])

                    edges_to_add.append((x, new_node))
                    T.add_node(new_node, bag=frozenset([u]))

        T.node[x][bag] = T.node[x][bag].union(T.node[x][bag], frozenset([u]))

    T.add_edges_from(edges_to_add)
    T.remove_node(7)
    T.add_edge(1, 0)

    C = {}

    return memoisation(root, T, C, frozenset([u]), frozenset([frozenset([u])]), labels)


T = {4: [3]
    , 3: [5]
    , 5: [2]
    , 2: [6]
    , 6: [1]
    , 1: [7]
    , 7: [0]
    , 0: []}

bags = {4: frozenset([1])
    , 3: frozenset([1, 2])
    , 5: frozenset([1, 2])
    , 2: frozenset([0, 1, 2])
    , 6: frozenset([0, 1, 2])
    , 1: frozenset([0, 1])
    , 7: frozenset([0, 1])
    , 0: frozenset([0])}

labels = {4: (FORGET_NODE, 2)
    , 3: (INTRODUCE_EDGE_NODE, (1, 2))
    , 5: (FORGET_NODE, 0)
    , 2: (INTRODUCE_EDGE_NODE, (0, 2))
    , 6: (INTRODUCE_VERTEX_NODE, 2)
    , 1: (INTRODUCE_EDGE_NODE, (0, 1))
    , 7: (INTRODUCE_VERTEX_NODE, 1)
    , 0: (LEAF_NODE, 0)
          }

edges_to_add = []
Tree = nx.DiGraph()
for x in T.keys():
    Tree.add_node(x, bag=bags[x])
    # print(x, Tree.node[x][bag])
    for y in T[x]:
        edges_to_add.append((x, y))
Tree.add_edges_from(edges_to_add)
K = [1, 2, 0]
print(tree_decomp_dynamic(T, Tree, 4, labels, K, 5))
# nx.draw(Tree, with_labels=True)
