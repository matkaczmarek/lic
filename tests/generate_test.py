import itertools
import sys

from itertools import chain, combinations
import networkx as nx
import matplotlib.pyplot as plt

LEAF_NODE, INTRODUCE_VERTEX_NODE, INTRODUCE_EDGE_NODE = "Leaf node", "Introduce vertex node", "Introduce edge node"
FORGET_NODE, JOIN_NODE = "Forget node", "Join node"
bag = 'bag'


def powerset_without_empty(X):
    S = list(X)
    return chain.from_iterable(combinations(S, r) for r in range(1, len(S)))


def subsets_without_empty(X):
    return list(map(list, powerset_without_empty(X)))


def powerset(X, l):
    S = list(X)
    return chain.from_iterable(combinations(S, r) for r in range(l + 1))


def subsets(X, l):
    return list(map(list, powerset(X, l)))


def inclusion_brute_force(G, K, l):
    if G.edge_number() < l:
        l = G.edge_number()

    # G = Graph(G)
    potential = subsets(G.verticies(), l + 1)
    for X in potential:
        if not set(K).issubset(set(X)):
            continue

        X = G.subgraph(X)
        root = X.verticies()[0]
        visited, stack = set(), [root]
        while stack:
            v = stack.pop()
            if v not in visited:
                for x in X.neighbours(v):
                    stack.append(x)
            visited.add(v)

        if X.verticies() == list(visited):
            return True

    return False


def set_dynamic_brute_force(G, K, w=None):
    if w is None:
        w = {}
        for v in G.verticies():
            for u in G.neighbours(v):
                w[(v, u)] = 1

    # G = Graph(G)
    potential = subsets(G.verticies(), len(G.verticies()))
    mini = sys.maxsize
    for X in potential:
        if not set(K).issubset(set(X)):
            continue

        X = G.subgraph(X)
        root = X.verticies()[0]
        visited, stack = set(), [root]
        while stack:
            v = stack.pop()
            if v not in visited:
                for x in X.neighbours(v):
                    stack.append(x)
            visited.add(v)

        if X.verticies() == list(visited):
            G_nx = nx.Graph()
            G_nx.add_nodes_from(X.verticies())
            for k, v in X.map.items():
                for t in v:
                    G_nx.add_edge(k, t, weight=w[(k, t)])
            T = nx.minimum_spanning_edges(G_nx, data=True)
            s = 0
            for x in list(T):
                s += x[2]['weight']
            if s < mini:
                mini = s
    return mini


# TREE DECOMPOSITION
#
#
#
#
###################

def make_pseudo_binary(G, v):
    suc = list(G.successors(v))
    B = G.node[v][bag]
    if len(suc) == 0:
        return
    if len(suc) > 1:
        p, q = len(G.nodes), len(G.nodes) + 1
        G.add_node(p, bag=B)
        G.add_node(q, bag=B)
        G.add_edge(v, p)
        G.add_edge(v, q)
        for i in range(len(suc)):
            G.remove_edge(v, suc[i])
            if i == 0:
                G.add_edge(p, suc[i])
            elif i == 1:
                G.add_edge(q, suc[i])
            else:
                G.add_edge(q, suc[i])
        make_pseudo_binary(G, p)
        make_pseudo_binary(G, q)
    else:
        make_pseudo_binary(G, suc[0])


def add_path_tree(G, v):
    suc = list(G.successors(v))

    if len(suc) > 1:
        add_path_tree(G, suc[0])
        add_path_tree(G, suc[1])
    elif len(suc) == 0:
        A = G.node[v][bag]
        iter = frozenset(G.node[v][bag])
        for x in A:
            iter = iter.difference([x])
            u = len(G.nodes)
            G.add_node(u, bag=iter)
            G.add_edge(v, u)
            v = u
        return
    else:
        A = G.node[v][bag]
        B = G.node[suc[0]][bag]

        A_out_B = A.difference(B)
        B_out_A = B.difference(A)

        if len(A_out_B) == 0 and len(B_out_A) == 0:
            add_path_tree(G, suc[0])
            return

        if (A.issubset(B) and len(A) + 1 == len(B)) or (B.issubset(A) and len(A) == len(B) + 1):
            add_path_tree(G, suc[0])
            return

        G.remove_edge(v, suc[0])

        iter = frozenset(A)

        for x in A_out_B:
            iter = iter.difference([x])
            u = len(G.nodes)
            G.add_node(u, bag=iter)
            G.add_edge(v, u)
            v = u

        iter = A.intersection(B)

        if iter == B:
            G.add_edge(v, suc[0])

        for x in B_out_A:
            iter = iter.union([x])
            u = len(G.nodes)
            if iter == B:
                G.add_edge(v, suc[0])
            else:
                G.add_node(u, bag=iter)
                G.add_edge(v, u)
                v = u

        add_path_tree(G, suc[0])


def nice_tree_decomp(G, root):
    make_pseudo_binary(G, root)
    A = G.node[root][bag]
    iter = frozenset(G.node[root][bag])
    v = root
    for x in A:
        iter = iter.difference([x])
        u = len(G.nodes)
        G.add_node(u, bag=iter)
        G.add_edge(u, v)
        v = u
    add_path_tree(G, root)
    return v  # return new root


def min_fill_in_heuristic(graph):
    if len(graph) == 0:
        return None

    min_fill_in_node = None

    min_fill_in = sys.maxsize

    # create sorted list of (degree, node)
    degree_list = [(len(graph[node]), node) for node in graph]
    degree_list.sort()

    # abort condition
    min_degree = degree_list[0][0]
    if min_degree == len(graph) - 1:
        return None

    for (_, node) in degree_list:
        num_fill_in = 0
        nbrs = graph[node]
        for nbr in nbrs:
            # count how many nodes in nbrs current nbr is not connected to
            # subtract 1 for the node itself
            num_fill_in += len(nbrs - graph[nbr]) - 1
            if num_fill_in >= 2 * min_fill_in:
                break

        num_fill_in /= 2  # divide by 2 because of double counting

        if num_fill_in < min_fill_in:  # update min-fill-in node
            if num_fill_in == 0:
                return node
            min_fill_in = num_fill_in
            min_fill_in_node = node

    return min_fill_in_node


def treewidth_decomp(G, heuristic=min_fill_in_heuristic):
    graph = {n: set(G[n]) - set([n]) for n in G}

    # stack containing nodes and neighbors in the order from the heuristic
    node_stack = []

    # get first node from heuristic
    elim_node = heuristic(graph)
    while elim_node is not None:
        # connect all neighbours with each other
        nbrs = graph[elim_node]
        for u, v in itertools.permutations(nbrs, 2):
            if v not in graph[u]:
                graph[u].add(v)

        # push node and its current neighbors on stack
        node_stack.append((elim_node, nbrs))

        # remove node from graph
        for u in graph[elim_node]:
            graph[u].remove(elim_node)

        del graph[elim_node]
        elim_node = heuristic(graph)

    # the abort condition is met; put all remaining nodes into one bag
    decomp = nx.Graph()
    first_bag = frozenset(graph.keys())
    decomp.add_node(first_bag)

    treewidth = len(first_bag) - 1

    while node_stack:
        # get node and its neighbors from the stack
        (curr_node, nbrs) = node_stack.pop()

        # find a bag all neighbors are in
        old_bag = None
        for bag in decomp.nodes:
            if nbrs <= bag:
                old_bag = bag
                break

        if old_bag is None:
            # no old_bag was found: just connect to the first_bag
            old_bag = first_bag

        # create new node for decomposition
        nbrs.add(curr_node)
        new_bag = frozenset(nbrs)

        # update treewidth
        treewidth = max(treewidth, len(new_bag) - 1)

        # add edge to decomposition (implicitly also adds the new node)
        decomp.add_edge(old_bag, new_bag)

    decomp = nx.convert_node_labels_to_integers(decomp, label_attribute='bag')
    # nice_tree_decomp(out)
    # nx.draw(decomp, with_labels=True)
    # plt.show()

    return treewidth, decomp


def DFS(G, v, visited, new_G):
    for x in G.neighbors(v):
        if x not in visited:
            new_G.add_edge(v, x)
            visited.append(x)
            DFS(G, x, visited, new_G)


def make_directed(G):
    visited = [0]
    root = 0
    new_G = nx.DiGraph()
    new_G.add_nodes_from(G.nodes(data=True))
    DFS(G, root, visited, new_G)

    return new_G


def tree_decomposition(G):
    # matplotlib.interactive(True)
    # G = {0: [1], 1: [0, 2, 3], 2: [1, 4, 5], 3: [1, 4, 6], 4: [2, 3, 5, 6], 5: [2, 4], 6: [3, 4]}
    # G = {0: [1, 2, 3, 4], 1: [0, 2, 3, 4], 2: [0, 1, 3, 4], 3: [0, 1, 2, 4], 4: [0, 1, 2, 3]}
    G_nx = nx.DiGraph()
    G_nx.add_nodes_from(G.keys())
    for k, v in G.items():
        G_nx.add_edges_from(([(k, t) for t in v]))
    return treewidth_decomp(G)


def read_stp_file(filename):
    lines = [line.rstrip('\n') for line in open(filename)]
    edges = []
    K = []

    for line in lines:
        line = line.split(" ")
        if line[0] == "E":
            edges.append((int(line[1]), int(line[2])))
        if line[0] == "T":
            K.append(int(line[1]))

    return edges, K


def x_print_tree(Tree, v, labels):
    suc = list(Tree.successors(v))
    print(v, Tree.node[v][bag], suc, labels[v])
    for x in suc:
        x_print_tree(Tree, x, labels)


def make_labels(G, v, labels):
    suc = list(G.successors(v))

    if len(suc) == 0:
        labels[v] = (LEAF_NODE, v)
    elif len(suc) == 1:
        u = suc[0]
        A = G.node[v][bag]
        B = G.node[u][bag]

        if len(A) == len(B):
            print(A, B, "ERR", v)
            raise SyntaxError
        elif len(A) == len(B) + 1:
            labels[v] = (INTRODUCE_VERTEX_NODE, list(A.difference(B))[0])
        elif len(B) == len(A) + 1:
            labels[v] = (FORGET_NODE, list(B.difference(A))[0])
        else:
            raise SyntaxError
    elif len(suc) == 2:
        labels[v] = (JOIN_NODE, v)
    else:
        raise SyntaxError

    for x in suc:
        make_labels(G, x, labels)


def add_ienode(Tree, v, edges, labels):
    if len(edges) == 0:
        return
    suc = list(Tree.successors(v))
    if labels[v][0] == FORGET_NODE:
        u = suc[0]
        forget = labels[v][1]
        B = Tree.node[u][bag]

        iter = edges[:]
        for x, y in edges:
            if (x == forget and y in B) or (y == forget and x in B):
                p = len(Tree.nodes)
                Tree.add_node(p, bag=B)
                Tree.remove_edge(v, u)
                Tree.add_edge(v, p)
                Tree.add_edge(p, u)
                u = p
                labels[p] = (INTRODUCE_EDGE_NODE, (x, y))
                iter.remove((x, y))

        edges = iter
    for x in suc:
        add_ienode(Tree, x, edges, labels)


def add_ienodes(Tree, root, edges, labels):
    add_ienode(Tree, root, edges, labels)


def make_test(filename):
    edges, K = read_stp_file(filename)

    G = nx.Graph()
    G.add_edges_from(edges)

    tw, tree_decomp = treewidth_decomp(G)
    print("TW", tw)
    tree_decomp = make_directed(tree_decomp)
    root = nice_tree_decomp(tree_decomp, 0)
    labels = {}
    make_labels(tree_decomp, root, labels)
    add_ienodes(tree_decomp, root, edges, labels)

    v = labels[root][1]

    if v not in K:
        K.append(v)

    return G, K, tree_decomp, labels, root
