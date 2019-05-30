import itertools
import sys
from functools import reduce

from graph import Graph
from itertools import chain, combinations
import networkx as nx
import matplotlib


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
                    G_nx.add_edge(k, t, weight=w[(k,t)])
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

def nice_tree_decomp(G):
    out = {}
    for node, neighbours in G.items():
        vert = list(node)
        if len(neighbours) == 1:
            for i in range(1, len(vert)):
                out[frozenset(vert[0:i])] = [frozenset(vert[0:i + 1])]
            else:
                for N in neighbours:
                    Xj_Xi = [x for x in N if x not in vert]
                    interXi_Xj = [x for x in N if x in vert]
                    for i in range(len(interXi_Xj)):
                        pass  # TODO
    print(out)


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

    out = {}
    for node in decomp.nodes:
        e = []
        for edge in decomp.edges:
            if edge[0] == node:
                e.append(edge[1])
            if edge[1] == node:
                e.append(edge[0])
        out[node] = e
    print(out)
    print(list(decomp.nodes()))
    #nice_tree_decomp(out)
    #nx.draw(decomp, with_labels=True)
    return treewidth, decomp


def tree_decomposition():
    matplotlib.interactive(True)
    #G = {0: [1], 1: [0, 2, 3], 2: [1, 4], 3: [1, 4], 4: [2, 3, 5], 5: [4]}
    G = {0: [1, 2, 3, 4], 1: [0, 2, 3, 4], 2: [0, 1, 3, 4], 3: [0, 1, 2, 4], 4: [0, 1, 2, 3]}
    G_nx = nx.DiGraph()
    G_nx.add_nodes_from(G.keys())
    for k, v in G.items():
        G_nx.add_edges_from(([(k, t) for t in v]))
    return treewidth_decomp(G)

tree_decomposition()
