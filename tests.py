import copy

import networkx as nx

from graph import Graph
from inclusion import count_steiner_tree
from set_dynamic import set_dynamic
from generate_test import inclusion_brute_force, set_dynamic_brute_force, make_test,x_print_tree
from gaussian_elimination_dynamic import gaussian_elim_dynamic
from tree_decomposition_dynamic import tree_decomp_dynamic
import matplotlib.pyplot as plt


def test_inclusion():
    G = Graph([[1], [0, 2, 3], [1, 4], [1, 4], [2, 3, 5], [4]])
    K = [0, 2, 5]
    w = {(0, 1): 2, (1, 0): 2, (1, 2): 10, (1, 3): 3, (2, 1): 10,(2, 4): 1,(4, 2): 1, (3, 1): 3, (3, 4): 2, (4, 3): 2, (4, 5): 2, (5, 4): 2}
    l = 5
    print(set_dynamic(G, K, w), set_dynamic_brute_force(G, K, w))
    print(inclusion_brute_force(G, K, l), count_steiner_tree(G, K, l))
    print(inclusion_brute_force(G, K, l+1), count_steiner_tree(G, K, l+1))
    print(inclusion_brute_force(G, K, l + 2), count_steiner_tree(G, K, l + 2))

G, K, tree_decomp, labels, root = make_test('stp/es10fst01.stp')

#print("SET", set_dynamic(Graph(Temp), K))

#nx.draw(tree_decomp, with_labels=True)
#plt.show()
#x_print_tree(tree_decomp, root, labels)
#print(K)
#print("SET", set_dynamic(G, K))
print("GAUSS", gaussian_elim_dynamic(tree_decomp, root, labels, K, G))
print("NORMAL", tree_decomp_dynamic(tree_decomp, root, labels, K, G))