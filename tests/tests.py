import copy
import time

from tests.generate_test import inclusion_brute_force, set_dynamic_brute_force, make_test
from tests.graph import Graph
from tests.inclusion import count_steiner_tree
from steiner_tree_algorithms.set_dynamic import set_dynamic
from steiner_tree_algorithms.gaussian_elimination_dynamic import gaussian_elim_dynamic
from steiner_tree_algorithms.tree_decomposition_dynamic import tree_decomp_dynamic


def test_inclusion():
    G = Graph([[1], [0, 2, 3], [1, 4], [1, 4], [2, 3, 5], [4]])
    K = [0, 2, 5]
    w = {(0, 1): 2, (1, 0): 2, (1, 2): 10, (1, 3): 3, (2, 1): 10, (2, 4): 1, (4, 2): 1, (3, 1): 3, (3, 4): 2, (4, 3): 2,
         (4, 5): 2, (5, 4): 2}
    l = 5
    print(set_dynamic(G, K, w), set_dynamic_brute_force(G, K, w))
    print(inclusion_brute_force(G, K, l), count_steiner_tree(G, K, l))
    print(inclusion_brute_force(G, K, l + 1), count_steiner_tree(G, K, l + 1))
    print(inclusion_brute_force(G, K, l + 2), count_steiner_tree(G, K, l + 2))


def do_test(filename):
    print("Instance", filename)
    G, K, tree_decomp, labels, root = make_test(filename)
    start = time.time()
    print("RBA", gaussian_elim_dynamic(tree_decomp, root, labels, K, G))
    end = time.time()
    print("RBA: ", (end - start) * 1000)


def all_test(filename):
    print("Instance", filename)
    G, K, tree_decomp, labels, root = make_test(filename)
    start = time.time()
    print("RBA", gaussian_elim_dynamic(tree_decomp, root, labels, K, copy.deepcopy(G)))
    end = time.time()
    print("RBA: ", (end - start) * 1000)

    start = time.time()
    print("CTD", tree_decomp_dynamic(tree_decomp, root, labels, K, copy.deepcopy(G)))
    end = time.time()
    print("CDT duration: ", (end - start) * 1000)

    start = time.time()
    print("DPS", set_dynamic(copy.deepcopy(G), K))
    end = time.time()
    print("DPS duration: ", (end - start) * 1000)

def test_prefix_b():
    do_test('stp/b/b01.stp')
    do_test('stp/b/b02.stp')
    do_test('stp/b/b08.stp')
    do_test('stp/b/b09.stp')
    do_test('stp/b/b15.stp')

def test_prefix_es():
    all_test('stp/es/es10fst11.stp')
    all_test('stp/es/es10fst12.stp')
    all_test('stp/es/es10fst13.stp')
    all_test('stp/es/es10fst14.stp')
    all_test('stp/es/es10fst15.stp')
    all_test('stp/es/es10fst16.stp')

def test_prefix_eslarge():
    all_test('stp/eslarge/es90fst12.stp')
    all_test('stp/eslarge/es100fst10.stp')
    all_test('stp/eslarge/es80fst06.stp')
    all_test('stp/eslarge/es90fst01.stp')
    all_test('stp/eslarge/es100fst14.stp')