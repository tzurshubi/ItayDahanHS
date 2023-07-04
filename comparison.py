import os

from algorithms.search_algorithms.a_star.run_weighted_astar import run_weighted
from algorithms.search_algorithms.dfbnb.run_dfbnb import run_dfbnb
from experiments.experiment import run_other, mother_of_tests
from experiments.tests.get_max_test import test_new_spqr
from helpers import index_to_node_stuff
from helpers.COMMON import *
from helpers.graph_builder_funcs import generate_hard_grid, parse_graph_png, crop_and_parse_graph
from helpers.helper_funcs import draw_grid
from heuristics.heuristics_interface_calls import spqr_recursive_h, ex_pairs_using_spqr
import time as t


def run_mom_test():
    mother_of_tests(algorithm=run_weighted, world=GRIDS_MODE, mode=LSP_MODE, n=None)


def compare_alt_to_astar():
    grid, graph, start, target, index_to_node = crop_and_parse_graph('/mnt/c/Users/itay/Desktop/notebooks/all_graphs/graph_721.png', 20, 20)
    index_to_node_stuff.index_to_node = index_to_node
    index_to_node_stuff.grid = grid
    draw_grid("", graph, grid, start, target, index_to_node, path=[])

    time_start = t.time()
    path_alt = run_other(graph, start, target)
    time_alt = t.time() - time_start

    print(time_alt)
    print(len(path_alt))
    draw_grid("alt", graph, grid, start, target, index_to_node, path=path_alt)

    time_start = t.time()
    path_astar = run_weighted(ex_pairs_using_spqr, graph, start, target, 1, 50000, 2000, True, mode=LSP_MODE)[0]
    time_astar = t.time() - time_start

    print(time_astar)
    print(len(path_astar))

    draw_grid("astar", graph, grid, start, target, index_to_node, path=path_alt)


# def recursive_vs_pairs():
#     while True:
#         grid, graph, start, target, index_to_node = generate_hard_grid(50, 50, 0.5, is_snake=False)
#         index_to_node_stuff.index_to_node = index_to_node
#         index_to_node_stuff.grid = grid
#         ares = run_weighted(test_new_spqr, graph, start, target, 1, 50000, 2000, True, mode=LSP_MODE)



if __name__ == '__main__':
    run_mom_test()

