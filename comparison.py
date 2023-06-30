from algorithms.search_algorithms.a_star.run_weighted_astar import run_weighted
from algorithms.search_algorithms.dfbnb.run_dfbnb import run_dfbnb
from experiments.experiment import run_other, mother_of_tests
from helpers import index_to_node_stuff
from helpers.COMMON import *
from helpers.graph_builder_funcs import generate_hard_grid
from helpers.helper_funcs import draw_grid
from heuristics.heuristics_interface_calls import spqr_recursive_h
import time as t

if __name__ == '__main__':
    grid, graph, start, target, index_to_node = generate_hard_grid(30, 30, 0.4, is_snake=False)
    index_to_node_stuff.index_to_node = index_to_node
    index_to_node_stuff.grid = grid
    # mother_of_tests(algorithm=run_weighted, world=GRIDS_MODE, mode=LSP_MODE, n=None)
    draw_grid("", graph, grid, 0, target, index_to_node, path=[])
    #
    time_start = t.time()
    path_alt = run_other(graph, start, target)
    time_alt = t.time() - time_start

    print(time_alt)
    print(len(path_alt))
    draw_grid("alt", graph, grid, 0, max(graph.nodes), index_to_node, path=path_alt)

    time_start = t.time()
    path_astar = run_weighted(spqr_recursive_h, graph, start, target, 1, 50000, 2000, True, mode="not snake")[0]
    time_astar = t.time() - time_start

    print(time_astar)
    print(len(path_astar))

    draw_grid("astar", graph, grid, 0, max(graph.nodes), index_to_node, path=path_alt)

