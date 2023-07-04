from algorithms.general_algorithms import ex_pairs
from helpers.COMMON import LSP_MODE, SNAKE_MODE
from heuristics.heauristics.naive_spqr.naive_spqr import get_max_nodes_spqr_new
from heuristics.heauristics.old_spqr.old_spqr import get_max_nodes_spqr_old
from heuristics.heauristics.recursive_spqr.recursive_spqr import get_max_nodes_spqr_recursive
from algorithms.incremental_algorithms import ex_pairs_incremental


def spqr_recursive_h(state, G, target, is_incremental=False):
    if is_incremental:
        return ex_pairs_incremental(state, G, target, lambda g,i,o: get_max_nodes_spqr_recursive(g, i, o), mode=LSP_MODE)
    return ex_pairs(state, G, target, lambda g,i,o: get_max_nodes_spqr_recursive(g, i, o), mode=LSP_MODE)

def ex_pairs_using_spqr(state, G, target, is_incremental=False, mode=LSP_MODE):
    if is_incremental:
        return ex_pairs_incremental(state, G, target, lambda g, i, o: get_max_nodes_spqr_new(g, i, o), mode)
    return ex_pairs(state, G, target, lambda g,i,o: get_max_nodes_spqr_new(g, i, o), mode)

def ex_pairs_using_old_spqr(state, G, target, is_incremental=False, mode=LSP_MODE):
    if is_incremental:
        return ex_pairs_incremental(state, G, target, lambda g, i, o: get_max_nodes_spqr_old(g, i, o), mode)
    return ex_pairs(state, G, target, lambda g,i,o: get_max_nodes_spqr_old(g, i, o), mode)


def snake_ex_pairs_using_spqr_prune(state, G, target, is_incremental=False, x_filter=False, y_filter=False, in_neighbors=False, out_neighbors=False):
    if is_incremental:
        return ex_pairs_incremental(state, G, target, lambda g, i, o: get_max_nodes_spqr_new(g, i, o, x_filter, y_filter, in_neighbors, out_neighbors), mode=SNAKE_MODE)
    return ex_pairs(state, G, target, lambda g,i,o: get_max_nodes_spqr_new(g, i, o), mode=SNAKE_MODE)

def snake_rec_spqr(state, G, target, is_incremental=False):
    if is_incremental:
        return ex_pairs_incremental(state, G, target, lambda g, i, o: get_max_nodes_spqr_recursive(g, i, o), mode=SNAKE_MODE)
    return ex_pairs(state, G, target, lambda g,i,o: get_max_nodes_spqr_recursive(g, i, o), mode=SNAKE_MODE)


snake_only = lambda state, G, target, incremental: snake_ex_pairs_using_spqr_prune(state, G, target, incremental)
snake_y = lambda state, G, target, incremental: snake_ex_pairs_using_spqr_prune(state, G, target, incremental, y_filter=True)
snake_y_in_neighbors = lambda state, G, target, incremental: snake_ex_pairs_using_spqr_prune(state, G, target, incremental, y_filter=True, in_neighbors=True)
snake_y_all_neighbors = lambda state, G, target, incremental: snake_ex_pairs_using_spqr_prune(state, G, target, incremental, y_filter=True, in_neighbors=True, out_neighbors=True)
# snake_y = lambda state, G, target, incremental: snake_ex_pairs_using_spqr_prune(state, G, target, incremental, y_filter=True)
# snake_rec = lambda state, G, target, incremental: snake_ex_pairs_using_rec_spqr(state, G, target, incremental, y_filter=True, in_neighbors=True)
