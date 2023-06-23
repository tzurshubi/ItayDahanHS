from algorithms.general_algorithms import ex_pairs
from heuristics.heauristics.recursive_spqr.recursive_spqr import get_max_nodes_spqr_recursive
from algorithms.incremental_algorithms import ex_pairs_incremental


def spqr_recursive_h(state, G, target, is_incremental=False):
    if is_incremental:
        return ex_pairs_incremental(state, G, target, lambda g,i,o: get_max_nodes_spqr_recursive(g, i, o))
    return ex_pairs(state, G, target, lambda g,i,o: get_max_nodes_spqr_recursive(g, i, o))
