from algorithms.general_algorithms import ex_pairs
from algorithms.incremental_algorithms import ex_pairs_incremental
from helpers.COMMON import SNAKE_MODE, LSP_MODE
from helpers.helper_funcs import max_disj_set_upper_bound, bcc_thingy
from heuristics.heauristics.naive_spqr.naive_spqr import get_max_nodes_spqr_new


def count_nodes_bcc(state, G, target, is_incremental=False):
    if is_incremental:
        return ex_pairs_incremental(state, G, target, lambda g, i, o: len(g.nodes))
    _, _, relevant_comps, _, _, _ = bcc_thingy(state, G, target)
    # print("releveannnttttt", relevant_comps)
    if relevant_comps == -1:
        return -1  # if theres no path
    ret = 1
    for comp in relevant_comps:
        ret += len(comp) - 1
    return ret

def count_nodes_bcc_x(state, G, target, is_incremental=False):
    if is_incremental:
        return ex_pairs_incremental(state, G, target, lambda g,i,o: max_disj_set_upper_bound(g.nodes, [], True, False, g), mode=SNAKE_MODE)
    return ex_pairs(state, G, target, lambda g,i,o: max_disj_set_upper_bound(g.nodes, [], True, False, g), mode=SNAKE_MODE)


def count_nodes_bcc_y(state, G, target, is_incremental=False):
    if is_incremental:
        return ex_pairs_incremental(state, G, target, lambda g,i,o: max_disj_set_upper_bound(g.nodes, [], False, True, g), mode=SNAKE_MODE)
    return ex_pairs(state, G, target, lambda g,i,o: max_disj_set_upper_bound(g.nodes, [], False, True, g), mode=SNAKE_MODE)


# def ex_pairs_using_sage_flow(state, G, target, is_incremental=False):
#     if is_incremental:
#         return ex_pairs_incremental(state, G, target, lambda g, i, o: get_max_nodes(g, i, o, sage_flow))
#     return ex_pairs(state, G, target, lambda g,i,o: get_max_nodes(g, i, o, sage_flow))
#

