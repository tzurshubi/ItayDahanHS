
from Definitions.state import BiCompEntry
from helpers.helper_funcs import get_dis_pairs, max_disj_set_upper_bound, bcc_thingy, intersection

NUM_OF_PAIRS = 5
N = 0


def g(state):
    path = state.path
    g_value = len(path) - 1
    return g_value


def function(state, heuristic, G, target):
    return heuristic(state, G, target)


def calc_comps(state, G, target, algorithm):
    _, _, relevant_comps, _, reach_nested, current_node = bcc_thingy(state, G, target)
    if relevant_comps == -1:
        return -1  # no path
    n = len(relevant_comps)
    if n == 0:
        return 0
    cut_nodes = [(current_node, target)] if n == 1 else [(current_node, list(intersection(relevant_comps[0], relevant_comps[1]))[0])] + [(list(intersection(relevant_comps[i-1], relevant_comps[i]))[0], list(intersection(relevant_comps[i+1], relevant_comps[i]))[0]) for i in range(1,n-1)] + [(list(intersection(relevant_comps[n-2], relevant_comps[n-1]))[0], target)]
    subgraphs = [reach_nested.subgraph(comp) for comp in relevant_comps]
    comps_hs = [BiCompEntry(in_node, out_node, reach_nested.subgraph(comp), algorithm(comp, in_node, out_node)) for (in_node, out_node), comp in zip(cut_nodes, subgraphs)]
    return comps_hs


# get sum of nodes from each comp, dont know why its named like that
def ex_pairs(state, G, target, algorithm):
    current_node = state.current
    if current_node == target:
        return 1
    comp_hs = calc_comps(state, G, target, algorithm)
    if isinstance(comp_hs, int) and comp_hs <= 0:
        return comp_hs
    relevant_nodes = 1 + sum([c.h - 1 for c in comp_hs])
    return relevant_nodes

