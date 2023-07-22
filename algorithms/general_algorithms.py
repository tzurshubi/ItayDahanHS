from Definitions.bi_comp_entry import BiCompEntry
from helpers.COMMON import LSP_MODE
from helpers.helper_funcs import bcc_thingy, intersection
from heuristics.heauristics.snake_spqr.snake_general import snake_exclusion_set_spqr

NUM_OF_PAIRS = 5
N = 0


def g(state):
    path = state.path
    g_value = len(path) - 1
    return g_value


def function(state, heuristic, G, target):
    return heuristic(state, G, target)


def calc_comps(state, G, target, algorithm, mode=LSP_MODE):
    _, _, relevant_comps, _, reach_nested, current_node = bcc_thingy(state, G, target)
    if relevant_comps == -1:
        return -1  # no path
    n = len(relevant_comps)
    if n == 0:
        return
    cut_nodes = [(current_node, target)] if n == 1 else [(current_node,
                                                          list(intersection(relevant_comps[0], relevant_comps[1]))[
                                                              0])] + [(list(
        intersection(relevant_comps[i - 1], relevant_comps[i]))[0], list(
        intersection(relevant_comps[i + 1], relevant_comps[i]))[0]) for i in range(1, n - 1)] + [(list(
        intersection(relevant_comps[n - 2], relevant_comps[n - 1]))[0], target)]

    if mode != LSP_MODE:
        subgraphs = [reach_nested.subgraph(comp) for comp in relevant_comps]
    else:
        subgraphs = [reach_nested.subgraph(snake_exclusion_set_spqr(reach_nested.subgraph(comp), in_node, out_node)) for comp, (in_node, out_node) in zip(relevant_comps, cut_nodes)]

    # comps_hs = [BiCompEntry(in_node, out_node, reach_nested.subgraph(comp), algorithm(comp, in_node, out_node)) for
    #             (in_node, out_node), comp in zip(cut_nodes, subgraphs)]

    comps_hs = [BiCompEntry(in_node, out_node, comp, algorithm(comp, in_node, out_node)) for (in_node, out_node), comp in zip(cut_nodes, subgraphs)]

    return comps_hs


# def snake_calc_comps(state, G, target, algorithm):
#     #     state.print()
#     #     print("target:", target)
#     _, _, relevant_comps, _, reach_nested, current_node = bcc_thingy(state, G, target)
#     #     print(relevant_comps)
#     if relevant_comps == -1:
#         #         print("relevant comps = -1")
#         return -1  # no path
#     n = len(relevant_comps)
#     if n == 0:
#         #         print("no relevant comps")
#         return 0
#     cut_nodes = [(current_node, target)] if n == 1 else [(current_node,
#                                                           list(intersection(relevant_comps[0], relevant_comps[1]))[
#                                                               0])] + [(list(
#         intersection(relevant_comps[i - 1], relevant_comps[i]))[0], list(
#         intersection(relevant_comps[i + 1], relevant_comps[i]))[0]) for i in range(1, n - 1)] + [(list(
#         intersection(relevant_comps[n - 2], relevant_comps[n - 1]))[0], target)]
#     subgraphs = [reach_nested.subgraph(snake_exclusion_set_spqr(reach_nested.subgraph(comp), in_node, out_node)) for
#                  comp, (in_node, out_node) in zip(relevant_comps, cut_nodes)]
#     comps_hs = [BiCompEntry(in_node, out_node, comp, algorithm(comp, in_node, out_node)) for (in_node, out_node), comp
#                 in zip(cut_nodes, subgraphs)]
#
#     return comps_hs


# get sum of nodes from each comp, don't know why it's named like that
def ex_pairs(state, G, target, algorithm, mode=LSP_MODE, prune=False):
    current_node = state.current
    if current_node == target:
        return 1
    comp_hs = calc_comps(state, G, target, algorithm, mode=mode)
    if isinstance(comp_hs, int) and comp_hs <= 0:
        return comp_hs
    relevant_nodes = 1 + sum([c.h - 1 for c in comp_hs])
    return relevant_nodes

