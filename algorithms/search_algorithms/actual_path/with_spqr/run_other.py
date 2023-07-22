from Definitions.state import State
from algorithms.search_algorithms.actual_path.with_spqr.with_spqr import get_comp_path
from helpers.COMMON import LSP_MODE
from helpers.helper_funcs import diff, bcc_thingy, intersection
import time as t


def run_other(heuristic, graph, start, target, weight, cutoff, timeout, is_incremental, n=None, save_dir="", mode=LSP_MODE):
    start_time = t.time()
    if n is not None:
        print(f'n: {n}')
        first_node = tuple([0] * (n - 1) + [1])
        not_availables = diff([first_node] + list(graph.neighbors(first_node)), [start, target])
        start_available = tuple(diff(list(graph.nodes), not_availables))
    else:
        start_available = tuple(diff(list(graph.nodes), {start}))
    start_path = (start,)
#     bcc_dict = {}
    start_state = State(start, start_path, start_available)
    start_state.update_dimension(2)
    _, _, relevant_comps, _, reach_nested, current_node = bcc_thingy(start_state, graph, target)
    if relevant_comps == -1:
        return -1  # no path
    n = len(relevant_comps)
    if n == 0:
        return 0

    # h = (lambda x: heuristic(x, graph, target, is_incremental))
    # stron = (lambda x: ex_pairs(x, graph, target))
    cut_nodes = [(current_node, target)] if n == 1 else [(current_node,
                                                          list(intersection(relevant_comps[0], relevant_comps[1]))[
                                                              0])] + [(list(
        intersection(relevant_comps[i - 1], relevant_comps[i]))[0], list(
        intersection(relevant_comps[i + 1], relevant_comps[i]))[0]) for i in range(1, n - 1)] + [(list(
        intersection(relevant_comps[n - 2], relevant_comps[n - 1]))[0], target)]
    subgraphs = [reach_nested.subgraph(comp) for comp in relevant_comps]
    paths = [get_comp_path(comp, in_node, out_node, mode=mode) for (in_node, out_node), comp in zip(cut_nodes, subgraphs)]
    path = [start]
    # for (in_node, out_node),comp in zip(cut_nodes, relevant_comps):
    #     print((in_node, out_node),comp)
    for p in paths:
        # draw_grid('', 'p', g1, [[0]*20]*20, source, target, itn, path= p)
        path += list(p)
    runtime = t.time() - start_time
    return path, -9, runtime, -9, -9, -9, -9
    # expansions, runtime, nodes_extracted_heuristic_values, nodes_extracted_path_len, nodes_chosen, generated_nodes = data
    # return end_state.path if end_state != -1 else [], expansions, runtime, nodes_extracted_heuristic_values, nodes_extracted_path_len, nodes_chosen, generated_nodes


# def run_other(graph, start, target):
#     start_available = tuple(diff(list(graph.nodes), {start}))
#     start_path = (start,)
# #     bcc_dict = {}
#     state = State(start, start_path, start_available)
#     _, _, relevant_comps, _, reach_nested, current_node = bcc_thingy(state, graph, target)
#     if relevant_comps == -1:
#         return -1  # no path
#     n = len(relevant_comps)
#     if n == 0:
#         return 0
#     cut_nodes = [(current_node, target)] if n == 1 else [(current_node, list(intersection(relevant_comps[0], relevant_comps[1]))[0])] + [(list(intersection(relevant_comps[i-1], relevant_comps[i]))[0], list(intersection(relevant_comps[i+1], relevant_comps[i]))[0]) for i in range(1,n-1)] + [(list(intersection(relevant_comps[n-2], relevant_comps[n-1]))[0], target)]
#     subgraphs = [reach_nested.subgraph(comp) for comp in relevant_comps]
#     paths = [get_comp_path(comp, in_node, out_node) for (in_node, out_node), comp in zip(cut_nodes, subgraphs)]
#     path = [start]
#     # for (in_node, out_node),comp in zip(cut_nodes, relevant_comps):
#     #     print((in_node, out_node),comp)
#     for p in paths:
#         # draw_grid('', 'p', g1, [[0]*20]*20, source, target, itn, path= p)
#         path += list(p)
#     return tuple(set(path))