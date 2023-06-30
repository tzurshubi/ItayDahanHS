# incremental

from Definitions.state import State
from algorithms.general_algorithms import calc_comps
from helpers.COMMON import LSP_MODE
from helpers.helper_funcs import intersection

# get sum of nodes from each comp, dont know why its named like that
def ex_pairs_incremental(state, G, target, algorithm, mode=LSP_MODE):
    # print('hi')
    current_node = state.current
    if current_node == target:
        return 1
    if not state.bccs:
        # print('start 1')
        state.bccs = calc_comps(state, G, target, algorithm, mode)
        # print('end 1')
        # insert as object
    else:
        bccs = state.bccs
        current_comp_in = bccs[0].in_node
        if current_comp_in != current_node:
            # print('!!!!!!!!!!!!!!!!!!!!!', current_node in bccs[0].nodes)
            first_comp_nodes = bccs[0].nodes
            p_availables = list(intersection(state.available_nodes, first_comp_nodes)) + [current_node]
            pseudo_state = State(current_node, [state.path[-2], current_node], p_availables)
            pseudo_target = bccs[0].out_node
            pseudo_G = G.subgraph(first_comp_nodes)
            # print('start 2')
            extra_comps = calc_comps(pseudo_state, pseudo_G, pseudo_target, algorithm, mode)
            # print('end 1')
            # state.print()
            # state.print_bccs()
            if isinstance(extra_comps, int):
                return extra_comps
            state.bccs = extra_comps + bccs[1:]
            # insert as object
    # state.print()
    # state.print_bccs()
    relevant_nodes = 1 + sum([c.h - 1 for c in state.bccs])
    return relevant_nodes

# def snake_ex_pairs_incremental(state, G, target, algorithm):
#     # print('hi')
#     current_node = state.current
#     if current_node == target:
#         return 1
#     if not state.bccs:
# #         print('start 1')
#         state.bccs = snake_calc_comps(state, G, target, algorithm)
# #         state.print()
#         # print('end 1')
#         # insert as object
#     else:
#         bccs = state.bccs
#         current_comp_in = bccs[0].in_node
#         if current_comp_in != current_node:
#             # print('!!!!!!!!!!!!!!!!!!!!!', current_node in bccs[0].nodes)
#             first_comp_nodes = bccs[0].nodes
#             p_availables = list(intersection(state.available_nodes, first_comp_nodes)) + [current_node]
#             pseudo_state = State(current_node, [state.path[-2], current_node], p_availables)
#             pseudo_target = bccs[0].out_node
#             pseudo_G = G.subgraph(first_comp_nodes)
#             # print('start 2')
#             extra_comps = calc_comps(pseudo_state, pseudo_G, pseudo_target, algorithm, mode)
#             # print('end 1')
#             # state.print()
#             # state.print_bccs()
#             if isinstance(extra_comps, int):
#                 return extra_comps
#             state.bccs = extra_comps + bccs[1:]
#             # insert as object
#     # state.print()
# #     state.print_bccs()
#     relevant_nodes = 1 + sum([c.h - 1 for c in state.bccs])
#     return relevant_nodes