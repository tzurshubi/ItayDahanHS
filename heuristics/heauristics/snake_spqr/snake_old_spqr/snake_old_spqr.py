import networkx as nx
from sage.graphs.connectivity import spqr_tree
from sage.graphs.graph import Graph

from helpers.helper_funcs import flatten, intersection, max_disj_set_upper_bound
from heuristics.heauristics.old_spqr.old_spqr import get_max_nodes_spqr_old, get_all_spqr_pairs_old


# def snake_exclusion_pairs_spqr_old(comp, in_node, out_node, x_filter=False, y_filter=False):
#     # comp_sage = Graph(comp)
#     # tree = spqr_tree(comp_sage)
#     # path = get_path(in_node, out_node, tree)
#     # [[prune_t(x, p, tree, comp, in_node, out_node) for x in tree.neighbors(p) if x not in path] for p in path]
#     # pairs = get_all_spqr_pairs(tree, comp, in_node, out_node)
#     # res = max_disj_set_upper_bound(comp.nodes, pairs)
#     # print(res)
#     nodes = snake_exclusion_set_spqr_old(comp, in_node, out_node)
#     comp_reduced = comp.subgraph(nodes)
#     res = get_max_nodes_spqr_old(comp_reduced, in_node, out_node, x_filter, y_filter)
#     return res


# def get_max_nodes_spqr_actual_set(comp_reduced, in_node, out_node):
#     pass


# def snake_exclusion_pairs_spqr_actual_set(comp, in_node, out_node):
#     # comp_sage = Graph(comp)
#     # tree = spqr_tree(comp_sage)
#     # path = get_path(in_node, out_node, tree)
#     # [[prune_t(x, p, tree, comp, in_node, out_node) for x in tree.neighbors(p) if x not in path] for p in path]
#     # pairs = get_all_spqr_pairs(tree, comp, in_node, out_node)
#     # res = max_disj_set_upper_bound(comp.nodes, pairs)
#     # print(res)
#     nodes = snake_exclusion_set_spqr(comp, in_node, out_node)
#     comp_reduced = comp.subgraph(nodes)
#     res = get_max_nodes_spqr_actual_set(comp_reduced, in_node, out_node)
#     return res


def get_max_nodes_spqr_snake_old(component, in_node, out_node, x_filter=False, y_filter=False, in_neighbors=False, out_neighbors=False, return_pairs=False):
    # print(f"s:{s}, t:{t}")
    component = component.copy()
    # if s and t are connected, the snake must go directly to t
    if component.has_edge(in_node, out_node):
        return 2
    else:
        component.add_edge(in_node, out_node)
    comp_sage = Graph(component)
    tree = spqr_tree(comp_sage)
#     for x in tree:
#         print(x)
#     print('--------------------------------')
    pairs = get_all_spqr_pairs_old(tree, component, in_node, out_node)
    # if in_neighbors:
    #     pairs.update(get_neighbors_pairs(component, in_node))
    # if out_neighbors:
    #     pairs.update(get_neighbors_pairs(component, out_node))
    res = pairs if return_pairs else max_disj_set_upper_bound(component.nodes, pairs, x_filter, y_filter, component)
    # print('ret', res)
    return res



# def snake_exclusion_set_len_spqr(comp, in_node, out_node):
#     return len(snake_exclusion_set_spqr_old(comp, in_node, out_node))
