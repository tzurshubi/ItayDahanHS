import networkx as nx
from sage.graphs.connectivity import spqr_tree
from sage.graphs.graph import Graph

from helpers.helper_funcs import flatten, intersection
from heuristics.heauristics.old_spqr.old_spqr import get_max_nodes_spqr_old






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




def snake_exclusion_set_len_spqr(comp, in_node, out_node):
    return len(snake_exclusion_set_spqr_old(comp, in_node, out_node))
