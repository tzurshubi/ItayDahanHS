# snake spqr pairs
import networkx as nx
import sage.all
from sage.graphs.connectivity import spqr_tree
from sage.graphs.graph import Graph

from helpers import index_to_node_stuff
from helpers.helper_funcs import flatten, intersection, max_disj_set_upper_bound, draw_grid
from heuristics.heauristics.naive_spqr.naive_spqr import get_neighbors_pairs, \
    get_all_spqr_pairs_new







# def prune_t(current_sn, parent_sn, tree, g, s, t):
#     current_sps = list(current_sn[1].networkx_graph().nodes)
#     for neighbor in tree.neighbors(current_sn):
#         if neighbor == parent_sn:
#             continue
#         intersection_sps = intersection(neighbor[1].networkx_graph().nodes, current_sps)
#         if s not in intersection_sps and t not in intersection_sps and g.has_edge(intersection_sps[0], intersection_sps[1]):
#             tree.delete_edge((current_sn, neighbor))
#         else:
#             prune_t(neighbor, current_sn, tree, g, s, t)


# def snake_exclusion_pairs_spqr(comp, in_node, out_node, x_filter=False, y_filter=True, in_neighbors=False, out_neighbors=False):
#     # comp_sage = Graph(comp)
#     # tree = spqr_tree(comp_sage)
#     # path = get_path(in_node, out_node, tree)
#     # [[prune_t(x, p, tree, comp, in_node, out_node) for x in tree.neighbors(p) if x not in path] for p in path]
#     # pairs = get_all_spqr_pairs(tree, comp, in_node, out_node)
#     # res = max_disj_set_upper_bound(comp.nodes, pairs)
#     # print(res)
#     res = get_max_nodes_spqr_snake(comp, in_node, out_node, x_filter, y_filter, in_neighbors, out_neighbors)
#     return res


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

def get_max_nodes_spqr_snake(component, in_node, out_node, x_filter=False, y_filter=False, in_neighbors=False, out_neighbors=False, return_pairs=False):
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
    pairs = get_all_spqr_pairs_new(tree, component, in_node, out_node)
    # if in_neighbors:
    #     pairs.update(get_neighbors_pairs(component, in_node))
    # if out_neighbors:
    #     pairs.update(get_neighbors_pairs(component, out_node))
    res = pairs if return_pairs else max_disj_set_upper_bound(component.nodes, pairs, x_filter, y_filter, component)
    # print('ret', res)
    return res



