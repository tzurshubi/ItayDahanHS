# spqr pairs
import networkx as nx
from future.moves import itertools
from sage.graphs.connectivity import spqr_tree
from sage.graphs.graph import Graph

from helpers import COMMON
from helpers.helper_funcs import max_disj_set_upper_bound, flatten, diff, intersection


def nodes_of_sn(current_sn, parent_sn, tree):
    nodes = list(current_sn[1].networkx_graph().nodes)
    for neighbor in tree.neighbors(current_sn):
        if neighbor == parent_sn:
            continue
        nodes += nodes_of_sn(neighbor, current_sn, tree)
    return nodes


def pairs_p_old(current_sn, tree, g, s, t):
    sub_nodes = [diff(nodes_of_sn(neighbor_sn, current_sn, tree), current_sn[1].networkx_graph().nodes) for neighbor_sn
                 in tree.networkx_graph().neighbors(current_sn)]
    sub_nodes = [nodes for nodes in sub_nodes if ((s not in nodes or s in current_sn[1].networkx_graph().nodes) and (
                t not in nodes or t in current_sn[1].networkx_graph().nodes))]
    pairs = []
    for i in range(len(sub_nodes)):
        for j in range(i + 1, len(sub_nodes)):
            pairs += flatten([[(n1, n2) if n1 < n2 else (n2, n1) for n1 in sub_nodes[i]] for n2 in sub_nodes[j]])
    return pairs


def get_all_spqr_pairs_old(tree, component, in_node, out_node):
    #     print('spqr nodes', (tree.networkx_graph().nodes))
    pairs = []
    for c in tree:
        pairs += pairs_spqr_old(c, tree, component, in_node, out_node)
    pairs = set(pairs)
    #     print('--------------------------------PAIRS', len(pairs))
    return pairs



def comp_nodes_for_path(in_node, out_node, node_dict):
    return node_dict[(in_node, out_node)] if (in_node, out_node) in node_dict.keys() else node_dict[
        (out_node, in_node)] if (out_node, in_node) in node_dict.keys() else [in_node, out_node]


def pairs_s_old(current_sn, tree, g, s, t):
    sub_nodes = [(tuple(intersection(current_sn[1].networkx_graph().nodes, neighbor_sn[1].networkx_graph().nodes)),
                  set(nodes_of_sn(neighbor_sn, current_sn, tree))) for neighbor_sn in
                 tree.networkx_graph().neighbors(current_sn)]
    node_dict = dict(sub_nodes)

    sp_nodes = list(current_sn[1].networkx_graph().nodes)
    local_g = g.subgraph(sp_nodes).copy()

    local_g.add_edges_from(
        [(x, y) for x, y in node_dict.keys() if ((x, y) not in local_g.edges and (y, x) not in local_g.edges)])

    possible_s = [(x, y) for (x, y), nodes in sub_nodes if x != s and y != s and s in nodes]
    possible_t = [(x, y) for (x, y), nodes in sub_nodes if x != t and y != t and t in nodes]

    xs, ys = -1, -1
    xt, yt = -1, -1
    if possible_s:
        xs, ys = possible_s[0]
        local_g.add_edge("start", xs)
        local_g.add_edge("start", ys)
        if local_g.has_edge(xs, ys):
            local_g.remove_edge(xs, ys)
        else:
            local_g.remove_edge(ys, xs)
        s = "start"

    if possible_t:
        xt, yt = possible_t[0]
        if xt == xs and yt == ys:
            return []
        local_g.add_edge(xt, "end")
        local_g.add_edge(yt, "end")
        if local_g.has_edge(xt, yt):
            local_g.remove_edge(xt, yt)
        else:
            local_g.remove_edge(yt, xt)
        t = "end"

    paths = list(nx.edge_disjoint_paths(local_g, s, t))

    nodes_in_paths = [
        diff(flatten([comp_nodes_for_path(path[i], path[i + 1], node_dict) for i in range(len(path) - 1)]),
             (s, t, xs, ys, xt, yt)) for path in nx.edge_disjoint_paths(local_g, s, t)]
    p1_nodes = nodes_in_paths[0]
    p2_nodes = nodes_in_paths[1]

    pairs = [(n1, n2) if n1 < n2 else (n2, n1) for n1, n2 in itertools.product(p1_nodes, p2_nodes)]
    return pairs


def pairs_spqr_old(current_sn, tree, g, s, t):
    if current_sn[0] == 'S':
        pairs = pairs_s_old(current_sn, tree, g, s, t)
        return pairs
    if current_sn[0] == 'P':
        return pairs_p_old(current_sn, tree, g, s, t)
    return []


def get_max_nodes_spqr_old(component, in_node, out_node, x_filter=False, y_filter=False):
    # print(f"s:{s}, t:{t}")
    comp_sage = Graph(component)
    tree = spqr_tree(comp_sage)
    pairs = get_all_spqr_pairs_old(tree, component, in_node, out_node)
    # COMMON.pairs_idk = pairs
    res = max_disj_set_upper_bound(component.nodes, pairs, x_filter, y_filter, component)
    # print('ret', res)
    return res


# def get_max_nodes_spqr_y(component, in_node, out_node):
#     # print(f"s:{s}, t:{t}")
#     comp_sage = Graph(component)
#     tree = spqr_tree(comp_sage)
#     pairs = get_all_spqr_pairs(tree, component, in_node, out_node)
#     res = max_disj_set_upper_bound(component.nodes, pairs, False, True, component)
#     # print('ret', res)
#     return res
