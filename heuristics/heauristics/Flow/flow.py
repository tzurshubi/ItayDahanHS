import networkx as nx

from helpers.helper_funcs import get_dis_pairs, max_disj_set_upper_bound, get_vertex_disjoint_directed


def get_max_nodes(component, in_node, out_node, algorithm):
    good_pairs = set()
    for path in nx.node_disjoint_paths(component, in_node, out_node):
        p = len(path)
        for i in range(p):
            for j in range(i, p):
                good_pairs.add((path[i], path[j]))
#     r_pairs = get_all_r_pairs(component)
#     good_pairs.update(r_pairs)
    possible_pairs = get_dis_pairs(in_node, out_node, component.nodes, good_pairs)  ### NOT REALLY DISJOINT
    pairs = [(x1, x2) for x1, x2 in possible_pairs if
             (not algorithm(in_node, x1, x2, out_node, component)
              and not algorithm(in_node, x2, x1, out_node, component))]
    res = max_disj_set_upper_bound(component.nodes, pairs)
    return res

def has_flow(s, x, y, t, g):
    # print(list(g.nodes))
    g = get_vertex_disjoint_directed(g)
    g.add_edge("flow_start", str(s) + "out", capacity=1)
    g.add_edge("flow_start", str(x) + "out", capacity=2)
    g[str(s) + 'in'][str(s) + 'out']['capacity'] = 0
    g[str(x) + 'in'][str(x) + 'out']['capacity'] = 0
    g[str(y) + 'in'][str(y) + 'out']['capacity'] = 0
    g[str(t) + 'in'][str(t) + 'out']['capacity'] = 0
    g.add_edge(str(t) + "in", "flow_end", capacity=1)
    g.add_edge(str(y) + "in", "flow_end", capacity=2)
    flow_value, flow_dict = nx.maximum_flow(g, "flow_start", "flow_end")
    g.remove_edge("flow_start", str(s) + "out")
    g.remove_edge("flow_start", str(x) + "out")
    g.remove_edge(str(t) + "in", "flow_end")
    g.remove_edge(str(y) + "in", "flow_end")
    # if flow_value != 3:
    #     print(f"s: {index_to_node[s]} \tx: {index_to_node[x]} \ty: {index_to_node[y]} \tt: {index_to_node[t]} \tflow: {flow_value}")
    return flow_value == 3