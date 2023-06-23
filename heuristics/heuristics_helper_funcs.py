import itertools
import networkx as nx
from helpers.helper_funcs import intersection


def get_relevant_cuts(graph, possible_edges):
    cut_edges = [(e1, e2) for e1, e2 in itertools.combinations(possible_edges, 2) if is_edge_cut((e1, e2), graph)]
    return cut_edges


def is_edge_cut(edges, graph):
    g = graph.copy()
    for u, v in edges:
        g.remove_edge(u, v)
    return not nx.is_connected(g)


def find_root_sn(tree, s, t):
    return [x for x in tree if ({s, t} & set(x[1].networkx_graph().nodes)) == {s, t}][0]


def edge_seperators(tree):
    sp_dict = {}
    for (t1, g1), (t2, g2) in tree.networkx_graph().edges:
        sp = tuple(intersection(g1.networkx_graph().nodes, g2.networkx_graph().nodes))
        sp = min(sp), max(sp)
        sp_dict[((t1, g1), (t2, g2))] = sp
        sp_dict[((t2, g2), (t1, g1))] = sp
    return sp_dict


def get_edge_nodes(e, sp_dict):
    return sp_dict[e] if e in sp_dict else sp_dict[(e[1], e[0])]
