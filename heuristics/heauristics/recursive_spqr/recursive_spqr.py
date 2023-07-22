
# recursive attempt
import itertools
import networkx as nx
from sage.graphs.connectivity import spqr_tree
from sage.graphs.graph import Graph

from heuristics.heuristics_helper_funcs import get_relevant_cuts
from helpers.helper_funcs import intersection, flatten, diff


def spqr_nodes(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict):
    if current_sn[0] == 'R':
        return nodes_r(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict)
    if current_sn[0] == 'P':
        return nodes_p(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict)
    if current_sn[0] == 'S':
        return nodes_s(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict)
    return []


def nodes_s(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict):
    sn_sp = [(neighbor_sn, sp_dict[(neighbor_sn, current_sn)]) for neighbor_sn in tree.networkx_graph().neighbors(current_sn) if neighbor_sn != parent_sn]
    super_n_nodes = [((i ,o), spqr_nodes(neighbor_sn, current_sn, tree, g, i, o, sp_dict)) for neighbor_sn, (i, o) in sn_sp]
    in_out_sn = [n for (i ,o), n in super_n_nodes if (i ,o) == (in_node, out_node)]
    ret = []
    if in_out_sn:
        print('iosn', in_node, out_node, in_out_sn)
        print(g.has_edge(in_node, out_node))
        # with open('D:/Heuristic Tests/improved_spqr_results/'+str(cur_t)+'weird_s.txt', "a+") as f:
        #     f.write(f'{pair_i} \nsource, target = {in_node, out_node} \nnodes = {list(g.nodes)}\nedges = {list(g.edges)} \n')
        #     f.write(f'itn = {str(index_to_node)}\n')
        #     f.write('\n\n')
        other_path = flatten([n for (i ,o), n in super_n_nodes if (i ,o) != (in_node, out_node)]) + list \
            (current_sn[1].networkx_graph().nodes)
        ret = max((in_out_sn[0] + [in_node, out_node], other_path), key=len)
    else:
        ret = flatten([n for (i ,o), n in super_n_nodes]) + list(current_sn[1].networkx_graph().nodes)
    ret = ret if not parent_sn else diff(ret, [in_node, out_node])
    #     print('s', ret)
    return ret


def nodes_p(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict):
    sn_sp = [(neighbor_sn, sp_dict[(neighbor_sn, current_sn)]) for neighbor_sn in tree.networkx_graph().neighbors(current_sn) if neighbor_sn != parent_sn]
    super_n_nodes = [spqr_nodes(neighbor_sn, current_sn, tree, g, i, o, sp_dict) for neighbor_sn, (i, o) in sn_sp]
    ret = max(super_n_nodes, key=len)
    ret = ret + [in_node, out_node] if not parent_sn else ret
    #     print('p', ret)
    return ret


def get_edge_nodes(e, sp_dict):
    return sp_dict[e] if e in sp_dict else sp_dict[(e[1], e[0])]


def nodes_r(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict):
    #     print('hi')
    sn_sp = [(neighbor_sn, sp_dict[(neighbor_sn, current_sn)]) for neighbor_sn in tree.networkx_graph().neighbors(current_sn) if neighbor_sn != parent_sn]
    super_n_nodes_dict = [((i ,o) ,spqr_nodes(neighbor_sn, current_sn, tree, g, i, o, sp_dict)) for neighbor_sn, (i, o) in sn_sp]
    super_n_nodes_dict = dict(super_n_nodes_dict)
    i_o_sn = super_n_nodes_dict[(in_node, out_node)] if (in_node, out_node) in super_n_nodes_dict.keys() else []
    super_n_nodes_dict.pop((in_node, out_node), None)

    #     with open('D:/Heuristic Tests/improved_spqr_results/'+str(cur_t)+'r_size.txt', "a+") as f:
    #         f.write(f'\tr -- {len(super_n_nodes_dict)}\n')

    # with open('D:/Heuristic Tests/improved_spqr_results/'+str(cur_t)+'r_neighbors_count.txt', "a+") as f:
    #         f.write(f'{len(super_n_nodes_dict)}\n')
    sp_nodes = list(current_sn[1].networkx_graph().nodes)
    #     print(sp_nodes)
    local_g = g.subgraph(sp_nodes).copy()

    local_g.add_edges_from([e for e in super_n_nodes_dict.keys() if e not in local_g.edges])

    # in_node edges
    in_edges = [e for e in super_n_nodes_dict.keys() if in_node in e]

    # out_node edges
    out_edges = [e for e in super_n_nodes_dict.keys() if out_node in e]

    if local_g.has_edge(in_node, out_node):
        local_g.remove_edge(in_node, out_node)

    # cut edges
    cut_edges = get_relevant_cuts(local_g, super_n_nodes_dict.keys())

    # easy nodes
    in_out_edges = in_edges + out_edges
    cut_maxes = [(max((e1, e2), key=lambda x: len(super_n_nodes_dict[x])), (e1 ,e2)) for e1 ,e2 in cut_edges]
    easy_cut_nodes = [(super_n_nodes_dict[e], (e1 ,e2)) for e ,(e1 ,e2) in cut_maxes if e not in in_out_edges]
    easy_nodes = [n for n, (e1, e2) in easy_cut_nodes]
    easy_nodes += [super_n_nodes_dict[e] for e in diff(super_n_nodes_dict.keys(), flatten(cut_edges) + in_out_edges)]

    # s_t_cuts = [(e1, e2) for e1, e2 in cut_edges if (e1 in in_edges and e2 in out_edges) or (e2 in in_edges and e1 in out_edges)]
    # if s_t_cuts:
    #     print(s_t_cuts)
    #     with open('D:/Heuristic Tests/improved_spqr_results/ ' +str(cur_t ) +'s_t_cut.txt', "a+") as f:
    #         f.write \
    #             (f'{s_t_cuts} \nsource, target = {in_node, out_node} \nnodes = {list(g.nodes)}\nedges = {list(g.edges)} \n')
    #         f.write(f'itn = {str(index_to_node)}\n')
    #         f.write('\n\n')

    # draw_grid('', 'rc', g, [[0]*20]*20, in_node, out_node, itn, path=local_g.nodes)
    # print('in , out - ', in_node, out_node)
    # print('edges -', list(local_g.edges))
    # print('cuts -', get_relevant_cuts(local_g, local_g.edges))
    # print('ve')
    # for k,v in super_n_nodes_dict.items():
    #     print(k, '\t', v)

    # exclusion nodes
    relevant_cuts = diff(cut_edges, [t[1] for t in easy_cut_nodes])

    in_cuts = [(e1, e2) for e1, e2 in relevant_cuts if (in_node in e1 or in_node in e2) and not (in_node in e1 and in_node in e2)]

    out_cuts = [(e1, e2) for e1, e2 in relevant_cuts if (out_node in e1 or out_node in e2) and not (out_node in e1 and out_node in e2)]

    c = get_max_comb(in_cuts, in_edges, super_n_nodes_dict) + get_max_comb(out_cuts, out_edges, super_n_nodes_dict)

    #     print(len(complement_exg.nodes))
    ret = max((flatten(easy_nodes + [super_n_nodes_dict[e] for e in c]) + sp_nodes, i_o_sn + [in_node, out_node]), key=len)
    ret = ret if not parent_sn else diff(ret, [in_node, out_node])
    #     print('r', ret)
    return ret


def get_max_comb(relevant_cuts, node_edges, super_n_nodes_dict):
    exg = nx.Graph()
    ex_edges = node_edges + flatten(relevant_cuts)
    exg.add_nodes_from(ex_edges)
    exg_pairs = list(itertools.combinations(node_edges, 2)) + relevant_cuts
    exg.add_edges_from(exg_pairs)

    # need to program it way more efficient
    complement_exg = nx.complement(exg)
    # print(len(complement_exg.nodes))
    for node in complement_exg:
        complement_exg.nodes[node]['l'] = len(super_n_nodes_dict[node])
    c, w = nx.max_weight_clique(complement_exg, weight='l')
    return c


def edge_seperators(tree):
    sp_dict = {}
    for (t1, g1), (t2, g2) in tree.networkx_graph().edges:
        sp = tuple(intersection(g1.networkx_graph().nodes, g2.networkx_graph().nodes))
        sp = min(sp), max(sp)
        sp_dict[((t1, g1), (t2, g2))] = sp
        sp_dict[((t2, g2), (t1, g1))] = sp
    return sp_dict


def find_root_sn(tree, s, t, sp_dict):
    # print(sorted([x for x in tree if (set([s, t]) & set(x[1].networkx_graph().nodes)) == set([s, t])], key=lambda x: 0 if x[0] == 'P' else 1))
    return sorted([x for x in tree if ({s, t} & set(x[1].networkx_graph().nodes)) == {s, t}], key=lambda x: 0 if x[0] == 'P' else 1)[0]


def get_max_nodes_spqr_recursive(component, in_node, out_node, return_nodes=False):
    # print(f"s:{s}, t:{t}")
    #     with open('D:/Heuristic Tests/improved_spqr_results/'+str(cur_t)+'r_size.txt', "a+") as f:
    #         f.write(f'\ncomp len -- {len(component)}\n')
    comp = component.copy()
    if len(comp.nodes) == 2:
        return comp.nodes if return_nodes else len(comp.nodes)
    if not comp.has_edge(in_node, out_node):
        #         print(f'adding {(in_node, out_node)}')
        comp.add_edge(in_node, out_node)
    comp_sage = Graph(comp)
    tree = spqr_tree(comp_sage)
    # for x in tree:
    #     print(x[0], list(x[1].networkx_graph().nodes))
    sp_dict = edge_seperators(tree)
    root_node = find_root_sn(tree, in_node, out_node, sp_dict)
    #     print(root_node)
    nodes = spqr_nodes(root_node, [], tree, comp, min(in_node, out_node),  max(in_node, out_node), sp_dict)
    # print('ret', res)
    return nodes if return_nodes else len(nodes)
