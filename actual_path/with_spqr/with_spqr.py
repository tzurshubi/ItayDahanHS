# different approach recursive
import networkx as nx

from heuristics.heuristics_helper_funcs import edge_seperators, find_root_sn
from heuristics.heuristics_interface_calls import spqr_recursive_h
from algorithms.run_weighted_astar import run_weighted
from helpers.helper_funcs import diff, flatten


def get_sub_search_graph(s, t, sub_g, edge_to_nodes):
    sub_g = sub_g.copy()
    for (x1, x2), nodes in edge_to_nodes.items():
        if sub_g.has_edge(x1, x2):
            sub_g.remove_edge(x1, x2)
        nx.add_path(sub_g, [x1] + nodes + [x2])
        # draw_grid('', 'p add', g1, [[0]*20]*20, source, target, itn, path=[x1] + nodes + [x2])

    for node in sub_g:
        sub_g.nodes[node]['constraint_nodes'] = [node]
    return sub_g


def spqr_nodes_actual_path(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict):
    if current_sn[0] == 'R':
        return nodes_r_actual_path(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict)
    if current_sn[0] == 'P':
        return nodes_p_actual_path(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict)
    if current_sn[0] == 'S':
        return nodes_s_actual_path(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict)
    return []


def nodes_r_actual_path(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict):
    # print('hi')
    sn_sp = [(neighbor_sn, sp_dict[(neighbor_sn, current_sn)]) for neighbor_sn in
             tree.networkx_graph().neighbors(current_sn) if neighbor_sn != parent_sn]
    super_n_nodes_dict = [((i, o), spqr_nodes_actual_path(neighbor_sn, current_sn, tree, g, i, o, sp_dict)) for
                          neighbor_sn, (i, o) in sn_sp]
    super_n_nodes_dict = dict(super_n_nodes_dict)
    super_n_nodes_dict.pop((in_node, out_node), None)

    #     with open('D:/Heuristic Tests/improved_spqr_results/'+str(cur_t)+'r_size.txt', "a+") as f:
    #         f.write(f'\tr -- {len(super_n_nodes_dict)}\n')

    sp_nodes = list(current_sn[1].networkx_graph().nodes)
    local_g = g.subgraph(sp_nodes).copy()
    local_g.add_edges_from([e for e in super_n_nodes_dict.keys() if e not in local_g.edges])

    graph = get_sub_search_graph(in_node, out_node, local_g, super_n_nodes_dict)
    path = run_weighted(spqr_recursive_h, graph, in_node, out_node, 1, 50000, 2000, True)[0]
    path = path if not parent_sn else diff(path, [in_node, out_node])
    return path


def nodes_s_actual_path(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict):
    sn_sp = [(neighbor_sn, sp_dict[(neighbor_sn, current_sn)]) for neighbor_sn in
             tree.networkx_graph().neighbors(current_sn) if neighbor_sn != parent_sn]
    super_n_nodes = [((i, o), spqr_nodes_actual_path(neighbor_sn, current_sn, tree, g, i, o, sp_dict)) for
                     neighbor_sn, (i, o) in sn_sp]
    in_out_sn = [n for (i, o), n in super_n_nodes if (i, o) == (in_node, out_node)]
    ret = []
    if in_out_sn:
        print('iosn', in_node, out_node, in_out_sn)
        print(g.has_edge(in_node, out_node))
        # with open('D:/Heuristic Tests/improved_spqr_results/'+str(cur_t)+'weird_s.txt', "a+") as f:
        #     f.write(f'{pair_i} \nsource, target = {in_node, out_node} \nnodes = {list(g.nodes)}\nedges = {list(g.edges)} \n')
        #     f.write(f'itn = {str(index_to_node)}\n')
        #     f.write('\n\n')
        other_path = flatten([n for (i, o), n in super_n_nodes if (i, o) != (in_node, out_node)]) + list(
            current_sn[1].networkx_graph().nodes)
        ret = max((in_out_sn[0] + [in_node, out_node], other_path), key=len)
    else:
        ret = flatten([n for (i, o), n in super_n_nodes]) + list(current_sn[1].networkx_graph().nodes)
    ret = ret if not parent_sn else diff(ret, [in_node, out_node])
    #     print('s', ret)
    return ret


def nodes_p_actual_path(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict):
    sn_sp = [(neighbor_sn, sp_dict[(neighbor_sn, current_sn)]) for neighbor_sn in
             tree.networkx_graph().neighbors(current_sn) if neighbor_sn != parent_sn]
    super_n_nodes = [spqr_nodes_actual_path(neighbor_sn, current_sn, tree, g, i, o, sp_dict) for neighbor_sn, (i, o) in
                     sn_sp]
    ret = max(super_n_nodes, key=len)
    ret = ret + [in_node, out_node] if not parent_sn else ret
    #     print('p', ret)
    return ret


def get_comp_path(component, in_node, out_node):
    # print(f"s:{s}, t:{t}")
    #     with open('D:/Heuristic Tests/improved_spqr_results/'+str(cur_t)+'r_size.txt', "a+") as f:
    #         f.write(f'\ncomp len -- {len(component)}\n')
    comp = component.copy()
    if len(comp.nodes) == 2:
        return comp.nodes
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
    path = spqr_nodes_actual_path(root_node, [], tree, comp, min(in_node, out_node), max(in_node, out_node), sp_dict)
    # print('ret', res)
    return path
