import time as t
from algorithms.general_algorithms import ex_pairs
from algorithms.incremental_algorithms import ex_pairs_incremental
from helpers.helper_funcs import max_disj_set_upper_bound
from helpers.index_to_node_stuff import index_to_node
from heuristics.heauristics.naive_spqr.naive_spqr import get_max_nodes_spqr_new
from heuristics.heauristics.recursive_spqr.recursive_spqr import get_max_nodes_spqr_recursive
from heuristics.heauristics.snake_spqr.snake_new_spqr.snake_spqr import get_max_nodes_spqr_snake

pair_i = 0
good_bcc = 0
cur_t = t.time()


def is_legal(nodes, pairs):
    for node in nodes:
        for u, v in pairs:
            if u == node and v in nodes:
                return False
            elif v == node and u in nodes:
                return False
    return True


def get_max_test(comp, in_node, out_node):
    global pair_i
    global good_bcc

    h1 = snake_exclusion_pairs_spqr(comp, in_node, out_node, False, True, True, True)
    h2 = get_max_nodes_spqr_snake(comp, in_node, out_node, False, True, True, True)

    res_new = h1
    res_basic = h2

    # print('------')
    # print('2 - 3', diff(res2, res3))
    # print('3 - 2', diff(res3, res2))
    with open('/mnt/d/Heuristic Tests/recursive_spqr_results/' + str(cur_t) + 'index.txt', "a+") as f:
        f.write(f'{pair_i}\n')
        f.write('\n\n')

    # if not is_legal(nodes, pairs):
    #     with open('/mnt/d/Heuristic Tests/recursive_spqr_results/' + str(cur_t) + 'not_supposed_to_happen.txt', "a+") as f:
    #         f.write(
    #             f'{pair_i} \nsource, target = {in_node, out_node} \nnodes = {list(comp.nodes)}\nedges = {list(comp.edges)} \n')
    #         f.write(f'itn = {str(index_to_node)}\n')
    #         f.write('\n\n')

    if res_basic > res_new:
        with open('/mnt/d/Heuristic Tests/recursive_spqr_results/' + str(cur_t) + 'woops.txt', "a+") as f:
            f.write(
                f'{pair_i} \nsource, target = {in_node, out_node} \nnodes = {list(comp.nodes)}\nedges = {list(comp.edges)} \n')
            f.write(f'itn = {str(index_to_node)}\n')
            f.write('\n\n')

    if res_basic < res_new:
        good_bcc += 1
        with open('/mnt/d/Heuristic Tests/recursive_spqr_results/' + str(cur_t) + 'yas.txt', "a+") as f:
            f.write(
                f'{good_bcc},  {good_bcc / pair_i} \nsource, target = {in_node, out_node} \nnodes = {list(comp.nodes)}\nedges = {list(comp.edges)} \n')
            f.write(f'itn = {str(index_to_node)}\n')
            f.write('\n\n')

    pair_i += 1

    return res_basic


def test_new_spqr(state, G, target, is_incremental=False):
    if is_incremental:
        return ex_pairs_incremental(state, G, target, lambda g, i, o: get_max_test(g, i, o))
    return ex_pairs(state, G, target, lambda g, i, o: get_max_test(g, i, o))
