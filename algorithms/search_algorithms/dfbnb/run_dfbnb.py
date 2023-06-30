from Definitions.state import State
from algorithms.algorithms_helpers import expand_with_snake_constraints, expand_with_constraints
from algorithms.general_algorithms import g
from algorithms.search_algorithms.a_star.run_weighted_astar import get_goal_func
from algorithms.search_algorithms.dfbnb.dfbnb import max_dfbnb_iterative
from helpers.COMMON import LSP_MODE, SNAKE_MODE
from helpers.helper_funcs import diff


def run_dfbnb(heuristic, graph, start, target, weight, cutoff, timeout, is_incremental, save_dir, n=None, mode=LSP_MODE):
    print('running dfbnb')
    if mode == SNAKE_MODE:
        first_node = tuple([0] * (n - 1) + [1])
        not_availables = diff([first_node] + list(graph.neighbors(first_node)), [start, target])
        start_available = tuple(diff(list(graph.nodes), not_availables))
    else:
        start_available = tuple(diff(list(graph.nodes), {start}))
    start_path = (start,)
#     bcc_dict = {}
    start_state = State(start, start_path, start_available)
    start_state.update_dimension(2)
    h = (lambda x: heuristic(x, graph, target, is_incremental))
    expand_function=expand_with_snake_constraints if mode==SNAKE_MODE else expand_with_constraints
    # stron = (lambda x: ex_pairs(x, graph, target))
    end_state, data = max_dfbnb_iterative(graph,
                                start_state,
                                get_goal_func(target),
                                h,
                                g,
                                is_incremental,
                                weight=weight,
                                cutoff=cutoff,
                                timeout=timeout,
                                hypercube_dimension=n,
                                expand=expand_function,
                                save_dir=save_dir
                                )
    expansions, runtime, nodes_extracted_heuristic_values, nodes_extracted_path_len, nodes_chosen, generated_nodes = data
    return end_state.path if end_state != -1 else [], expansions, runtime, nodes_extracted_heuristic_values, nodes_extracted_path_len, nodes_chosen, generated_nodes

