from Definitions.state import State
from algorithms.search_algorithms.a_star.astar import max_weighted_a_star
from algorithms.algorithms_helpers import expand_with_constraints, expand_with_snake_constraints
from algorithms.general_algorithms import g
from helpers.COMMON import SNAKE_MODE, LSP_MODE
from helpers.graph_builder_funcs import generate_grids
from helpers.helper_funcs import diff


def get_goal_func(target):
    def goal_check_path(state):  # graph is original graph of nodes!
        return state.current == target

    return goal_check_path


def run_weighted(heuristic, graph, start, target, weight, cutoff, timeout, is_incremental, n=None, save_dir="", mode=LSP_MODE):
    if mode == SNAKE_MODE:
        print(f'n: {n}')
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
    # stron = (lambda x: ex_pairs(x, graph, target))
    expand_function = expand_with_snake_constraints if mode == SNAKE_MODE else expand_with_constraints
    end_state, data = max_weighted_a_star(graph,
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
                                          # save_dir=save_dir
                                         )
    expansions, runtime, nodes_extracted_heuristic_values, nodes_extracted_path_len, nodes_chosen, generated_nodes = data
    return end_state.path if end_state != -1 else [], expansions, runtime, nodes_extracted_heuristic_values, nodes_extracted_path_len, nodes_chosen, generated_nodes



def grid_setup(runs, height, width, block_p):
    return lambda: generate_grids(runs, height, width, block_p)
