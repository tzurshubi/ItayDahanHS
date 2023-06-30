# maze tests
from datetime import datetime
import os
import csv
from Definitions.state import State
from algorithms.search_algorithms.actual_path.with_spqr.with_spqr import get_comp_path
from algorithms.search_algorithms.a_star.run_weighted_astar import run_weighted
from experiments.experiment_helpers import write_header_file, write_to_file, write_to_csv_file, save_heuristic_plot, \
    save_graph_picture, heuristics
from helpers.COMMON import SNAKE_MODE, LSP_MODE, GRIDS_MODE, MAZE_MODE, CUBE_MODE
from helpers.graph_builder_funcs import generate_hard_grids, generate_hypercube
from helpers.helper_funcs import diff, bcc_thingy, intersection
from heuristics.heuristics_interface_calls import snake_y_all_neighbors

CUTOFF = -1
TIMEOUT = 50

runs_per_params = 1
weights = [1]  # [0.7 + 0.1 * i for i in range(6)]
grid_sizes = [(10 * i, 10 * i) for i in range(1, 3)]
block_ps = [0.1 * i for i in range(4, 6)]




def search_experiment(graph=-9, start_node=-9, target_node=-9, world=GRIDS_MODE, n=None, mode=LSP_MODE, algorithm=run_weighted):
    global best_path, best_path_len
    header = ['Graph', 'Grid Size', 'Blocks', 'A* weight', 'Heuristic', 'Expansions', 'Runtime', 'Generated Nodes']
    start_time = datetime.now()
    date_string = start_time.strftime("%d-%m-%Y-%H-%M-%S")
    directory = '/mnt/d/Heuristic Tests/Experiments/Experiment' + date_string
    save_dir = directory + '/saved_state'
    graph_dir = directory + '/graphs'
    plot_dir = directory + '/plots'
    os.mkdir(directory)
    os.mkdir(graph_dir)
    os.mkdir(plot_dir)
    os.mkdir(save_dir)
    csv_file_name = f'{directory}/test_map_{date_string}.csv'
    text_file_name = f'{directory}/test_map_{date_string}.txt'
    latex_file_name = f'{directory}/test_map_latex_{date_string}.txt'
    write_header_file(csv_file_name, header)
    #     exp_flag = False
    #     mat = box
    #     mats = [mat]
    #     for i in range(10):
    #         mat = remove_blocks_rectangles_2(k, mat, og_maze)
    #         mats += [mat]
    graphs = []
    if world == GRIDS_MODE:
        for bp in block_ps:
            for n, m in grid_sizes:
                graphs += [(bp,) + g for g in generate_hard_grids(runs_per_params, n, m, bp)]
    elif world == MAZE_MODE:
        pass
    elif world == CUBE_MODE:
        graphs = [(0, 0, graph, start_node, target_node, 0)]
    i = 0
    runs = len(graphs)
    names = [name for name, h, _ in heuristics]
    sum_runtimes = dict.fromkeys(names, 0)
    sum_expansions = dict.fromkeys(names, 0)
    sum_path_lengths = dict.fromkeys(names, 0)
    hs_per_run = {}
    ls_per_run = {}
    expansions_per_run = {}
    for name, _, _ in heuristics:
        hs_per_run[name] = [0] * runs
        ls_per_run[name] = [0] * runs
        expansions_per_run[name] = [0] * runs
    graph_i = 0
    for w in weights:
        print('w == ', w)
        for bp, mat, graph, start, target, itn in graphs:
            print(f"GRAPH {graph_i}:")
            for name, h, incremental in heuristics:
                path, expansions, runtime, hs, ls, ns, ng = algorithm(h,
                                                                      graph,
                                                                      start,
                                                                      target,
                                                                      w,
                                                                      CUTOFF,
                                                                      TIMEOUT,
                                                                      incremental,
                                                                      save_dir,
                                                                      n,
                                                                      mode=mode
                                                                      )

                print("path length: ", len(path) - 1, '\npath: ', path)
                sum_path_lengths[name] += len(path) - 1
                sum_expansions[name] += expansions
                sum_runtimes[name] += runtime
                hs_per_run[name][graph_i] = hs
                first_hs = hs[0]
                last_hs = hs[len(hs) - 1]
                ls_per_run[name][graph_i] = ls
                expansions_per_run[name][graph_i] = expansions
                if runtime > TIMEOUT:
                    exp_flag = True
                #                 print(f"{name} {hs_per_run[name][graph_i]}")
                # print(
                #     f"\tNAME: {name}, \t\tPATH-LENGTH: {len(path)}, \t\tEXPANSIONS: {expansions} \t\tRUNTIME: {runtime}")
                if n is not None:
                    write_to_file(text_file_name, n, name, graph, expansions, runtime, hs,
                                  ls, -1, w, -1)
                    write_to_csv_file(csv_file_name, n, name, expansions, runtime, last_hs, w, first_hs, ng)
                else:
                    write_to_file(text_file_name, graph_i, name, mat, expansions, runtime, hs, ls, n, w, bp)
                    write_to_csv_file(csv_file_name, graph_i, name, expansions, runtime, n, w, bp, ng)
            if mode in (GRIDS_MODE, MAZE_MODE):
                save_heuristic_plot(plot_dir, graph_i, hs_per_run)
                save_graph_picture(graph_dir, graph_i, mat, graph, start, target, itn)
                graph_i += 1

        # convert_to_latex(csv_file_name, snake_latex_file_name)
#
def run_other(graph, start, target):
    start_available = tuple(diff(list(graph.nodes), {start}))
    start_path = (start,)
#     bcc_dict = {}
    state = State(start, start_path, start_available)
    _, _, relevant_comps, _, reach_nested, current_node = bcc_thingy(state, graph, target)
    if relevant_comps == -1:
        return -1  # no path
    n = len(relevant_comps)
    if n == 0:
        return 0
    cut_nodes = [(current_node, target)] if n == 1 else [(current_node, list(intersection(relevant_comps[0], relevant_comps[1]))[0])] + [(list(intersection(relevant_comps[i-1], relevant_comps[i]))[0], list(intersection(relevant_comps[i+1], relevant_comps[i]))[0]) for i in range(1,n-1)] + [(list(intersection(relevant_comps[n-2], relevant_comps[n-1]))[0], target)]
    subgraphs = [reach_nested.subgraph(comp) for comp in relevant_comps]
    paths = [get_comp_path(comp, in_node, out_node) for (in_node, out_node), comp in zip(cut_nodes, subgraphs)]
    path = [start]
    # for (in_node, out_node),comp in zip(cut_nodes, relevant_comps):
    #     print((in_node, out_node),comp)
    for p in paths:
        # draw_grid('', 'p', g1, [[0]*20]*20, source, target, itn, path= p)
        path += list(p)
    return tuple(set(path))


def mother_of_tests(algorithm=run_weighted, world=GRIDS_MODE, mode=LSP_MODE, n=None):
    header = ['Graph', 'Grid Size', 'Blocks', 'A* weight', 'Heuristic', 'Expansions', 'Runtime', 'Generated Nodes']
    avg_header = ['Grid Size', 'Blocks', 'A* weight', 'Heuristic', 'Expansions Avg', 'Runtime Avg',
                  'Generated Nodes Avg']
    #     write_header_file(raw_csv_file_name, header)
    if world == CUBE_MODE:
        cube, node_to_index = generate_hypercube(n)
        target = tuple([0] * n)
        start = tuple([0] * (n - 2) + [1, 1])
        search_experiment(cube, start, target, world=CUBE_MODE, n=n, mode=mode, algorithm=algorithm)
    else:
        search_experiment(world=world, mode=mode, algorithm=algorithm)
