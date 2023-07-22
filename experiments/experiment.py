# maze tests
from datetime import datetime
import os
import csv
from Definitions.state import State
from algorithms.search_algorithms.actual_path.with_spqr.run_other import run_other
from algorithms.search_algorithms.actual_path.with_spqr.with_spqr import get_comp_path
from algorithms.search_algorithms.a_star.run_weighted_astar import run_weighted
from experiments.experiment_helpers import write_header_file, write_to_file, write_to_csv_file, save_heuristic_plot, \
    save_graph_picture, convert_to_latex, create_graphs_from_folder, heuristics_lib, snake_heuristics_lib, \
    other_heuristic_lib
from helpers.COMMON import SNAKE_MODE, LSP_MODE, GRIDS_MODE, MAZE_MODE, CUBE_MODE
from helpers.graph_builder_funcs import generate_hard_grids, generate_hypercube, generate_aaai_showcase, generate_grid, \
    generate_rooms, build_heuristic_showcase, graph_for_grid, generate_aaai_showcase_2, generate_og_maze, \
    generate_aaai_showcase_original
from helpers.helper_funcs import diff, bcc_thingy, intersection, remove_blocks_2, remove_blocks, draw_grid

CUTOFF = -1
TIMEOUT = 15000

runs_per_params = 1
weights = [1]  # [0.7 + 0.1 * i for i in range(6)]
# grid_sizes = [(10 * i, 10 * i) for i in range(1, 3)]
# block_ps = [0.1 * i for i in range(4, 6)]




def search_experiment(graph=-9, start_node=-9, target_node=-9, world=GRIDS_MODE, n=None, algorithm=run_weighted):
    global best_path, best_path_len
    print(f'n: {n}')
    header = ['Graph', 'Grid Size', 'Heuristic', 'path', 'first_hs', 'Expansions', 'Runtime', 'Generated Nodes']
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
    # text_file_name = f'{directory}/test_map_{date_string}.txt'
    latex_file_name = f'{directory}/test_map_latex_{date_string}.txt'
    write_header_file(csv_file_name, header)
    #     exp_flag = False
    #     mat = box
    #     mats = [mat]
    #     for i in range(10):
    #         mat = remove_blocks_rectangles_2(k, mat, og_maze)
    #         mats += [mat]

    if world == GRIDS_MODE:
        graphs = [generate_aaai_showcase()]
        #--------------------------------------------------------
        # grid_n = 15
        # graphs = create_graphs_from_folder('/mnt/c/Users/itay/Desktop/notebooks/all_graphs/grids_15', grid_n, grid_n,
        #                                    mode=mode)
        #---------------------------------------------
        # for bp in block_ps:
        #     for n, m in grid_sizes:
        #         graphs += [(bp,) + g for g in generate_hard_grids(runs_per_params, n, m, bp)]
    elif world == MAZE_MODE:
        # if mode == LSP_MODE:
        graphs, snake_graphs = generate_rooms(k=1, n=1)
        run_experiment(graphs, heuristics_lib, save_dir, csv_file_name, plot_dir, graph_dir, latex_file_name, world=world, mode=LSP_MODE, n=n, algorithm=algorithm)
        run_experiment(snake_graphs, snake_heuristics_lib, save_dir, csv_file_name, plot_dir, graph_dir, latex_file_name, world=world, mode=SNAKE_MODE, n=n, algorithm=algorithm)
        run_experiment(graphs, other_heuristic_lib, save_dir, csv_file_name, plot_dir, graph_dir, latex_file_name,
                       world=world, mode=LSP_MODE, n=n, algorithm=run_other)

        # # grid_n = len(mat)
            # # og_mat = mat.copy()
            # k=5
            # # graphs = []
            # graphs = [[name, mat, graph, start_node, target_node, index_to_node]]
            # for i in range(5):
            #     mat = remove_blocks(k, mat)
            #     temp_graph = graph_for_grid(mat, index_to_node[start_node], index_to_node[target_node], mode=mode)
            #     graphs.append((f'{name}_{i}', ) + temp_graph)

        # elif mode == SNAKE_MODE:
        #     pass

    elif world == CUBE_MODE:
        graphs = [(0, 0, graph, start_node, target_node, 0)]

def run_experiment(graphs, heuristics, save_dir, csv_file_name, plot_dir, graph_dir, latex_file_name, world=GRIDS_MODE, mode=LSP_MODE, n=None, algorithm=run_weighted):
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
        hs_per_run[name] = dict()
        ls_per_run[name] = dict()
        expansions_per_run[name] = dict()
    # graph_i = 0
    for w in weights:
        print('w == ', w)
        for graph_i, mat, graph, start, target, itn in graphs:
            print(f"GRAPH {graph_i}:")
            for name, h, incremental in heuristics:
                print(f'running with {name} heuristic ')
                path, expansions, runtime, hs, ls, ns, ng = algorithm(h,
                                                                      graph,
                                                                      start,
                                                                      target,
                                                                      w,
                                                                      CUTOFF,
                                                                      TIMEOUT,
                                                                      incremental,
                                                                      save_dir=save_dir,
                                                                      n=n,
                                                                      mode=mode
                                                                      )
                print("path length: ", len(path) - 1, '\npath: ', path)
                draw_grid(name, graph, mat, start, target, itn, path=path)
                if algorithm != run_other:
                    print("path length: ", len(path) - 1, '\npath: ', path)
                    sum_path_lengths[name] += len(path) - 1
                    sum_expansions[name] += expansions
                    sum_runtimes[name] += runtime
                    hs_per_run[name][graph_i] = hs
                    first_hs = hs[0]
                    last_hs = hs[len(hs) - 1]
                    ls_per_run[name][graph_i] = ls
                    expansions_per_run[name][graph_i] = expansions
                else:
                    first_hs = -999
                if runtime > TIMEOUT:
                    write_to_csv_file(csv_file_name, graph_i, name, expansions, runtime, -9999, first_hs, len(path), ng)

                    # exp_flag = True
                #                 print(f"{name} {hs_per_run[name][graph_i]}")
                # print(
                #     f"\tNAME: {name}, \t\tPATH-LENGTH: {len(path)}, \t\tEXPANSIONS: {expansions} \t\tRUNTIME: {runtime}")
                if n is not None:
                    # write_to_file(text_file_name, n, name, graph, expansions, runtime, hs,
                    #               ls, -1, w, -1)
                    write_to_csv_file(csv_file_name, n, name, expansions, runtime, last_hs, w, first_hs, ng)
                else:
                    write_to_csv_file(csv_file_name, graph_i, name, expansions, runtime, -9999, first_hs, len(path), ng)
            if world in (GRIDS_MODE, MAZE_MODE) and algorithm != run_other:
                save_heuristic_plot(heuristics, plot_dir, graph_i, hs_per_run)
                save_graph_picture(graph_dir, graph_i, mat, graph, start, target, itn)

        convert_to_latex(csv_file_name, latex_file_name)


#





def mother_of_tests(algorithm=run_weighted, world=GRIDS_MODE, mode=LSP_MODE, n=None):
    header = ['Graph', 'Grid Size', 'Blocks', 'A* weight', 'Heuristic', 'Expansions', 'Runtime', 'Generated Nodes']
    avg_header = ['Grid Size', 'Blocks', 'A* weight', 'Heuristic', 'Expansions Avg', 'Runtime Avg',
                  'Generated Nodes Avg']
    #     write_header_file(raw_csv_file_name, header)
    if world == CUBE_MODE:
        cube, node_to_index = generate_hypercube(n)
        target = tuple([0] * n)
        start = tuple([0] * (n - 2) + [1, 1])
        search_experiment(cube, start, target, world=CUBE_MODE, n=n, algorithm=algorithm)
    else:
        search_experiment(world=world, algorithm=algorithm)
