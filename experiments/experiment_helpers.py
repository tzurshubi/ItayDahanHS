import csv

from matplotlib import pyplot as plt

# from experiments.experiment import heuristics
from helpers.helper_funcs import draw_grid
from heuristics.heauristics.other_heuristics import count_nodes_bcc_x, count_nodes_bcc_y, count_nodes_bcc
from heuristics.heuristics_interface_calls import snake_y_all_neighbors, snake_rec_spqr, snake_only, spqr_recursive_h, \
    ex_pairs_using_spqr

heuristics = [
        # ["bcc x", count_nodes_bcc_x, False],
        ["bcc",count_nodes_bcc , True],
        # ["bcc y incremental", count_nodes_bcc_y, True],
        # ["spqr", ex_pairs_using_spqr, True],
    #     ["snake spqr", snake_only, True],
    #     ["snake spqr y", snake_y, True],
    #     ["snake spqr y in neighbors", snake_y_in_neighbors, True],
    #     ["snake spqr y all neighbors", snake_y_all_neighbors, True],
    #     ["snake spqr y recursive", snake_rec_spqr, True],
    ["spqr", ex_pairs_using_spqr, True],
    ["spqr rec", spqr_recursive_h, True]

]
def build_mazes():
    pass


def generate_grids():
    pass

def write_header_file(file_name, header):
    with open(file_name, 'a', encoding='UTF8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)


def write_to_file(file_name, graph_i, h_name, graph_mat, expansions, runtime, hs, ls, grid_n, astar_w, block_p):
    with open(file_name, 'a') as f:
        f.write(str((graph_i, grid_n, astar_w, block_p, h_name, graph_mat, expansions, runtime, hs, ls)))
        f.write('\n')


def write_to_csv_file(file_name, graph_i, h_name, expansions, runtime, grid_n, astar_w, block_p, generated_nodes):
    with open(file_name, 'a', encoding='UTF8') as csv_file:
        writer = csv.writer(csv_file)
        row = [str(x) for x in [graph_i, grid_n, block_p, astar_w, h_name, expansions, runtime, generated_nodes]]
        writer.writerow(row)



def save_graph_picture(folder_name, pic_name, mat, graph, start, target, itn):
    draw_grid(folder_name, pic_name, graph, mat, start, target, itn)


def save_heuristic_plot(folder_name, graph_i, hs_per_run):
    fig, ax = plt.subplots()
    for name, _, _ in heuristics:
        # print(name, hs_per_run[name][graph_i])
        ax.plot(range(len(hs_per_run[name][graph_i])), hs_per_run[name][graph_i], label=f"hs - {name}")
        # plt.plot(range(len(pl_per_run[name][graph_i])), pl_per_run[name][graph_i], label=f"pl - {name}")
    plt.title('graph ' + str(graph_i))
    plt.legend()
    # plt.show()
#     save_results_to = 'D:/Heuristic Tests/scatters7/'

    fig.savefig(folder_name + f'/scatter_{str(graph_i)}.png')
