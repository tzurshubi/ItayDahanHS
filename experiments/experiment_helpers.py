import csv
import os

from matplotlib import pyplot as plt

from helpers.COMMON import LSP_MODE
from helpers.graph_builder_funcs import parse_graph_png, crop_and_parse_graph
# from experiments.experiment import heuristics
from helpers.helper_funcs import draw_grid
from heuristics.heauristics.other_heuristics import count_nodes_bcc_x, count_nodes_bcc_y, count_nodes_bcc
from heuristics.heuristics_interface_calls import snake_y_all_neighbors, snake_rec_spqr, snake_only, spqr_recursive_h, \
    ex_pairs_using_spqr

heuristics = [
        # ["bcc x", count_nodes_bcc_x, False],
        ["bcc", count_nodes_bcc , False],
        ["bcc inc", count_nodes_bcc , True],
        # ["bcc y incremental", count_nodes_bcc_y, True],
        # ["spqr", ex_pairs_using_spqr, True],
    #     ["snake spqr", snake_only, True],
    #     ["snake spqr y", snake_y, True],
    #     ["snake spqr y in neighbors", snake_y_in_neighbors, True],
    #     ["snake spqr y all neighbors", snake_y_all_neighbors, Tru
    #
    #
    #     e],
        # ["snake spqr y recursive", snake_rec_spqr, True],
    ["old spqr", ex_pairs_using_spqr, True],
    ["new spqr", ex_pairs_using_spqr, True],
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


def write_to_csv_file(file_name, graph_i, h_name, expansions, runtime, grid_n, first_hs, path_len, generated_nodes):
    with open(file_name, 'a', encoding='UTF8') as csv_file:
        writer = csv.writer(csv_file)
        row = [str(x) for x in [graph_i, grid_n, h_name, path_len, first_hs, expansions, runtime, generated_nodes]]
        writer.writerow(row)



def save_graph_picture(folder_name, pic_name, mat, graph, start, target, itn):
    draw_grid(pic_name, graph, mat, start, target, itn, folder_name=folder_name)

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

def convert_to_latex(file_name_in, file_name_out):
    with open(file_name_out, 'a') as output_file:
        with open(file_name_in, 'r') as csv_file:
            reader = csv.reader(csv_file)
            s_i, s_f_val, s_h_val, s_name, s_expansions, s_runtime, b_i, b_f_val, b_h_val, b_name, b_expansions, b_runtime, inc_runtime = 0,0,0,0,0,0, -1,-1,-1,-1,-1,-1,-1
            for row in reader:
                name = row[4]
                if name == 'spqr incremental':
                    s_i, s_f_val, s_h_val, s_name, s_expansions, s_runtime = row[0], row[1], row[2], row[4], row[5], \
                                                                             row[6]
                elif name == 'bcc incremental':
                    inc_runtime = row[6]
                else:
                    b_i, b_f_val, b_h_val, b_name, b_expansions, b_runtime = row[0], row[1], row[2], row[4], row[5], \
                                                                             row[6]
                if s_i == b_i and s_f_val == b_f_val:
                    line = f"& {str(5*int(s_i))} & {s_f_val} & {b_expansions} & {b_h_val} & {round(float(b_runtime), 2)} & {round(float(inc_runtime), 2)} & {s_expansions} & {s_h_val} & {round(float(s_runtime), 2)} \\\\"
                    output_file.write(line)
                    output_file.write('\n')

def create_graphs_from_folder(folder_name, rows, cols, mode=LSP_MODE):
    graphs = []
    for filename in os.listdir(folder_name):
        f = os.path.join(folder_name, filename)
        graph_name = filename.split('.')[0]
        graphs.append((graph_name,) + (crop_and_parse_graph(f, rows, cols, mode=mode)))
    return graphs

