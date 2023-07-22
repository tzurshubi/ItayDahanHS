# graph builder
import os
import random
import shutil

# import cv2
# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import networkx as nx

from experiments.tests.grid_cutter import find_largest_rectangle
from helpers import index_to_node_stuff
from helpers.COMMON import LSP_MODE, SNAKE_MODE
from helpers.helper_funcs import flatten, draw_grid, remove_blocks, remove_blocks_2, remove_blocks_rectangles_2
from helpers.index_to_node_stuff import update_index_to_node


def build_room_graph():
    graph = nx.Graph()
    node_index = 0
    index_to_node = {}
    # set up edges
    for i in range(9):
        for j in range(14):
            graph.add_node(node_index)
            index_to_node[(i, j)] = node_index
            if i > 0:
                graph.add_edge(node_index, index_to_node[(i - 1, j)])
            if j > 0:
                graph.add_edge(node_index, index_to_node[(i, j - 1)])
            node_index += 1

    removed_nodes = [
                        (3, 0), (4, 0),
                        (4, 1), (7, 1),
                        (1, 2), (2, 2), (3, 2),
                        (0, 4), (1, 4), (4, 4), (5, 4), (6, 4), (8, 4),
                        (4, 5),
                        (0, 6), (4, 6),
                        (4, 11),
                        (2, 13), (4, 13), (7, 13)
                    ] + [(i, 9) for i in range(8)]

    remove_nodes_indexes = [index_to_node[node] for node in removed_nodes]
    graph.remove_nodes_from(remove_nodes_indexes)
    for node in graph.nodes:
        graph.nodes[node]["constraint_nodes"] = [node]

    return [(graph, index_to_node[(0, 0)], index_to_node[(8, 13)])]


def build_small_grid():
    global index_to_node
    # graph = nx.grid_2d_graph(5, 5)

    graph = nx.Graph()
    node_index = 0
    index_to_node = {}
    node_to_index = {}
    # set up edges
    for i in range(5):
        for j in range(5):
            graph.add_node(node_index)
            node_to_index[(i, j)] = node_index
            index_to_node[node_index] = (i, j)
            if i > 0:
                graph.add_edge(node_index, node_to_index[(i - 1, j)])
            if j > 0:
                graph.add_edge(node_index, node_to_index[(i, j - 1)])
            node_index += 1

    removed_nodes = [(0, 1), (1, 1), (1, 3), (3, 3)]
    # mat = [[1 if (j, i) in removed_nodes else 0 for i in range (5)] for j in range(5)]

    remove_nodes_indexes = [node_to_index[node] for node in removed_nodes]
    graph.remove_nodes_from(remove_nodes_indexes)

    for node in graph.nodes:
        graph.nodes[node]["constraint_nodes"] = [node]

    update_index_to_node(index_to_node)
    return 'small grid', graph, node_to_index[(0, 0)], node_to_index[(4, 4)]


def build_small_grid2():
    # graph = nx.grid_2d_graph(5, 5)

    graph = nx.Graph()
    node_index = 0
    index_to_node = {}
    node_to_index = {}
    # set up edges
    for i in range(5):
        for j in range(5):
            graph.add_node(node_index)
            node_to_index[(i, j)] = node_index
            index_to_node[node_index] = (i, j)
            if i > 0:
                graph.add_edge(node_index, node_to_index[(i - 1, j)])
            if j > 0:
                graph.add_edge(node_index, node_to_index[(i, j - 1)])
            node_index += 1

    removed_nodes = [(0, 1), (1, 1), (1, 3), (3, 3)]
    mat = [[1 if (j, i) in removed_nodes else 0 for i in range(5)] for j in range(5)]

    remove_nodes_indexes = [node_to_index[node] for node in removed_nodes]
    graph.remove_nodes_from(remove_nodes_indexes)

    for node in graph.nodes:
        graph.nodes[node]["constraint_nodes"] = [node]

    return mat, graph, node_to_index[(0, 0)], node_to_index[(4, 4)], index_to_node, node_to_index


def build_small_grid_test():
    # graph = nx.grid_2d_graph(5, 5)

    graph = nx.Graph()
    node_index = 0
    index_to_node = {}
    node_to_index = {}
    # set up edges
    for i in range(5):
        for j in range(5):
            graph.add_node(node_index)
            node_to_index[(i, j)] = node_index
            index_to_node[node_index] = (i, j)
            if i > 0:
                graph.add_edge(node_index, node_to_index[(i - 1, j)])
            if j > 0:
                graph.add_edge(node_index, node_to_index[(i, j - 1)])
            node_index += 1

    removed_nodes = [(0, 1), (1, 1), (1, 3), (3, 3)]
    mat = [[1 if (j, i) in removed_nodes else 0 for i in range(5)] for j in range(5)]

    remove_nodes_indexes = [node_to_index[node] for node in removed_nodes]
    graph.remove_nodes_from(remove_nodes_indexes)

    for node in graph.nodes:
        graph.nodes[node]["constraint_nodes"] = [node]

    return [(graph, node_to_index[(0, 0)], node_to_index[(4, 4)], index_to_node, node_to_index)]


def build_heuristic_showcase(n):
    s = 0
    t = 4 * n
    path_line = list(range(1, int(n / 2) + 1))
    short_path_line1 = list(range(1 + int(n / 2), int(n / 2 + n / 8)))
    short_path_line2 = list(range(int(n / 2 + n / 8), int(n / 2 + n / 4)))
    blocky = list(range(int(n / 2 + n / 4), n))
    #     print(path_line)
    #     print(short_path_line1)
    #     print(short_path_line2)
    #     print(blocky)
    g = nx.Graph()
    for i in list(range(n + int(n / 8))) + [4 * n]:
        g.add_node(i, constraint_nodes=[i])
    for arr in [path_line, short_path_line1, short_path_line2]:
        for i in range(len(arr) - 1):
            g.add_edge(arr[i], arr[i + 1])

    for i in range(len(blocky)):
        for j in range(i + 1, len(blocky)):
            g.add_edge(blocky[i], blocky[j])

    g.add_edge(s, path_line[0])
    g.add_edge(s, blocky[0])
    g.add_edge(blocky[-1], short_path_line1[0])
    g.add_edge(blocky[-1], short_path_line2[0])
    g.add_edge(short_path_line1[-1], t)
    g.add_edge(short_path_line2[-1], t)
    g.add_edge(path_line[-1], t)

    return f'showcase {n}', g, s, t


def generate_random_grid(height, width, block_p):
    grid = [[(0 if random.uniform(0, 1) > block_p else 1) for i in range(width)] for j in range(height)]
    return generate_grid(grid)


def generate_grid(grid):
    height = len(grid)
    width = len(grid[0])
    path = []
    graph = nx.Graph()
    index_to_node = {}
    node_index = 0
    node_to_index = {}
    # set up edges
    for i in range(height):
        for j in range(width):
            if grid[i][j]:
                continue
            graph.add_node(node_index)
            node_to_index[(i, j)] = node_index
            index_to_node[node_index] = (i, j)
            if i > 0 and not grid[i - 1][j]:
                graph.add_edge(node_index, node_to_index[(i - 1, j)])
            if j > 0 and not grid[i][j - 1]:
                graph.add_edge(node_index, node_to_index[(i, j - 1)])
            node_index += 1

    # choose for path
    while True:
        indexes = range(len(graph.nodes))
        # print(indexes)
        start = list(graph.nodes)[random.choice(indexes)]
        target = list(graph.nodes)[random.choice(indexes)]
        if start == target:
            # print(start, target)
            continue
        try:
            path = nx.shortest_path(graph, source=start, target=target)
            break
        except:
            continue
    # print(path)

    for node in graph.nodes:
        graph.nodes[node]["constraint_nodes"] = [node]
    # print_mat(grid, index_to_node)
    return grid, graph, start, target, index_to_node


def generate_grids(num_of_runs, height, width, block_p):
    return [generate_random_grid(height, width, block_p) for i in range(num_of_runs)]


def generate_grid2(grid):
    height = len(grid)
    width = len(grid[0])
    path = []
    graph = nx.Graph()
    index_to_node = {}
    node_index = 0
    node_to_index = {}
    # set up edges
    for i in range(height):
        for j in range(width):
            if grid[i][j]:
                continue
            graph.add_node(node_index)
            node_to_index[(i, j)] = node_index
            index_to_node[node_index] = (i, j)
            if i > 0 and not grid[i - 1][j]:
                graph.add_edge(node_index, node_to_index[(i - 1, j)])
            if j > 0 and not grid[i][j - 1]:
                graph.add_edge(node_index, node_to_index[(i, j - 1)])
            node_index += 1

    # choose for path
    while True:
        indexes = range(len(graph.nodes))
        # print(indexes)
        start = list(graph.nodes)[random.choice(indexes)]
        target = list(graph.nodes)[random.choice(indexes)]
        if start == target:
            # print(start, target)
            continue
        try:
            path = nx.shortest_path(graph, source=start, target=target)
            break
        except:
            continue
    # print(path)

    for node in graph.nodes:
        graph.nodes[node]["constraint_nodes"] = [node]
    # print_mat(grid, index_to_node)
    return grid, graph, start, target, index_to_node, node_to_index


def generate_hard_grids(num_of_runs, height, width, block_p, is_snake=False):
    return [generate_hard_grid(height, width, block_p, is_snake) for i in range(num_of_runs)]


def generate_hard_grid(height, width, block_p, is_snake=False):
    while True:
        try:
            grid = [[(0 if random.uniform(0, 1) > block_p else 1) for i in range(width)] for j in range(height)]
            grid[0][0] = 0
            grid[height - 1][width - 1] = 0
            height = len(grid)
            width = len(grid[0])
            path = []
            graph = nx.Graph()
            index_to_node = {}
            node_index = 0
            node_to_index = {}
            # set up edges
            for i in range(height):
                for j in range(width):
                    if grid[i][j]:
                        continue
                    graph.add_node(node_index)
                    node_to_index[(i, j)] = node_index
                    index_to_node[node_index] = (i, j)
                    if i > 0 and not grid[i - 1][j]:
                        graph.add_edge(node_index, node_to_index[(i - 1, j)])
                    if j > 0 and not grid[i][j - 1]:
                        graph.add_edge(node_index, node_to_index[(i, j - 1)])
                    node_index += 1

            # choose for path
            n = len(index_to_node.keys())
            tries = 200
            start = node_to_index[(0, 0)]
            target = node_to_index[(height - 1, width - 1)]
            path = nx.shortest_path(graph, source=start, target=target)
            # print(path)

            for node in graph.nodes:
                graph.nodes[node]["constraint_nodes"] = list(graph.neighbors(node)) if is_snake else [node]
            break
        except:
            continue
    return grid, graph, start, target, index_to_node


def graph_for_grid(grid, start, target, mode=LSP_MODE):
    height = len(grid)
    width = len(grid[0])
    graph = nx.Graph()
    index_to_node = {}
    node_index = 0
    node_to_index = {}
    # set up edges
    for i in range(height):
        for j in range(width):
            if grid[i][j]:
                continue
            graph.add_node(node_index)
            node_to_index[(i, j)] = node_index
            index_to_node[node_index] = (i, j)
            if i > 0 and not grid[i - 1][j]:
                graph.add_edge(node_index, node_to_index[(i - 1, j)])
            if j > 0 and not grid[i][j - 1]:
                graph.add_edge(node_index, node_to_index[(i, j - 1)])
            node_index += 1
    start, target = node_to_index[start], node_to_index[target]
    try:
        nx.shortest_path(graph, source=start, target=target)
    except Exception:
        raise Exception('no path between start to target')

    for node in graph.nodes:
        graph.nodes[node]["constraint_nodes"] = list(graph.neighbors(node)) if mode==SNAKE_MODE else [node]
    return grid, graph, start, target, index_to_node


def parse_graph_png(path, rows, cols, mode=LSP_MODE, return_mat=False):
    p = (255, 0, 255, 0)
    start = ()
    target = ()
    img = Image.open(path).convert("CMYK")
    data = np.array(np.asarray(img))
    skip_row = data.shape[0] / rows
    skip_col = data.shape[1] / cols
    mat = []

    for j in range(cols):
        mat_row = []
        for i in range(rows):

            r = int(i * skip_row) + int(skip_row / 2)
            c = int(j * skip_col) + int(skip_col / 2)
            #         print(i,j,r,c)

            v = tuple(data[r, c])
            mat_row += [v]
            if (i,j) in [(0,0), (12,12)]:
                print(v)
            if v == (0, 255, 0, 0):
                start = (j, i)
            if v == (255, 0, 255, 0):
                target = (j, i)

            data[r, c] = p
            data[r, c + 1] = p
            data[r + 1, c] = p
            data[r + 1, c + 1] = p
        mat += [mat_row]
    mat = [[1 if m == (255, 255, 255, 0) else 0 for m in m_row] for m_row in mat]
    return mat if return_mat else graph_for_grid(mat, start, target, mode=mode)


def generate_maze(n, bs):
    mat = [[0] * n] * n
    indexes = flatten(
        [[(i, j) for i in range(n) if (i != 0 or j != 0) and (i != n - 1 or j != n - 1) and mat[i][j] == 0] for j in
         range(n)])
    random.shuffle(indexes)
    for i, j in indexes[:bs]:
        mat_temp = [[1 if i_ == i and j_ == j else mat[i_][j_] for i_ in range(n)] for j_ in range(n)]
        _, graph, _, _, _ = generate_grid(mat_temp)
        if len(list(nx.connected_components(graph))) == 1:
            mat = mat_temp
    grid, graph, start, target, index_to_node = generate_grid(mat)
    return grid, graph, 0, len(index_to_node) - 1, index_to_node


def generate_hypercube(n):
    cube = nx.hypercube_graph(n)
    node_to_index = {}
    i = 0
    for node in cube.nodes:
        cube.nodes[node]["constraint_nodes"] = list(cube.neighbors(node))
        node_to_index[node] = i
        i += 1
    return cube, node_to_index

def crop_and_parse_graph(image_path, rows, cols, mode=LSP_MODE):
    cropped_img = find_largest_rectangle(image_path)
    # Save the cropped image
    cropped_img_path = image_path[:-4]+'_crop.png'
    cv2.imwrite(cropped_img_path, cropped_img)
    res = parse_graph_png(cropped_img_path, rows, cols, mode=mode)
    os.remove(cropped_img_path)
    return res

def generate_aaai_showcase():
    x = 1
    S = 0
    T = 0

    grid = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, 0],
        [0, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, 0],
        [0, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, 0],
        [0, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [x, x, x, x, x, x, x, x, 0, x, 0, x, x, x, x, x, x, x, x, x],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [x, x, x, x, x, x, x, x, 0, x, 0, x, x, x, x, x, x, x, x, x],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, 0],
        [0, x, 0, 0, 0, 0, 0, 0, 0, x, 0, 0, 0, 0, 0, 0, 0, 0, x, 0],
        [0, x, 0, 0, 0, 0, 0, 0, 0, x, 0, 0, 0, 0, 0, 0, 0, 0, x, 0],
        [0, x, 0, 0, 0, x, x, x, x, x, x, x, x, x, x, 0, 0, 0, x, 0],
        [0, x, 0, x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x, 0, x, 0],
        [0, x, 0, x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x, 0, x, 0],
        [0, 0, S, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, T, 0, 0],
        [x, x, 0, x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x, 0, x, x],
        [0, 0, 0, x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x, 0, 0, 0],
        [0, 0, 0, 0, 0, x, x, x, x, x, x, x, x, x, x, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    start = (18, 2)
    target = (18, 17)
    grid, graph, start, target, index_to_node = graph_for_grid(grid, start, target, mode=LSP_MODE)
    index_to_node_stuff.index_to_node = index_to_node
    index_to_node_stuff.grid = grid
    draw_grid("", graph, grid, start, target, index_to_node, path=[])
    return 'showcase aaai', grid, graph, start, target, index_to_node

def generate_aaai_showcase_2(mode=LSP_MODE):
    x = 1
    S = 0
    T = 0

    grid = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, 0],
        [0, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, 0],
        [0, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, 0],
        [0, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, 0],
        [0, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, 0],
        [0, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, 0],
        [0, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, 0],
        [0, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, 0],
        [0, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, 0],
        [0, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, 0],
        [0, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, 0],
        [0, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, 0],
        [0, x, 0, 0, 0, 0, 0, 0, 0, x, 0, 0, 0, 0, 0, 0, 0, 0, x, 0],
        [0, x, 0, 0, 0, 0, 0, 0, 0, x, 0, 0, 0, 0, 0, 0, 0, 0, x, 0],
        [0, x, 0, 0, 0, x, x, x, x, x, x, x, x, x, x, 0, 0, 0, x, 0],
        [0, x, 0, x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x, 0, x, 0],
        [0, x, 0, x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x, 0, x, 0],
        [0, 0, S, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, T, 0, 0],
        [x, x, 0, x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x, 0, x, x],
        [0, 0, 0, x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x, 0, 0, 0],
        [0, 0, 0, 0, 0, x, x, x, x, x, x, x, x, x, x, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    start = (18, 2)
    target = (18, 17)
    grid, graph, start, target, index_to_node = graph_for_grid(grid, start, target, mode=mode)
    draw_grid("", graph, grid, start, target, index_to_node, path=[])
    return 'showcase aaai', grid, graph, start, target, index_to_node



def generate_rooms(mode=LSP_MODE, k=10, n=10):
    grid = [[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1],
            [0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1],
            [1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1],
            [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
            [1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

    og_mat = [
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]

    start = (0, 0)
    target = (21, 23)
    # grid, graph, start, target, index_to_node = graph_for_grid(grid, start, target, mode=mode)

    # draw_grid("", graph, grid, start, target, index_to_node, path=[])
    return generate_removed_blockes("room graph", grid, start, target, og_mat=og_mat, mode=mode, k=k)


def generate_aaai_showcase_original():
    X=1
    S=0
    T=0

    grid = [
        [0,0,0,0,0,0,0,X,X,X,X,X,X,X,X,X,X,X,X,X,X],
        [0,0,0,0,0,0,0,X,X,X,X,X,X,X,X,X,X,X,X,X,X],
        [X,X,X,X,X,0,0,0,0,0,0,0,0,0,X,X,X,X,X,X,X],
        [0,0,0,0,0,T,0,0,0,0,0,0,0,S,0,0,0,0,0,0,0],
        [0,X,X,X,X,0,0,0,0,0,0,0,0,0,X,X,X,X,X,X,0],
        [0,X,0,0,0,0,0,X,X,X,X,X,X,X,X,X,X,X,X,X,0],
        [0,X,0,0,0,0,0,X,X,X,X,X,X,X,X,X,X,X,X,X,0],
        [0,X,0,0,0,0,0,X,X,X,X,X,X,X,X,X,X,X,X,X,0],
        [0,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,X,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ]

    start = (3,13)
    target = (3,5)

    grid, graph, start, target, index_to_node = graph_for_grid(grid, start, target, mode=LSP_MODE)
    # draw_grid("", graph, grid, start, target, index_to_node, path=[])
    return 'showcase aaai', grid, graph, start, target, index_to_node


def generate_og_maze(mode=LSP_MODE, k=5, n=5):
    n = 13
    filename = '/mnt/c/Users/itay/Desktop/notebooks/all_graphs/mazes/maze_og/og_maze.png'
    mat, graph, start_node, target_node, itn = crop_and_parse_graph(filename, n, n, mode=mode)
    return generate_removed_blockes("maze", mat, itn[start_node], itn[target_node], mode=mode, k=k, n=n)


def generate_removed_blockes(name, grid, start, target, og_mat=None ,k=5, mode=LSP_MODE, n=5):
    mat, graph, start_node, target_node, itn = graph_for_grid(grid, start, target, mode=mode)

    graphs = [(name, mat, graph, start_node, target_node, itn)]
    for i in range(1, n):
        if mode == LSP_MODE:
            mat = remove_blocks(k, mat) if og_mat is None else remove_blocks_2(k, mat, og_mat)
        elif mode == SNAKE_MODE:
            mat = remove_blocks_rectangles_2(k, mat, og_mat)
        temp_graph = graph_for_grid(mat, itn[start_node], itn[target_node], mode=mode)
        graphs.append((f'{name}_{i}',) + temp_graph)
    return graphs