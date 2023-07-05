
# astar
import time as t
from Definitions.state import State
from helpers.helper_funcs import diff, intersection, check_dimension

F = {}  # state -> weak/strong, F value

MAX_PATH_LEN = 100
# STATE = (CURRENT NODE, PATH, AVAILABLE NODES)
FAILURE = -1

STRENGTH = 0
H_VALUE = 1
G_VALUE = 2

# F_VALUE = 1

def get_h_and_g(state):
    global F
    return F[state][H_VALUE], F[state][G_VALUE]


def state_value(state, weak_h, g, weight, algorithm='a star'):
    global F
    if algorithm == 'a star':
        if state not in F.keys():
            F[state] = ('weak', weak_h(state), g(state))
        return weight * F[state][H_VALUE] + F[state][G_VALUE]
    else:
        return weak_h(state), g(state)

def set_to_strong_state_value(state, strong_h, g):
    global F
    if state not in F.keys() or not F[state][STRENGTH] == 'strong':
        F[state] = ('strong', strong_h(state), g(state))
    # return F[state][F_VALUE]


def expand_with_constraints(state, G, is_incremental, OPEN=[], CLOSED=[], hypercube_dimension=None):
    ret = []
    bccs = state.bccs
    current_v = state.current
    neighbors = G.neighbors(current_v)
    available = state.available_nodes
    next_out_node = bccs[0].out_node if is_incremental else -1
    path = state.path
    new_availables = tuple(diff(available, G.nodes[current_v]["constraint_nodes"]))
    neighbor_pool = intersection(intersection(neighbors, available), state.bccs[0].nodes) if state.bccs else intersection(neighbors, available)
    for v in neighbor_pool:
        if is_incremental and (next_out_node not in new_availables and v != next_out_node):
            continue
        new_path = path + (v,)
        new_state = State(v, new_path, new_availables)
        if is_incremental:
            new_bccs = bccs[1:].copy() if v == next_out_node else bccs
            new_state.bccs = new_bccs
        if new_state not in OPEN and new_state not in CLOSED:
            ret.append(new_state)

    return ret


def expand_with_snake_constraints(state, G, is_incremental, OPEN=[], CLOSED=[], hypercube_dimension=None):
    ret = []
    bccs = state.bccs
    current_v = state.current
    dimension = state.snake_dimension
    neighbors = list(G.neighbors(current_v))

    # Only consider neighbors with dimension <= current dimension + 1
    if hypercube_dimension is not None:
        neighbors = [v for v in neighbors if check_dimension(hypercube_dimension, v) <= dimension + 1]

    available = state.available_nodes
    next_out_node = bccs[0].out_node if is_incremental else -1
    path = state.path
    new_availables = tuple(set(available) - set(G.nodes[current_v]["constraint_nodes"]))

    # Compute neighbor pool as intersection of neighbors, available nodes, and BCC nodes (if applicable)
    neighbor_pool = set(neighbors) & set(available)
    if state.bccs:
        neighbor_pool &= set(state.bccs[0].nodes)

    for v in neighbor_pool:
        if is_incremental and (next_out_node not in new_availables and v != next_out_node):
            continue
        new_path = path + (v,)
        new_state = State(v, new_path, new_availables)
        if hypercube_dimension is not None:
            new_state.update_dimension(max(dimension, check_dimension(hypercube_dimension, v)))
        if is_incremental:
            new_bccs = bccs[1:] if v == next_out_node else bccs
            new_state.bccs = new_bccs
        if new_state not in OPEN and new_state not in CLOSED:
            ret.append(new_state)

    return ret