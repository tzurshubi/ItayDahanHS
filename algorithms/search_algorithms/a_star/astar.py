import time as t
from algorithms.algorithms_helpers import state_value, expand_with_constraints


def max_weighted_a_star(G, start_state, is_goal, h, g, is_incremental=False, hypercube_dimension=None, expand=expand_with_constraints, weight=1, cutoff=-1, timeout=-1):
    global F
    global state_index

    def get_state_value(state):
        return state_value(state, h, g, weight=weight)

    state_index = 0
    F = {}
    start_time = t.time()
    h_vals = []
    nodes_chosen = []
    lens = []
    OPEN = [start_state]
    CLOSED = []
    expansions = 0
    count=0
    while OPEN:
        # if expansions != 0 and expansions % 1000 == 0:
        #     print(expansions)

        q = max(OPEN, key=get_state_value)
        OPEN.remove(q)
        # if expansions % 1000 == 0:
        #     h_val, g_val = get_h_and_g(q)
        #     print(f"state pulled from Open: H_val: {h_val}, g_val: {g_val}, f_val: {h_val + g_val}")

        h_vals += [get_state_value(q)]
        lens += [len(q.path)]
        nodes_chosen += [q.current]

        print(h_vals)
        count += 1
        if count > 2:
            return q, (expansions, t.time() - start_time, h_vals, lens, nodes_chosen, len(OPEN) + len(CLOSED))

        if expansions > cutoff > -1 or t.time() - start_time > timeout > -1:
            return q, (expansions, t.time() - start_time, h_vals, lens, nodes_chosen, len(OPEN) + len(CLOSED))
        if is_goal(q):
            return q, (expansions, t.time() - start_time, h_vals, lens, nodes_chosen, len(OPEN) + len(CLOSED))
        OPEN += expand(q, G, is_incremental, OPEN, CLOSED, hypercube_dimension=hypercube_dimension)
        expansions += 1
        CLOSED += [q]
    return -1, (expansions, t.time() - start_time, h_vals, lens, nodes_chosen, len(OPEN) + len(CLOSED))



