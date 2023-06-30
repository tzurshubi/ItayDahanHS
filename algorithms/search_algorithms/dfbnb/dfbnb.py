import time as t
from algorithms.algorithms_helpers import state_value, expand_with_snake_constraints
from helpers.helper_funcs import save_state


def max_dfbnb_iterative(G, start_state, is_goal, h, g, is_incremental=False, expand=expand_with_snake_constraints, weight=1, cutoff=-1, timeout=-1, hypercube_dimension=-1, save_dir=-1):
    #     global F
    global state_index
    global best_state

    def get_state_value(state):
        return state_value(state, h, g, weight=weight, algorithm='dfbnb')



    bounds = {3: 3, 4: 5, 5: 11, 6: 23, 7: 45, 8: 93, 9: 187, 10: 365, 11: 691, 12: 1343, 13: 2593}
    bound = bounds[hypercube_dimension]

    best_state = start_state

    #     OPEN = np.zeros((bound * 2, hypercube_dimension))
    #     print(arr)

    state_index = 0
    #     F = {}
    start_time = t.time()
    #     h_vals = []
    #     nodes_chosen = []
    #     lens = []
    #     OPEN[0][0] = start_state
    OPEN = [start_state]
    #     CLOSED = []
    expansions = 0
    pruned = 0

    while OPEN:
        # if expansions != 0 and expansions % 1000 == 0:
        #     print(expansions)
        #         q = OPEN[depth][place]
        q = OPEN.pop()
        h_val, g_val = get_state_value(q)
        q_val = weight * h_val + g_val

        if expansions % 10000 == 0:
            #             h_val, g_val = get_h_and_g(q)
            print(f"state pulled from Open: H_val: {h_val}, g_val: {g_val}, f_val: {h_val + g_val}")
            # save_state(save_dir, q, best_state, bound, OPEN, expansions, pruned, t.time() - start_time, dimension=hypercube_dimension)


        if expansions > cutoff > -1 or t.time() - start_time > timeout > -1:
            print('--------wtf------')
            return best_state, (expansions, t.time() - start_time, [0], [0], [0], pruned)

        if q_val <= bound:
            pruned += 1
        else:

            #             h_vals += [q_val]
            #             lens += [len(q.path)]
            #             nodes_chosen += [q.current]

            if is_goal(q):
                bound = q_val
                best_state = q
                print(f'found path: {bound - 1}')
                # save_state(save_dir, q, best_state, bound, OPEN, expansions, pruned, t.time() - start_time, dimension=hypercube_dimension, best=True)
            else:
                OPEN += expand(q, G, is_incremental, hypercube_dimension, OPEN)
                #                 if len(expanded) == 0:
                #                     print(f'open len: {len(OPEN)}')
                #                     print(f'depth: {len(q.path)}')
                #                 else:
                #                     OPEN += expanded
                expansions += 1
    #                 CLOSED += [q]
    return best_state, (expansions, t.time() - start_time, [0], [0], [0], pruned)


