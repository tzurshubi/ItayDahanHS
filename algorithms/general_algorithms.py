from Definitions.bi_comp_entry import BiCompEntry
from helpers.COMMON import LSP_MODE
from helpers.helper_funcs import bcc_thingy, intersection
from heuristics.heauristics.snake_spqr.snake_general import snake_exclusion_set_spqr

NUM_OF_PAIRS = 5
N = 0

# Computes the g-value of the current state
def g(state):
    """
    Computes the g-value of the current state, representing the cost to reach the current node.

    Parameters:
    - state: The current search state, which has an attribute `path` representing the path taken so far.

    Returns:
    - g_value: The length of the path from the start node to the current node minus one.
    """
    path = state.path
    g_value = len(path) - 1
    return g_value


#  Applies the given heuristic function to the provided state
def function(state, heuristic, G, target):
    """
    Applies the given heuristic function to the provided state.

    Parameters:
    - state: The current state of the search.
    - heuristic: A function that estimates the cost or distance to reach the target from the current state.
    - G: The graph in which the search is being performed.
    - target: The target node that the search aims to reach.

    Returns:
    - The heuristic value computed by the provided heuristic function.
    """
    return heuristic(state, G, target)


# Calculates the heuristic values for each biconnected component in the search graph.
def calc_comps(state, G, target, algorithm, mode=LSP_MODE):
    """
    Calculates the heuristic values for each biconnected component in the search graph.

    Parameters:
    - state: The current state of the search.
    - G: The graph being searched.
    - target: The target node to reach.
    - algorithm: A function that computes the heuristic value for each subgraph.
    - mode: Specifies the mode of the search, either LSP (Longest Simple Path) or other modes.

    Returns:
    - comps_hs: A list of BiCompEntry objects representing each biconnected component with its computed heuristic value.
    - -1 if no path is found.
    """
    # Extract relevant biconnected components and cut nodes between them.
    _, _, relevant_comps, _, reach_nested, current_node = bcc_thingy(state, G, target)
    if relevant_comps == -1:
        return -1  # Indicates no path exists between the nodes.
    
    n = len(relevant_comps)
    if n == 0:
        return
    
    # Define the cut nodes between consecutive components.
    cut_nodes = [(current_node, target)] if n == 1 else [
        (current_node, list(intersection(relevant_comps[0], relevant_comps[1]))[0])
    ] + [
        (list(intersection(relevant_comps[i - 1], relevant_comps[i]))[0], list(intersection(relevant_comps[i + 1], relevant_comps[i]))[0])
        for i in range(1, n - 1)
    ] + [
        (list(intersection(relevant_comps[n - 2], relevant_comps[n - 1]))[0], target)
    ]

    # Construct subgraphs for each component based on the mode.
    if mode != LSP_MODE:
        subgraphs = [reach_nested.subgraph(comp) for comp in relevant_comps]
    else:
        subgraphs = [
            reach_nested.subgraph(snake_exclusion_set_spqr(reach_nested.subgraph(comp), in_node, out_node))
            for comp, (in_node, out_node) in zip(relevant_comps, cut_nodes)
        ]

    # Create BiCompEntry objects containing the subgraph and its heuristic value.
    comps_hs = [
        BiCompEntry(in_node, out_node, comp, algorithm(comp, in_node, out_node))
        for (in_node, out_node), comp in zip(cut_nodes, subgraphs)
    ]

    return comps_hs


# Calculates the heuristic values for each biconnected component in the search graph for the Snake problem
def snake_calc_comps(state, G, target, algorithm):
    """
    Calculates the heuristic values for each biconnected component in the search graph for the Snake problem.

    Parameters:
    - state: The current state of the search, containing information about the path and current node.
    - G: The graph being searched.
    - target: The target node that the search aims to reach.
    - algorithm: A function that computes the heuristic value for each subgraph component.

    Returns:
    - comps_hs: A list of BiCompEntry objects representing each biconnected component with its computed heuristic value.
    - -1 if no path is found (when relevant_comps is -1).
    - 0 if there are no relevant components for further exploration.
    """
    # Extract relevant biconnected components and cut nodes between them.
    _, _, relevant_comps, _, reach_nested, current_node = bcc_thingy(state, G, target)
    if relevant_comps == -1:
        return -1  # Indicates no path exists between the nodes.

    n = len(relevant_comps)
    if n == 0:
        return 0  # No relevant components for further exploration.

    # Define the cut nodes between consecutive components.
    cut_nodes = [(current_node, target)] if n == 1 else [
        (current_node, list(intersection(relevant_comps[0], relevant_comps[1]))[0])
    ] + [
        (list(intersection(relevant_comps[i - 1], relevant_comps[i]))[0], list(intersection(relevant_comps[i + 1], relevant_comps[i]))[0])
        for i in range(1, n - 1)
    ] + [
        (list(intersection(relevant_comps[n - 2], relevant_comps[n - 1]))[0], target)
    ]

    # Construct subgraphs for each component considering the snake problem's exclusion set.
    subgraphs = [
        reach_nested.subgraph(snake_exclusion_set_spqr(reach_nested.subgraph(comp), in_node, out_node))
        for comp, (in_node, out_node) in zip(relevant_comps, cut_nodes)
    ]

    # Create BiCompEntry objects containing the subgraph and its heuristic value.
    comps_hs = [
        BiCompEntry(in_node, out_node, comp, algorithm(comp, in_node, out_node))
        for (in_node, out_node), comp in zip(cut_nodes, subgraphs)
    ]

    return comps_hs


#  Computes a heuristic value for the state by summing the heuristic estimates of each biconnected component
def ex_pairs(state, G, target, algorithm, mode=LSP_MODE, prune=False):
    """
    Computes a heuristic value for the state by summing the heuristic estimates of each biconnected component.

    Parameters:
    - state: The current state of the search.
    - G: The graph being searched.
    - target: The target node to reach.
    - algorithm: A function that computes the heuristic value for each subgraph.
    - mode: Specifies the mode of the search, either LSP (Longest Simple Path) or other modes.
    - prune: A boolean flag indicating if pruning should be applied (not currently implemented).

    Returns:
    - relevant_nodes: The total estimated heuristic value considering all components.
    - 1 if the current node is already the target.
    - The heuristic value from calc_comps if it returns an error value.
    """
    current_node = state.current
    if current_node == target:
        return 1  # Directly at the target node, so the estimated cost is 1.
    
    # Calculate the heuristic values for each relevant component.
    comp_hs = calc_comps(state, G, target, algorithm, mode=mode)
    if isinstance(comp_hs, int) and comp_hs <= 0:
        return comp_hs  # Indicates an error or that no path is found.
    
    # Sum the heuristic values of each component to estimate the overall heuristic.
    relevant_nodes = 1 + sum([c.h - 1 for c in comp_hs])
    return relevant_nodes

