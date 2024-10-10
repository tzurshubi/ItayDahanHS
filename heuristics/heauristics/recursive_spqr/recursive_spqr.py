
# recursive attempt
import itertools
import networkx as nx
from sage.graphs.connectivity import spqr_tree
from sage.graphs.graph import Graph

from heuristics.heuristics_helper_funcs import get_relevant_cuts
from helpers.helper_funcs import intersection, flatten, diff

# Handles traversal through the SPQR tree to identify nodes relevant to the heuristic.
# Dispatches the specific computation based on the type of SPQR tree node ('R', 'P', or 'S') by calling nodes_r, nodes_p, or nodes_s.
def spqr_nodes(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict):
    """
    Recursively traverses the SPQR tree to identify nodes relevant to the path between in_node and out_node.

    Parameters:
    - current_sn: The current SPQR tree node being processed.
    - parent_sn: The parent SPQR tree node from which the traversal originated.
    - tree: The SPQR tree of the graph.
    - g: The original graph.
    - in_node: The entry node into the current component.
    - out_node: The exit node from the current component.
    - sp_dict: A dictionary containing the separation pairs between components in the SPQR tree.

    Returns:
    - A list of nodes that are part of the identified path or structure.
    """
    if current_sn[0] == 'R':
        return nodes_r(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict)
    if current_sn[0] == 'P':
        return nodes_p(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict)
    if current_sn[0] == 'S':
        return nodes_s(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict)
    return []


# Handles 'S' (series) components by considering paths in a series and determining the longest sequences.
def nodes_s(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict):
    """
    Processes 'S' (series) components in the SPQR tree to identify relevant nodes.

    Parameters:
    - current_sn: The current series node in the SPQR tree.
    - parent_sn: The parent node in the SPQR tree from which this node was accessed.
    - tree: The SPQR tree of the graph.
    - g: The original graph.
    - in_node: The entry node into the component.
    - out_node: The exit node from the component.
    - sp_dict: A dictionary containing the separation pairs between components in the SPQR tree.

    Returns:
    - A list of nodes that form the longest sequence between in_node and out_node.
    """
    sn_sp = [(neighbor_sn, sp_dict[(neighbor_sn, current_sn)]) for neighbor_sn in tree.networkx_graph().neighbors(current_sn) if neighbor_sn != parent_sn]
    super_n_nodes = [((i, o), spqr_nodes(neighbor_sn, current_sn, tree, g, i, o, sp_dict)) for neighbor_sn, (i, o) in sn_sp]
    in_out_sn = [n for (i, o), n in super_n_nodes if (i, o) == (in_node, out_node)]
    ret = []
    if in_out_sn:
        # Print debug information for in-out sequence.
        print('iosn', in_node, out_node, in_out_sn)
        print(g.has_edge(in_node, out_node))
        other_path = flatten([n for (i, o), n in super_n_nodes if (i, o) != (in_node, out_node)]) + list(current_sn[1].networkx_graph().nodes)
        ret = max((in_out_sn[0] + [in_node, out_node], other_path), key=len)
    else:
        ret = flatten([n for (i, o), n in super_n_nodes]) + list(current_sn[1].networkx_graph().nodes)
    ret = ret if not parent_sn else diff(ret, [in_node, out_node])
    return ret


# Handles 'P' (parallel) components by identifying paths and taking the maximal node sets.
def nodes_p(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict):
    """
    Processes 'P' (parallel) components in the SPQR tree to identify relevant nodes.

    Parameters:
    - current_sn: The current parallel node in the SPQR tree.
    - parent_sn: The parent node in the SPQR tree from which this node was accessed.
    - tree: The SPQR tree of the graph.
    - g: The original graph.
    - in_node: The entry node into the component.
    - out_node: The exit node from the component.
    - sp_dict: A dictionary containing the separation pairs between components in the SPQR tree.

    Returns:
    - A list of nodes that form the largest path between in_node and out_node within the parallel structure.
    """
    sn_sp = [(neighbor_sn, sp_dict[(neighbor_sn, current_sn)]) for neighbor_sn in tree.networkx_graph().neighbors(current_sn) if neighbor_sn != parent_sn]
    super_n_nodes = [spqr_nodes(neighbor_sn, current_sn, tree, g, i, o, sp_dict) for neighbor_sn, (i, o) in sn_sp]
    ret = max(super_n_nodes, key=len)
    ret = ret + [in_node, out_node] if not parent_sn else ret
    return ret


# Handles 'R' (rigid) components, building subgraphs and computing relevant nodes by analyzing cut edges.
def nodes_r(current_sn, parent_sn, tree, g, in_node, out_node, sp_dict):
    """
    Processes 'R' (rigid) components in the SPQR tree to identify relevant nodes.

    Parameters:
    - current_sn: The current rigid node in the SPQR tree.
    - parent_sn: The parent node in the SPQR tree from which this node was accessed.
    - tree: The SPQR tree of the graph.
    - g: The original graph.
    - in_node: The entry node into the component.
    - out_node: The exit node from the component.
    - sp_dict: A dictionary containing the separation pairs between components in the SPQR tree.

    Returns:
    - A list of nodes that form the longest path or set of nodes relevant to the heuristic within the rigid structure.
    """
    sn_sp = [(neighbor_sn, sp_dict[(neighbor_sn, current_sn)]) for neighbor_sn in tree.networkx_graph().neighbors(current_sn) if neighbor_sn != parent_sn]
    super_n_nodes_dict = [((i, o), spqr_nodes(neighbor_sn, current_sn, tree, g, i, o, sp_dict)) for neighbor_sn, (i, o) in sn_sp]
    super_n_nodes_dict = dict(super_n_nodes_dict)
    i_o_sn = super_n_nodes_dict.get((in_node, out_node), [])
    super_n_nodes_dict.pop((in_node, out_node), None)

    # Extract nodes from the current component.
    sp_nodes = list(current_sn[1].networkx_graph().nodes)
    local_g = g.subgraph(sp_nodes).copy()
    local_g.add_edges_from([e for e in super_n_nodes_dict.keys() if e not in local_g.edges])

    # Identify edges connected to in_node and out_node.
    in_edges = [e for e in super_n_nodes_dict.keys() if in_node in e]
    out_edges = [e for e in super_n_nodes_dict.keys() if out_node in e]

    # Remove the direct edge between in_node and out_node if it exists.
    if local_g.has_edge(in_node, out_node):
        local_g.remove_edge(in_node, out_node)

    # Identify relevant cut edges.
    cut_edges = get_relevant_cuts(local_g, super_n_nodes_dict.keys())

    # Extract nodes that can be easily determined from the cut edges.
    in_out_edges = in_edges + out_edges
    cut_maxes = [(max((e1, e2), key=lambda x: len(super_n_nodes_dict[x])), (e1, e2)) for e1, e2 in cut_edges]
    easy_cut_nodes = [(super_n_nodes_dict[e], (e1, e2)) for e, (e1, e2) in cut_maxes if e not in in_out_edges]
    easy_nodes = [n for n, (e1, e2) in easy_cut_nodes]
    easy_nodes += [super_n_nodes_dict[e] for e in diff(super_n_nodes_dict.keys(), flatten(cut_edges) + in_out_edges)]

    # Compute exclusion nodes and select the longest set of relevant nodes.
    relevant_cuts = diff(cut_edges, [t[1] for t in easy_cut_nodes])
    in_cuts = [(e1, e2) for e1, e2 in relevant_cuts if (in_node in e1 or in_node in e2) and not (in_node in e1 and in_node in e2)]
    out_cuts = [(e1, e2) for e1, e2 in relevant_cuts if (out_node in e1 or out_node in e2) and not (out_node in e1 and out_node in e2)]
    c = get_max_comb(in_cuts, in_edges, super_n_nodes_dict) + get_max_comb(out_cuts, out_edges, super_n_nodes_dict)

    ret = max((flatten(easy_nodes + [super_n_nodes_dict[e] for e in c]) + sp_nodes, i_o_sn + [in_node, out_node]), key=len)
    ret = ret if not parent_sn else diff(ret, [in_node, out_node])
    return ret


#  Retrieves the nodes associated with a given edge from the SPQR tree's separator dictionary
def get_edge_nodes(e, sp_dict):
    """
    Retrieves the nodes associated with a given edge from the SPQR tree's separator dictionary.

    Parameters:
    - e: A tuple representing the edge for which the nodes are needed.
    - sp_dict: A dictionary containing the separation pairs between components in the SPQR tree.

    Returns:
    - A tuple representing the nodes that form the separator for the given edge.
    - If the edge is not found in the dictionary, it attempts to find its reverse.
    """
    return sp_dict[e] if e in sp_dict else sp_dict[(e[1], e[0])]


# Builds an exclusion graph using networkx.
# Creates a graph where edges represent exclusion pairs (2 nodes that cannot coexist on a path - due to the exclusion properties derived from SPQR tree analysis).
def get_max_comb(relevant_cuts, node_edges, super_n_nodes_dict):
    """
    Constructs an exclusion graph and finds the maximum independent set (MIS) by analyzing relevant cuts and node edges.

    Parameters:
    - relevant_cuts: A list of tuples representing edges that cannot coexist due to cut properties between nodes.
    - node_edges: A list of edges connected to the in_node or out_node, indicating potential exclusion relationships.
    - super_n_nodes_dict: A dictionary mapping edges to their associated node lists.

    Returns:
    - c: A list of edges that form the largest clique in the complement of the exclusion graph,
         which corresponds to the maximum independent set (MIS) in the original graph.
    """
    # Create a new graph to represent the exclusion relationships.
    exg = nx.Graph()

    # Combine node edges with flattened relevant cuts to create the initial set of edges for the exclusion graph.
    ex_edges = node_edges + flatten(relevant_cuts)
    exg.add_nodes_from(ex_edges)

    # Generate all pairs of node edges and add them as edges in the exclusion graph.
    # These edges represent pairs of nodes that cannot be on the same path.
    exg_pairs = list(itertools.combinations(node_edges, 2)) + relevant_cuts
    exg.add_edges_from(exg_pairs)

    # Construct the complement of the exclusion graph.
    # The complement graph contains edges between nodes that can coexist in a path.
    complement_exg = nx.complement(exg)

    # Assign weights to the nodes in the complement graph based on the length of their corresponding node lists.
    for node in complement_exg:
        complement_exg.nodes[node]['l'] = len(super_n_nodes_dict[node])

    # Find the maximum weight clique in the complement graph.
    # The clique corresponds to a set of nodes that can coexist, and thus to the largest independent set in the original exclusion graph.
    c, w = nx.max_weight_clique(complement_exg, weight='l')
    return c


#  Constructs a dictionary of separation pairs for each edge between components in the SPQR tree
def edge_seperators(tree):
    """
    Constructs a dictionary of separation pairs for each edge between components in the SPQR tree.

    Parameters:
    - tree: The SPQR tree of a graph, with each node representing a component of the graph.

    Returns:
    - sp_dict: A dictionary where each key is a tuple representing an edge between two components, 
      and the value is the tuple of nodes that forms the separator between those components.
    """
    sp_dict = {}
    for (t1, g1), (t2, g2) in tree.networkx_graph().edges:
        # Find the nodes that are shared between the two components.
        sp = tuple(intersection(g1.networkx_graph().nodes, g2.networkx_graph().nodes))
        sp = min(sp), max(sp)
        # Store the separator for both directions of the edge.
        sp_dict[((t1, g1), (t2, g2))] = sp
        sp_dict[((t2, g2), (t1, g1))] = sp
    return sp_dict


# Finds the root component in the SPQR tree that contains both the source and target nodes
def find_root_sn(tree, s, t, sp_dict):
    """
    Finds the root component in the SPQR tree that contains both the source and target nodes.

    Parameters:
    - tree: The SPQR tree of a graph.
    - s: The source node in the graph.
    - t: The target node in the graph.
    - sp_dict: A dictionary containing the separation pairs between components in the SPQR tree.

    Returns:
    - The root component of the SPQR tree that contains both the source (s) and target (t) nodes.
      The function prefers a 'P' (parallel) node over other types when multiple components match.
    """
    return sorted(
        [x for x in tree if ({s, t} & set(x[1].networkx_graph().nodes)) == {s, t}],
        key=lambda x: 0 if x[0] == 'P' else 1
    )[0]


# compute ℎ_𝑀𝐼𝑆 (with an appropriate graph component and starting/ending nodes)
# Computes the nodes relevant for the longest path problem using the SPQR tree of the component
def get_max_nodes_spqr_recursive(component, in_node, out_node, return_nodes=False):
    """
    Computes the nodes relevant for the longest path problem using the SPQR tree of the component.

    Parameters:
    - component: The graph component being analyzed.
    - in_node: The entry node into the component.
    - out_node: The exit node from the component.
    - return_nodes: If True, returns the list of nodes in the path; otherwise, returns the count of nodes.

    Returns:
    - A list of nodes that form the relevant part of the path if return_nodes is True.
    - An integer representing the number of nodes in the path if return_nodes is False.
    """
    # Create a copy of the component to avoid modifying the original.
    comp = component.copy()
    
    # If the component has only two nodes, return them directly.
    if len(comp.nodes) == 2:
        return comp.nodes if return_nodes else len(comp.nodes)
    
    # Ensure the component includes an edge between in_node and out_node.
    if not comp.has_edge(in_node, out_node):
        comp.add_edge(in_node, out_node)
    
    # Create an SPQR tree for the component using SageMath.
    comp_sage = Graph(comp)
    tree = spqr_tree(comp_sage)
    
    # Generate the dictionary of separation pairs between components.
    sp_dict = edge_seperators(tree)
    
    # Identify the root node of the SPQR tree that contains both in_node and out_node.
    root_node = find_root_sn(tree, in_node, out_node, sp_dict)
    
    # Traverse the SPQR tree from the root to determine the nodes relevant for the path.
    nodes = spqr_nodes(root_node, [], tree, comp, min(in_node, out_node), max(in_node, out_node), sp_dict)
    
    return nodes if return_nodes else len(nodes)
