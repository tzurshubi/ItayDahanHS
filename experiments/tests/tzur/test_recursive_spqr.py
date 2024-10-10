import networkx as nx
from sage.graphs.graph import Graph
from sage.graphs.connectivity import spqr_tree
from heuristics.recursive_spqr import get_max_nodes_spqr_recursive

def generate_random_graph(num_nodes=10, edge_prob=0.3):
    """
    Generate a random undirected graph using networkx.
    :param num_nodes: Number of nodes in the graph.
    :param edge_prob: Probability of edge creation.
    :return: A networkx Graph object.
    """`
    return nx.erdos_renyi_graph(num_nodes, edge_prob)

def test_recursive_spqr():
    # Generate a random graph
    num_nodes = 10
    edge_prob = 0.4
    G = generate_random_graph(num_nodes, edge_prob)

    # Pick two random nodes as start (in_node) and end (out_node)
    in_node, out_node = list(G.nodes)[:2]

    # Convert the networkx graph to a Sage graph
    G_sage = Graph(G)

    print(f"Testing on graph with {num_nodes} nodes and edge probability {edge_prob}.")
    print(f"Start node (in_node): {in_node}, End node (out_node): {out_node}")
    print(f"Nodes: {list(G.nodes)}")
    print(f"Edges: {list(G.edges)}")

    # Call the function to test the heuristic calculation
    try:
        heuristic_value = get_max_nodes_spqr_recursive(G, in_node, out_node, return_nodes=False)
        print(f"Calculated heuristic (number of nodes in the SPQR decomposition): {heuristic_value}")
    except Exception as e:
        print(f"Error during SPQR heuristic calculation: {e}")

if __name__ == "__main__":
    test_recursive_spqr()
