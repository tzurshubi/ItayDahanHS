# shimony-dinitz edge cut thingy

def find_edge_cuts(graph, s, t, virtual_edges):
    #     global new_time
    #     start_time = time.time()
    # print(s, t, list(graph.edges), virtual_edges)
    cuts = []
    queue = []
    potential_cut = []

    flow_g = graph.to_directed()

    is_virtual = {e: 0 for e in flow_g.edges}
    for u, v in virtual_edges:
        is_virtual[(u, v)] = 1
        is_virtual[(v, u)] = 1

    paths = nx.edge_disjoint_paths(flow_g, s, t)
    p1 = next(paths)
    p2 = next(paths)

    nx.set_edge_attributes(flow_g, 0, 'flow')

    for v, u in zip(p1, p1[1:]):
        flow_g[v][u]['flow'] = 1
    for v, u in zip(p2, p2[1:]):
        flow_g[v][u]['flow'] = 1

    nx.set_node_attributes(flow_g, 0, 'label')
    flow_g.nodes[s]['label'] = 1
    queue += [s]

    while flow_g.nodes[t]['label'] == 0:
        # print(queue)
        if not queue:
            potential_cut = [(u, v) for (u, v) in potential_cut
                             if not (flow_g.nodes[u]['label'] == 1 and flow_g.nodes[v]['label'] == 1)]
            # print('cut? len ', len(potential_cut), potential_cut)
            (u1, u2), (v1, v2) = potential_cut[0], potential_cut[1]
            (u1, u2), (v1, v2) = (min(u1, u2), max(u1, u2)), (min(v1, v2), max(v1, v2))
            if is_virtual[(u1, u2)] and is_virtual[(v1, v2)]:
                cuts += [((u1, u2), (v1, v2))]
            flow_g.nodes[u2]['label'] = 1
            flow_g.nodes[v2]['label'] = 1
            queue += [u2, v2]
            # else:
            #     print(len(potential_cut), ' excuse me')
            potential_cut = []
        else:
            v = queue.pop(0)
            # print('-----------')
            # print(v)
            for v1 in [v1 for v1 in graph.neighbors(v) if flow_g.nodes[v1]['label'] == 0]:
                # print(v1)
                if flow_g[v][v1]['flow'] == 1:
                    potential_cut += [(v, v1)]
                else:
                    flow_g.nodes[v1]['label'] = 1
                    queue += [v1]

    #     new_time += time.time() - start_time
    return cuts
