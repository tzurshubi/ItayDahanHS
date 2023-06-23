# # linear programming (sage)
#
# def sage_flow(s, x, y, t, g):
#     g = get_vertex_disjoint_directed(g)
#     edges = list(g.edges)
#     nodes = list(g.nodes)
#     # print('starting')
#
#     prob = MixedIntegerLinearProgram()
#
#     vars = prob.new_variable(real=True, nonnegative=True)
#
#     # objective
#     prob.set_objective(prob.sum([vars[(e, 1)] for e in g.out_edges(str(s) + 'out')]
#                                 + [vars[(e, 2)] for e in g.out_edges(str(x) + 'out')]
#                                 + [vars[(e, 3)] for e in g.out_edges(str(y) + 'out')]))
#
#     # max total flow is 1 for each edge
#     for e in edges:
#         prob.add_constraint(prob.sum([vars[(e, f)] for f in [1, 2, 3]]) <= 1)
#
#     # in flow is 1 max
#     for node in nodes:
#         if node not in (str(s) + "in", str(x) + "in"):
#             prob.add_constraint(prob.sum(flatten([[vars[(e, f)] for e in g.in_edges(node)] for f in [1, 2, 3]])) <= 1)
#
#     for node in nodes:
#         if node not in (str(s) + "in", str(x) + "in"):
#             prob.add_constraint(prob.sum([vars[(e, 1)] for e in g.in_edges(node)] + [-1 * vars[(e, 1)] for e in
#                                                                                      g.out_edges(node)]) == 0)
#
#     for node in nodes:
#         if node not in (str(x) + "in", str(y) + "in"):
#             prob.add_constraint(prob.sum([vars[(e, 2)] for e in g.in_edges(node)] + [-1 * vars[(e, 2)] for e in
#                                                                                      g.out_edges(node)]) == 0)
#
#     for node in nodes:
#         if node not in (str(y) + "in", str(t) + "in"):
#             prob.add_constraint(prob.sum([vars[(e, 3)] for e in g.in_edges(node)] + [-1 * vars[(e, 3)] for e in
#                                                                                      g.out_edges(node)]) == 0)
#
#     # s -> x constraint
#     prob.add_constraint(vars[((str(s) + 'in', str(s) + 'out'), 1)] == 1)
#     prob.add_constraint(vars[((str(s) + 'in', str(s) + 'out'), 2)] == 0)
#     prob.add_constraint(vars[((str(s) + 'in', str(s) + 'out'), 3)] == 0)
#     prob.add_constraint(prob.sum([vars[(e, 1)] for e in g.in_edges(str(x) + 'in')]) == 1)
#
#     # x -> y constraint
#     prob.add_constraint(vars[((str(x) + 'in', str(x) + 'out'), 2)] == 1)
#     prob.add_constraint(vars[((str(x) + 'in', str(x) + 'out'), 1)] == 0)
#     prob.add_constraint(vars[((str(x) + 'in', str(x) + 'out'), 3)] == 0)
#     prob.add_constraint(prob.sum([vars[(e, 2)] for e in g.in_edges(str(y) + 'in')]) == 1)
#     #
#     # # y -> t constraint
#     prob.add_constraint(vars[((str(y) + 'in', str(y) + 'out'), 3)] == 1)
#     prob.add_constraint(vars[((str(y) + 'in', str(y) + 'out'), 2)] == 0)
#     prob.add_constraint(vars[((str(y) + 'in', str(y) + 'out'), 1)] == 0)
#     prob.add_constraint(prob.sum([vars[(e, 3)] for e in g.in_edges(str(t) + 'in')]) == 1)
#
#     prob.add_constraint(prob.sum([vars[(e, 1)] for e in g.in_edges(str(s) + 'in')]) == 0)
#     prob.add_constraint(prob.sum([vars[(e, 2)] for e in g.in_edges(str(x) + 'in')]) == 0)
#     prob.add_constraint(prob.sum([vars[(e, 3)] for e in g.in_edges(str(y) + 'in')]) == 0)
#
#     # print("+++++++")
#     # print(prob)
#
#     # print('solving')
#     try:
#         prob.solve()
#         # prob.show()
#         # print(1)
#         return True
#     except MIPSolverException as lpe:
#         # print(0)
#         return False
#     # print("status", prob.status, "Total flow =", value(prob.objective))
#     # if prob.status != -1:
#     #     print(prob.status == 1)
#     #     for v in prob.variables():
#     #         print(v.name, "=", v.varValue)