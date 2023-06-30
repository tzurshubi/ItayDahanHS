
class BiCompEntry:
    def __init__(self, in_node, out_node, nodes, h):
        self.in_node = in_node
        self.out_node = out_node
        self.nodes = nodes
        self.h = h

    def print(self):
        print('h', self.h, ' || in', self.in_node, ' || out', self.out_node, ' || nodes', self.nodes)

    def to_tuple(self):
        return self.h, self.in_node, self.out_node, list(self.nodes)