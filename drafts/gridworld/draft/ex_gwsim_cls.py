from ggsolver.graph import Graph
from ggsolver.gridworld.draft.viz import GWSim


if __name__ == '__main__':
    g = Graph()
    g.add_nodes(4)

    sim = GWSim(graph=g)

