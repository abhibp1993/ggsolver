import ggsolver.graph as graph
from pprint import pprint


if __name__ == '__main__':
    game_graph = graph.Graph.load("out/3by3_rng1_fixed.gm")

    print(len(game_graph.nodes()))