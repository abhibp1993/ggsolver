import ggsolver.graph as graph
from pprint import pprint


if __name__ == '__main__':
    game_graph = graph.Graph.load("4by4_rng2_random.gm")

    states = {game_graph["state"][uid] for uid in game_graph.nodes()}
    finals = set()
    for state in states:
        if state[1] == 0:
            finals.add(state)

    pprint(finals)