import itertools

import ggsolver.graph as ggraph
import ggsolver.dtptb.pgsolver as dtptb

if __name__ == '__main__':
    for i, j in itertools.product(range(5), range(5)):
        graph = ggraph.Graph.load(f"out/ex14_5x5wumpus/{i}_{j}/ex14_5x5wumpus_{i}_{j}_p1.ggraph")
        final = set()
        for uid in graph.nodes():
            if graph["state"][uid][1] == 0:
                final.add(graph["state"][uid])
        swin = dtptb.SWinReach(graph, final)
        swin.solve()

        v0 = graph["init_state"]

        print((i, j), swin.state_winner(v0))