import ggsolver


class JobstmannGame(ggsolver.Game):
    def __init__(self, **kwargs):
        super(JobstmannGame, self).__init__(is_deterministic=True)
        self._en_acts = {
            0: [(0, 1), (0, 3)],
            1: [(1, 0), (1, 2), (1, 4)],
            2: [(2, 4), (2, 2)],
            3: [(3, 0), (3, 4), (3, 5)],
            4: [(4, 3)],
            5: [(5, 3), (5, 6)],
            6: [(6, 6), (6, 7)],
            7: [(7, 0), (7, 3)]
        }

    def states(self):
        return (f"s{i}" for i in range(8))

    def enabled_acts(self, state):
        i = int(state[1:])
        return self._en_acts[i]

    def delta(self, state: str, act: tuple) -> str:
        return f"s{act[1]}"

    def final(self, state):
        return True if state in ["s3", "s4"] else False


if __name__ == '__main__':
    game = JobstmannGame()
    game.initialize("s0")
    graph = game.graphify(pointed=True, cores=3)
    edges = [(0, 1), (0, 3), (1, 0), (1, 2), (1, 4), (2, 4), (2, 2), (3, 0), (3, 4), (3, 5), (4, 3), (5, 3), (5, 6),
             (6, 6), (6, 7), (7, 0), (7, 3)]

    print("state", list(graph.nodes()))
    for u, v, k in graph.edges():
        print(graph["state"][u], graph["input"][u, v, k], graph["state"][v])
