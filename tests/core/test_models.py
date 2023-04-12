import unittest
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


class TestGameGraphify(unittest.TestCase):
    def setUp(self):
        self.game = JobstmannGame()

    def test_graphify_up_sc(self):
        graph = self.game.graphify()
        edges = [(0, 1), (0, 3), (1, 0), (1, 2), (1, 4), (2, 4), (2, 2), (3, 0), (3, 4), (3, 5), (4, 3), (5, 3), (5, 6),
                (6, 6), (6, 7), (7, 0), (7, 3)]
        self.assertEqual(set(graph.nodes()), set(range(8)))
        self.assertEqual(set(graph.edges()), {(u, v, 0) for u, v in edges})
        self.assertEqual(graph["state"][0], "s0")
        self.assertEqual(graph["state"][1], "s1")

    def test_graphify_p_sc(self):
        self.game.initialize("s0")
        graph = self.game.graphify(pointed=True)
        edges = [(0, 1), (0, 3), (1, 0), (1, 2), (1, 4), (2, 4), (2, 2), (3, 0), (3, 4), (3, 5), (4, 3), (5, 3), (5, 6),
                 (6, 6), (6, 7), (7, 0), (7, 3)]

        self.assertEqual(set(graph["state"][i] for i in graph.nodes()), set(f"s{i}" for i in range(8)))
        self.assertEqual(set((graph["state"][u], graph["state"][v]) for u, v, _ in graph.edges()),
                         set((f"s{i}", f"s{j}") for i, j in edges))
        self.assertEqual(graph["state"][0], "s0")
        self.assertEqual(graph["state"][1], "s1")

    def test_graphify_up_mc(self):
        graph = self.game.graphify(cores=3)
        edges = [(0, 1), (0, 3), (1, 0), (1, 2), (1, 4), (2, 4), (2, 2), (3, 0), (3, 4), (3, 5), (4, 3), (5, 3), (5, 6),
                 (6, 6), (6, 7), (7, 0), (7, 3)]

        self.assertEqual(set(graph["state"][i] for i in graph.nodes()), set(f"s{i}" for i in range(8)))
        self.assertEqual(set((graph["state"][u], graph["state"][v]) for u, v, _ in graph.edges()),
                         set((f"s{i}", f"s{j}") for i, j in edges))
        self.assertEqual(graph["state"][0], "s0")
        self.assertEqual(graph["state"][1], "s1")

