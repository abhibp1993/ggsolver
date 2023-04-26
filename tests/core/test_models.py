import unittest
import ggsolver


class JobstmannGame(ggsolver.Game):
    def __init__(self, **kwargs):
        super(JobstmannGame, self).__init__(is_deterministic=True, **kwargs)
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

    def __eq__(self, other: ggsolver.Game):
        # Compare states
        states = set(self.states())
        if states != set(other.states()):
            return False

        # Compare final states
        if self.final() != other.final():
            return False

        # Compare enabled_acts and delta function
        for s in states:
            if self.actions(s) != other.actions(s):
                return False

            for a in self.actions(s):
                if self.delta(s, a) != other.delta(s, a):
                    return False

        return True

    def states(self):
        return (f"s{i}" for i in range(8))

    def actions(self, state):
        try:
            i = int(state[1:])
            return self._en_acts[i]
        except ValueError:
            pass

    def delta(self, state: str, act: tuple) -> str:
        if state == "SINK":
            return "SINK"
        if int(state[1:]) == act[0]:
            return f"s{act[1]}"

    def final(self):
        return {"s3", "s4"}


class TestGameGraphify(unittest.TestCase):
    def setUp(self):
        self.game = JobstmannGame(name="MyGame")

    def test_graphify_up_sc(self):
        graph = self.game.graphify()
        edges = [(0, 1), (0, 3), (1, 0), (1, 2), (1, 4), (2, 4), (2, 2), (3, 0), (3, 4), (3, 5), (4, 3), (5, 3), (5, 6),
                 (6, 6), (6, 7), (7, 0), (7, 3)]

        # Check states
        self.assertEqual(set(graph.nodes()), set(range(8)))
        self.assertEqual({graph["state"][uid] for uid in graph.nodes()}, set(self.game.states()))

        # Since f"s{i}" could be mapped to node j, to check equality of edges, check if transitions are as expected.
        expected_trans = {(f"s{i}", (i, j), f"s{j}") for i, j in edges}
        actual_trans = {(graph["state"][u], graph["act"][u, v, k], graph["state"][v]) for u, v, k in graph.edges()}
        self.assertEqual(expected_trans, actual_trans)

    def test_graphify_p_sc(self):
        self.game.initialize("s0")
        graph = self.game.graphify(pointed=True)
        edges = [(0, 1), (0, 3), (1, 0), (1, 2), (1, 4), (2, 4), (2, 2), (3, 0), (3, 4), (3, 5), (4, 3), (5, 3), (5, 6),
                 (6, 6), (6, 7), (7, 0), (7, 3)]

        self.assertEqual(set(graph["state"][i] for i in graph.nodes()), set(f"s{i}" for i in range(8)))
        self.assertEqual(set((graph["state"][u], graph["state"][v]) for u, v, _ in graph.edges()),
                         set((f"s{i}", f"s{j}") for i, j in edges))

    def test_graphify_up_mc(self):
        graph = self.game.graphify(cores=3)
        edges = [(0, 1), (0, 3), (1, 0), (1, 2), (1, 4), (2, 4), (2, 2), (3, 0), (3, 4), (3, 5), (4, 3), (5, 3), (5, 6),
                 (6, 6), (6, 7), (7, 0), (7, 3)]

        # Check states
        self.assertEqual(set(graph.nodes()), set(range(8)))
        self.assertEqual({graph["state"][uid] for uid in graph.nodes()}, set(self.game.states()))

        # Since f"s{i}" could be mapped to node j, to check equality of edges, check if transitions are as expected.
        expected_trans = {(f"s{i}", (i, j), f"s{j}") for i, j in edges}
        actual_trans = {(graph["state"][u], graph["act"][u, v, k], graph["state"][v]) for u, v, k in graph.edges()}
        self.assertEqual(expected_trans, actual_trans)

    def test_graphify_p_mc(self):
        self.game.initialize("s0")
        graph = self.game.graphify(pointed=True, cores=3)
        edges = [(0, 1), (0, 3), (1, 0), (1, 2), (1, 4), (2, 4), (2, 2), (3, 0), (3, 4), (3, 5), (4, 3), (5, 3), (5, 6),
                 (6, 6), (6, 7), (7, 0), (7, 3)]

        print(graph.number_of_nodes())

        self.assertEqual(set(graph["state"][i] for i in graph.nodes()), set(f"s{i}" for i in range(8)))
        self.assertEqual(set((graph["state"][u], graph["state"][v]) for u, v, _ in graph.edges()),
                         set((f"s{i}", f"s{j}") for i, j in edges))

    def test_str(self):
        self.assertEqual("Deterministic JobstmannGame(name=MyGame)", self.game.__str__())


class TestCachedModel(unittest.TestCase):
    def setUp(self):
        self.game = JobstmannGame()

    def test_cached(self):
        game = self.game.make_cached()

        graph = game.graphify()
        edges = [(0, 1), (0, 3), (1, 0), (1, 2), (1, 4), (2, 4),
                 (2, 2), (3, 0), (3, 4), (3, 5), (4, 3), (5, 3),
                 (5, 6), (6, 6), (6, 7), (7, 0), (7, 3)]

        # Check nodes
        self.assertEqual(set(range(8)), set(graph.nodes()))

        # Since f"s{i}" could be mapped to node j, to check equality of edges, check if transitions are as expected.
        expected_trans = {(f"s{i}", (i, j), f"s{j}") for i, j in edges}
        actual_trans = {(graph["state"][u], graph["act"][u, v, k], graph["state"][v]) for u, v, k in graph.edges()}
        self.assertEqual(expected_trans, actual_trans)
        self.assertEqual({f"s{i}" for i in range(8)}, set(game._cache_states))


class TestCompleteGame(unittest.TestCase):
    def setUp(self):
        self.game = JobstmannGame()
        actions = [(0, 1), (0, 3), (1, 0), (1, 2), (1, 4), (2, 4),
                   (2, 2), (3, 0), (3, 4), (3, 5), (4, 3), (5, 3),
                   (5, 6), (6, 6), (6, 7), (7, 0), (7, 3)]
        self.game.inputs = lambda: actions

    def test_completed(self):
        game = self.game.make_complete(sink="SINK", sink_act="SINKACT")

        graph = game.graphify()
        edges = [(0, 1), (0, 3), (1, 0), (1, 2), (1, 4), (2, 4),
                 (2, 2), (3, 0), (3, 4), (3, 5), (4, 3), (5, 3),
                 (5, 6), (6, 6), (6, 7), (7, 0), (7, 3)]

        # Check nodes
        self.assertEqual({graph["state"][uid] for uid in graph.nodes()}, {f"s{i}" for i in range(8)} | {"SINK"})

        # # Since f"s{i}" could be mapped to node j, to check equality of edges, check if transitions are as expected.
        expected_trans = set()
        actual_trans = set()
        for u in range(8):
            for i, j in edges:
                if u == i:
                    expected_trans.add((f"s{u}", (i, j), f"s{j}"))

        expected_trans.update({("SINK", "SINKACT", "SINK")})

        for u, v, k in graph.edges():
            actual_trans.add((graph["state"][u], graph["act"][u, v, k], graph["state"][v]))

        self.assertEqual(expected_trans, actual_trans)


class TestFromGraph(unittest.TestCase):
    def setUp(self):
        self.game = JobstmannGame()

    def test_from_graph(self):
        graph = self.game.graphify()
        game = ggsolver.Game().from_graph(graph)
        self.assertEqual(self.game, game)