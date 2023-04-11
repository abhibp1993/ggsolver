import inspect
import pathlib
import pickle
import pytest
import unittest
import shutil

import ggsolver


class TestGraphTopology(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create folder if it does not exist.
        if not pathlib.Path("out/").exists():
            pathlib.Path("out/").mkdir()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("out/")

    def setUp(self):
        self.graph = ggsolver.Graph()
        self.fpath = "out/graph.ggraph"

    def test_add_nodes_and_edges(self):
        # Single node addition
        result = self.graph.add_node()
        self.assertEqual(result, 0)
        result = self.graph.add_node()
        self.assertEqual(result, 1)

        # Add multiple nodes at once
        result = self.graph.add_nodes(num_nodes=3)
        self.assertIsInstance(result, list)
        self.assertEqual(list(result), [2, 3, 4])

        # Add single edge
        result = self.graph.add_edge(0, 1)
        self.assertEqual(result, 0)
        result = self.graph.add_edge(0, 1)
        self.assertEqual(result, 1)
        result = self.graph.add_edge(1, 2)
        self.assertEqual(result, 0)

        with pytest.raises(KeyError):
            self.graph.add_edge(1, 10)

        # Add multiple edges
        result = self.graph.add_edges([(0, 1), (1, 2), (2, 3)])
        self.assertIsInstance(result, list)
        self.assertEqual(list(result), [2, 1, 0])

        # Get nodes and edges
        self.assertTrue(inspect.isgenerator(self.graph.nodes()))
        self.assertTrue(inspect.isgenerator(self.graph.edges()))
        self.assertEqual(set(self.graph.nodes()), {0, 1, 2, 3, 4})
        self.assertEqual(set(self.graph.edges()), {(0, 1, 0), (0, 1, 1), (0, 1, 2), (1, 2, 0), (1, 2, 1), (2, 3, 0)})

        # Check number of nodes and edges
        self.assertEqual(self.graph.number_of_nodes(), 5)
        self.assertEqual(self.graph.number_of_edges(), 6)

        # Check existence functions
        self.assertTrue(self.graph.has_node(0))
        self.assertFalse(self.graph.has_node(10))
        self.assertTrue(self.graph.has_edge(0, 1, 0))
        self.assertFalse(self.graph.has_edge(0, 1, 6))
        self.assertFalse(self.graph.has_edge(1, 3, 0))

    def test_repr_str(self):
        # Add some nodes and edges to graph
        self.graph.add_nodes(num_nodes=3)
        self.graph.add_edges([(0, 1), (1, 2), (0, 2)])

        self.assertEqual(self.graph.number_of_nodes(), 3)
        self.assertEqual(self.graph.number_of_edges(), 3)

        # Check repr without name
        self.assertEqual(self.graph.name, None)
        val = f"<Graph with |V|=3, |E|=3>"
        self.assertEqual(self.graph.__repr__(), val)
        self.assertEqual(self.graph.__str__(), val)

        # Check repr with name
        self.graph.name = "MyGraph"

        self.assertEqual(self.graph.name, "MyGraph")
        val = f"<Graph with |V|=3, |E|=3>"
        self.assertEqual(self.graph.__repr__(), val)
        val = f"Graph(MyGraph)"
        self.assertEqual(self.graph.__str__(), val)

    def test_neighborhood(self):
        # Add some nodes and edges to graph
        self.graph.add_nodes(num_nodes=5)
        self.graph.add_edges([(0, 1), (0, 1), (1, 2), (0, 2), (2, 3), (3, 0), (0, 0)])

        # Successors
        # self.assertIsInstance(self.graph.successors(0), Iterator)
        self.assertTrue(inspect.isgenerator(self.graph.successors(0)))
        self.assertEqual(set(self.graph.successors(0)), {0, 1, 2})

        # self.assertIsInstance(self.graph.successors(4), Iterator)
        self.assertTrue(inspect.isgenerator(self.graph.successors(4)))
        self.assertEqual(set(self.graph.successors(4)), set())

        # Predecessors
        # self.assertIsInstance(self.graph.predecessors(0), Iterator)
        self.assertTrue(inspect.isgenerator(self.graph.predecessors(0)))
        self.assertEqual(set(self.graph.predecessors(0)), {0, 3})

        # self.assertIsInstance(self.graph.predecessors(4), Iterator)
        self.assertTrue(inspect.isgenerator(self.graph.predecessors(4)))
        self.assertEqual(set(self.graph.predecessors(4)), set())

        # Neighbors
        # self.assertIsInstance(self.graph.neighbors(0), Iterator)
        self.assertTrue(inspect.isgenerator(self.graph.neighbors(0)))
        self.assertEqual(set(self.graph.neighbors(0)), {0, 1, 2, 3})

        # self.assertIsInstance(self.graph.neighbors(4), Iterator)
        self.assertTrue(inspect.isgenerator(self.graph.neighbors(4)))
        self.assertEqual(set(self.graph.neighbors(4)), set())

        # Ancestors
        self.assertTrue(inspect.isgenerator(self.graph.ancestors(0)))
        self.assertEqual(set(self.graph.ancestors(0)), {1, 2, 3})

        self.assertTrue(inspect.isgenerator(self.graph.ancestors(4)))
        self.assertEqual(set(self.graph.ancestors(4)), set())

        # Descendants
        self.assertTrue(inspect.isgenerator(self.graph.descendants(0)))
        self.assertEqual(set(self.graph.descendants(0)), {1, 2, 3})

        self.assertTrue(inspect.isgenerator(self.graph.descendants(4)))
        self.assertEqual(set(self.graph.descendants(4)), set())

        # in-edges
        # self.assertIsInstance(self.graph.in_edges(0), Iterator)
        self.assertTrue(inspect.isgenerator(self.graph.in_edges(4)))
        self.assertEqual(set(self.graph.in_edges(0)), {(0, 0, 0), (3, 0, 0)})

        # self.assertIsInstance(self.graph.in_edges(4), Iterator)
        self.assertTrue(inspect.isgenerator(self.graph.in_edges(4)))
        self.assertEqual(set(self.graph.in_edges(4)), set())

        # out-edges
        # self.assertIsInstance(self.graph.out_edges(0), Iterator)
        self.assertTrue(inspect.isgenerator(self.graph.out_edges(4)))
        self.assertEqual(set(self.graph.out_edges(0)), {(0, 2, 0), (0, 0, 0), (0, 1, 0), (0, 1, 1)})

        # self.assertIsInstance(self.graph.in_edges(4), Iterator)
        self.assertTrue(inspect.isgenerator(self.graph.out_edges(4)))
        self.assertEqual(set(self.graph.out_edges(4)), set())

    def test_properties(self):
        # Add some nodes and edges to graph
        self.graph.add_nodes(num_nodes=5)
        self.graph.add_edges([(0, 1), (0, 1), (0, 1), (1, 2), (0, 2), (2, 3), (3, 0), (0, 0)])

        # Graph properties
        self.graph["gp1"] = "gp1"
        self.assertEqual(self.graph["gp1"], "gp1")

        with pytest.raises(KeyError):
            p = self.graph["non-existent"]

        # Node properties
        p = self.graph.create_np(pname="np1")
        self.assertIsInstance(p, ggsolver.NodePMap)

        with pytest.raises(AssertionError):
            self.graph.create_np(pname="np1")

        p[0] = "s0"
        p[1] = "s1"
        with pytest.raises(KeyError):
            p[10] = "s10"

        self.assertEqual(self.graph["np1"][0], "s0")
        self.assertEqual(self.graph["np1"][1], "s1")
        self.assertEqual(self.graph["np1"][2], None)
        with pytest.raises(KeyError):
            self.assertEqual(self.graph["np1"][10], None)

        # Overwriting node properties
        p = self.graph.create_np(pname="np1", default="Not set", overwrite=True)
        self.assertEqual(self.graph["np1"][0], "Not set")
        self.assertEqual(self.graph["np1"][1], "Not set")
        self.assertEqual(self.graph["np1"][2], "Not set")
        with pytest.raises(KeyError):
            self.assertEqual(self.graph["np1"][10], "Not set")

        # Node properties via direct creation (not recommended)
        p = ggsolver.NodePMap(self.graph, pname="np1", default="default")
        self.graph["np1"] = p
        p[0] = "s0"
        self.assertEqual(self.graph["np1"][0], "s0")
        self.assertEqual(self.graph["np1"][1], "default")

        # Edge properties
        p = self.graph.create_ep(pname="ep1")
        self.assertIsInstance(p, ggsolver.EdgePMap)

        with pytest.raises(AssertionError):
            self.graph.create_ep(pname="ep1")

        p[0, 0, 0] = "e0"
        p[0, 1, 0] = "e1"
        p[0, 1, 1] = "e2"
        with pytest.raises(KeyError):
            p[0, 1, 5] = "e10"

        self.assertEqual(self.graph["ep1"][0, 0, 0], "e0")
        self.assertEqual(self.graph["ep1"][0, 1, 0], "e1")
        self.assertEqual(self.graph["ep1"][0, 1, 2], None)
        with pytest.raises(KeyError):
            self.assertEqual(self.graph["ep1"][0, 1, 5], None)

        # Overwriting edge properties
        p = self.graph.create_ep(pname="ep1", default="Not set", overwrite=True)
        self.assertEqual(self.graph["ep1"][0, 0, 0], "Not set")
        self.assertEqual(self.graph["ep1"][0, 1, 0], "Not set")
        self.assertEqual(self.graph["ep1"][0, 1, 2], "Not set")
        with pytest.raises(KeyError):
            self.assertEqual(self.graph["ep1"][0, 1, 5], None)

        # Node properties via direct creation (not recommended)
        p = ggsolver.EdgePMap(self.graph, pname="ep1", default="default")
        self.graph["ep1"] = p
        p[0, 0, 0] = "e0"
        self.assertEqual(self.graph["ep1"][0, 0, 0], "e0")
        self.assertEqual(self.graph["ep1"][0, 1, 0], "default")

    def test_serialization(self):
        # Add some nodes and edges to graph
        self.graph.add_nodes(num_nodes=5)
        self.graph.add_edges([(0, 1), (0, 1), (0, 1), (1, 2), (0, 2), (2, 3), (3, 0), (0, 0)])

        # Serialize to dictionary
        graph_dict = self.graph.serialize()
        graph1 = ggsolver.Graph().deserialize(graph_dict)
        self.assertEqual(self.graph, graph1)

    def test_pickle(self):
        # Add some nodes and edges to graph
        self.graph.add_nodes(num_nodes=5)
        self.graph.add_edges([(0, 1), (0, 1), (0, 1), (1, 2), (0, 2), (2, 3), (3, 0), (0, 0)])

        # Pickle and unpickle object
        graph_str = pickle.dumps(self.graph)
        graph1 = pickle.loads(graph_str)
        self.assertEqual(self.graph, graph1)

    def test_save_load(self):
        # Add some nodes and edges to graph
        self.graph.add_nodes(num_nodes=5)
        self.graph.add_edges([(0, 1), (0, 1), (0, 1), (1, 2), (0, 2), (2, 3), (3, 0), (0, 0)])

        # Pickle
        self.graph.save(self.fpath, protocol="pickle")
        graph1 = ggsolver.Graph().load(self.fpath, protocol="pickle")
        self.assertEqual(self.graph, graph1)

        # JSON
        with pytest.raises(FileExistsError):
            self.graph.save(self.fpath, protocol="json")

        self.graph.save(self.fpath, protocol="json", overwrite=True)
        graph1 = ggsolver.Graph().load(self.fpath, protocol="json")
        self.assertEqual(self.graph, graph1)


class TestNodeEdgePMap(unittest.TestCase):
    def setUp(self):
        self.graph = ggsolver.Graph()

        # Add some nodes and edges to graph
        self.graph.add_nodes(num_nodes=5)
        self.graph.add_edges([(0, 1), (0, 1), (0, 1), (1, 2), (0, 2), (2, 3), (3, 0), (0, 0)])

        # Graph properties
        self.graph["gp1"] = "gp1"

        # Node properties
        p = self.graph.create_np(pname="np1")
        p[0] = "s0"
        p[1] = "s1"

        # Edge properties
        p = self.graph.create_ep(pname="ep1")
        p[0, 0, 0] = "e0"
        p[0, 1, 0] = "e1"
        p[0, 1, 1] = "e2"

    def test_node_pmap_serialize(self):
        # Serialize property map
        np1 = self.graph["np1"]
        obj_dict = np1.serialize()
        print(obj_dict)

        # To deserialize, first create a node property map
        np2 = self.graph.create_np("np2")
        np2.deserialize(obj_dict)

        self.assertEqual(np1, np2)

    def test_edge_pmap_serialize(self):
        # Serialize property map
        ep1 = self.graph["ep1"]
        obj_dict = ep1.serialize()
        print(obj_dict)

        # To deserialize, first create a edge property map
        ep2 = self.graph.create_ep("ep2")
        ep2.deserialize(obj_dict)

        self.assertEqual(ep1, ep2)


class TestPMapView(unittest.TestCase):
    pass
