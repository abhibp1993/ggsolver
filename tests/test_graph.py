import itertools
import random
import unittest
import ggsolver.graph as graph
import types


class TestGraphTopology(unittest.TestCase):
    def setUp(self):
        self.graph = graph.Graph()

    def test_add_node(self):
        result = self.graph.add_node()
        self.assertEqual(result, 0)

        result = self.graph.add_node()
        self.assertEqual(result, 1)

        result = self.graph.add_node()
        self.assertEqual(result, 2)

    def test_add_nodes(self):
        result = self.graph.add_nodes(num_nodes=5)
        self.assertIsInstance(result, list)
        self.assertEqual(list(result), [0, 1, 2, 3, 4])


class TestSubGraphTopology(unittest.TestCase):
    def setUp(self):
        # Define base graph
        self.graph = graph.Graph()
        self.graph.add_nodes(num_nodes=5)
        self.edges_uv = [(1, 3), (3, 0), (3, 2), (0, 1)]
        self.edge_keys = self.graph.add_edges(self.edges_uv)
        self.edges = itertools.product(self.edges_uv, self.edge_keys)

        # Create subgraph
        self.subgraph = graph.SubGraph(self.graph)

    def test_add_nodes_no_hidden(self):
        # Check the state of subgraph
        self.assertEqual(list(self.subgraph.nodes()), list(range(5)))
        self.assertEqual(self.subgraph.number_of_nodes(), 5)

        # Add more nodes to graph
        self.graph.add_nodes(num_nodes=5)

        # Subgraph should see the new nodes
        self.assertEqual(list(self.subgraph.nodes()), list(range(10)))
        self.assertEqual(self.subgraph.number_of_nodes(), 10)

    def test_add_nodes_hidden(self):
        # Hide some states
        self.subgraph.hide_nodes([2, 3, 4])

        # Check the state of subgraph
        self.assertEqual(list(self.subgraph.nodes()), list(range(2)))
        self.assertEqual(self.subgraph.number_of_nodes(), 2)

        # Add more nodes to graph
        self.graph.add_nodes(num_nodes=5)

        # New nodes should be visible
        self.assertEqual(list(self.subgraph.nodes()), list(range(2)) + list(range(5, 10)))
        self.assertEqual(self.subgraph.number_of_nodes(), 7)

    def test_add_edges_no_hidden(self):
        # Check state of subgraph
        self.assertEqual(
            {(u, v) for u, v, _ in self.subgraph.edges()},
            {(1, 3), (3, 0), (3, 2), (0, 1)}
        )
        self.assertEqual(self.subgraph.number_of_edges(), 4)

        # Add more edges
        self.graph.add_edges([(1, 0), (2, 1), (1, 2), (2, 0), (2, 3), (4, 1)])

        # New state of subgraph
        self.assertEqual(
            {(u, v) for u, v, _ in self.subgraph.edges()},
            {(1, 3), (3, 0), (3, 2), (0, 1), (1, 0), (2, 1), (1, 2), (2, 0), (2, 3), (4, 1)}
        )
        self.assertEqual(self.subgraph.number_of_edges(), 10)

    def test_add_edges_hidden(self):
        # Hide some edges
        self.subgraph.hide_edges([(u, v, k) for (u, v), k in self.edges if (u, v) in [(1, 3), (3, 2)]])

        # Check state of subgraph
        self.assertEqual(
            {(u, v) for u, v, _ in self.subgraph.edges()},
            {(3, 0), (0, 1)}
        )
        self.assertEqual(self.subgraph.number_of_edges(), 2)

        # Add more edges
        self.graph.add_edges([(1, 0), (2, 1), (1, 2), (2, 0), (2, 3), (4, 1)])

        # New edges should be visible
        self.assertEqual(
            {(u, v) for u, v, _ in self.subgraph.edges()},
            {(3, 0), (0, 1), (1, 0), (2, 1), (1, 2), (2, 0), (2, 3), (4, 1)}
        )
        self.assertEqual(self.subgraph.number_of_edges(), 8)

    def test_node_edge_containment(self):
        # Hide some nodes
        self.subgraph.hide_nodes([1])

        # Check the node in subgraph
        self.assertTrue(self.subgraph.has_node(0))
        self.assertFalse(self.subgraph.has_node(1))

        # Hide some edges
        self.subgraph.hide_edge(3, 2, 0)

        # Check the edge in subgraph
        self.assertFalse(self.subgraph.has_edge(3, 2, 0))       # Because we removed it explicitly
        self.assertFalse(self.subgraph.has_edge(1, 3, 0))       # Because we removed node:1
        self.assertFalse(self.subgraph.has_edge(0, 1, 0))       # Because we removed node:1
        self.assertTrue(self.subgraph.has_edge(3, 0, 0))        # Only surviving edge

    def test_neighborhood(self):
        # Hide some nodes (only edge [(3, 0), (3, 2)] will remain in subgraph)
        self.subgraph.hide_nodes([1])

        # In-edges of 0: (3, 0)
        self.assertEqual(set(self.subgraph.in_edges(0)), {(3, 0, 0)})

        # In-edges of 1: raises error
        with self.assertRaises(KeyError) as context:
            self.subgraph.in_edges(1)
        self.assertIsInstance(context.exception, KeyError)

        # Out-edges of 3: (3, 0), (3, 2)
        self.assertEqual(set(self.subgraph.out_edges(3)), {(3, 0, 0), (3, 2, 0)})

        # Out-edges of 1: raises error
        with self.assertRaises(KeyError) as context:
            self.subgraph.out_edges(1)
        self.assertIsInstance(context.exception, KeyError)

        # In-neighbors of 0: 3
        self.assertEqual(set(self.subgraph.predecessors(0)), {3})

        # In-neighbors of 1: raises error
        with self.assertRaises(KeyError) as context:
            self.subgraph.predecessors(1)
        self.assertIsInstance(context.exception, KeyError)

        # Out-neighbors of 3: 0, 2
        self.assertEqual(set(self.subgraph.successors(3)), {0, 2})

        # Out-neighbors of 1: raises error
        with self.assertRaises(KeyError):
            self.subgraph.successors(1)
        self.assertIsInstance(context.exception, KeyError)


class TestSubGraphBranching(unittest.TestCase):
    def setUp(self):
        # Define a `graph`. (Using Jobstmann graph)
        self.graph = graph.Graph()
        self.nodes = self.graph.add_nodes(8)
        edge_uv = [(0, 1), (0, 3), (1, 0), (1, 2), (1, 4), (2, 4), (2, 2), (3, 0), (3, 4), (3, 5), (4, 1), (4, 3),
                   (5, 3), (5, 6), (6, 6), (6, 7), (7, 0), (7, 3)]
        edge_keys = self.graph.add_edges(edge_uv)
        self.edges = [(edge_uv[i][0], edge_uv[i][1], edge_keys[i]) for i in range(len(edge_keys))]

        # Create subgraph `sg1` of `graph`.
        self.subgraph1 = graph.SubGraph(self.graph)
        self.subgraph1.hide_nodes([0, 1])
        self.subgraph1.hide_edges([(7, 0, 0), (6, 7, 0)])

        # Create subgraph `sg2` of subgraph `graph`.
        self.subgraph2 = graph.SubGraph(self.graph)
        self.subgraph2.hide_nodes([2, 3])
        self.subgraph2.hide_edges([(5, 6, 0)])

    def test_nodes(self):
        # Check if hidden node of sg1 is hidden in sg1.
        self.assertEqual(list(self.subgraph1.nodes()), [2, 3, 4, 5, 6, 7])

        # Check if hidden node of sg2 is hidden in sg2.
        self.assertEqual(list(self.subgraph2.nodes()), [0, 1, 4, 5, 6, 7])

    def test_edges(self):
        # Check if hidden edges of sg1 is hidden in sg1.
        self.assertEqual(
            {(u, v) for u, v, _ in self.subgraph1.edges()},
            {(2, 4), (2, 2), (3, 4), (3, 5), (4, 3), (5, 3), (5, 6), (6, 6), (7, 3)})

        # Check if hidden edges of sg2 is hidden in sg2.
        self.assertEqual(
            {(u, v) for u, v, _ in self.subgraph2.edges()},
            {(0, 1), (1, 0), (1, 4), (4, 1), (6, 6), (6, 7), (7, 0)})

    @unittest.skip
    def test_serialize(self):
        # Ensure hidden_nodes and hidden_edges are handled properly for sg1.
        serialized_subgraph1 = self.subgraph1.serialize()

        # Ensure hidden_nodes and hidden_edges are handled properly for sg2.
        serialized_subgraph2 = self.subgraph2.serialize()

        # Ensure hidden_nodes and hidden_edges of sg1, sg2 do not interfere with each other.
        pass


class TestSubGraphHierarchy(unittest.TestCase):
    def setUp(self):
        # Define a `graph`. (Using Jobstmann graph)
        self.graph = graph.Graph()
        self.nodes = self.graph.add_nodes(8)
        edge_uv = [(0, 1), (0, 3), (1, 0), (1, 2), (1, 4), (2, 4), (2, 2), (3, 0), (3, 4), (3, 5), (4, 1), (4, 3),
                   (5, 3), (5, 6), (6, 6), (6, 7), (7, 0), (7, 3)]
        edge_keys = self.graph.add_edges(edge_uv)
        self.edges = [(edge_uv[i][0], edge_uv[i][1], edge_keys[i]) for i in range(len(edge_keys))]

        # Create subgraph `sg1` of `graph`.
        self.subgraph1 = graph.SubGraph(self.graph)
        self.subgraph1.hide_nodes([0, 1])
        self.subgraph1.hide_edges([(7, 0, 0), (6, 7, 0)])

        # Create subgraph `sg2` of subgraph `graph`.
        self.subgraph2 = graph.SubGraph(self.subgraph1)
        self.subgraph2.hide_nodes([2, 3])
        self.subgraph2.hide_edges([(5, 6, 0)])

    def test_nodes(self):
        # Check if hidden node of sg1 is hidden in sg1.
        self.assertEqual(list(self.subgraph1.nodes()), [2, 3, 4, 5, 6, 7])
        self.assertEqual(self.subgraph1.number_of_nodes(), 6)

        # Check if hidden node of sg2 is hidden in sg2.
        self.assertEqual(list(self.subgraph2.nodes()), [4, 5, 6, 7])
        self.assertEqual(self.subgraph2.number_of_nodes(), 4)

    def test_edges(self):
        # Check if hidden edges of sg1 is hidden in sg1.
        self.assertEqual(
            {(u, v) for u, v, _ in self.subgraph1.edges()},
            {(2, 4), (2, 2), (3, 4), (3, 5), (4, 3), (5, 3), (5, 6), (6, 6), (7, 3)})
        self.assertEqual(self.subgraph1.number_of_edges(), 9)

        # Check if hidden edges of sg2 is hidden in sg2.
        self.assertEqual(
            {(u, v) for u, v, _ in self.subgraph2.edges()},
            {(6, 6)})
        self.assertEqual(self.subgraph2.number_of_edges(), 1)

    @unittest.skip
    def test_serialize(self):
        # Ensure hidden_nodes and hidden_edges are handled properly for sg1.
        serialized_subgraph1 = self.subgraph1.serialize()

        # Ensure hidden_nodes and hidden_edges are handled properly for sg2.
        serialized_subgraph2 = self.subgraph2.serialize()

        # Ensure hidden_nodes and hidden_edges of sg1, sg2 do not interfere with each other.
        pass


class TestSubGraphProperties(unittest.TestCase):
    def setUp(self):
        # Define a `graph`. (Using Jobstmann graph)
        self.graph = graph.Graph()
        self.nodes = self.graph.add_nodes(8)
        edge_uv = [(0, 1), (0, 3), (1, 0), (1, 2), (1, 4), (2, 4), (2, 2), (3, 0), (3, 4), (3, 5), (4, 1), (4, 3),
                   (5, 3), (5, 6), (6, 6), (6, 7), (7, 0), (7, 3)]
        edge_keys = self.graph.add_edges(edge_uv)
        self.edges = [(edge_uv[i][0], edge_uv[i][1], edge_keys[i]) for i in range(len(edge_keys))]

        # Set node and edge property
        self.graph["node_winner"] = graph.NodePropertyMap(self.graph, default=None)
        for uid in self.graph.nodes():
            self.graph["node_winner"][uid] = 1

        self.graph["edge_winner"] = graph.EdgePropertyMap(self.graph, default=None)
        for edge in self.graph.edges():
            self.graph["edge_winner"][edge] = 1

        # Create subgraph `sg1` of `graph`.
        self.subgraph1 = graph.SubGraph(self.graph)
        self.subgraph1.hide_nodes([0, 1])
        self.subgraph1.hide_edges([(7, 0, 0), (6, 7, 0)])

    def test_sg_inherits_properties(self):
        # On creating a subgraph, SG automatically has property `node_winner`
        self.assertTrue("node_winner" in self.graph.node_properties)
        self.assertTrue("edge_winner" in self.graph.edge_properties)

        self.assertTrue("node_winner" in self.subgraph1.node_properties)
        self.assertTrue("edge_winner" in self.subgraph1.edge_properties)

        self.assertIsInstance(self.subgraph1["node_winner"], graph.PMapView)
        self.assertIsInstance(self.subgraph1["edge_winner"], graph.PMapView)

    def test_change_in_g_reflects_in_sg(self):
        # If value of property in G is changed, it reflects in SG.
        self.graph["node_winner"][0] = 2
        self.assertEqual(self.subgraph1["node_winner"][0], 2)

        self.graph["edge_winner"][0, 1, 0] = 2
        self.assertEqual(self.subgraph1["edge_winner"][0, 1, 0], 2)

    def test_sg_property_is_readonly(self):
        # The value cannot directly be changed in SG
        with self.assertRaises(PermissionError) as context:
            self.subgraph1["node_winner"][0] = 2
        self.assertIsInstance(context.exception, PermissionError)

        with self.assertRaises(PermissionError) as context:
            self.subgraph1["edge_winner"][0, 1, 0] = 2
        self.assertIsInstance(context.exception, PermissionError)

    def test_sg_property_modification_type1(self):
        # To modify the property, user has two options.
        # 1. If user wants to maintain the same property name
        self.subgraph1.make_property_local("node_winner")   # Creates a local copy of PropertyMap for SG.
        self.subgraph1.make_property_local("edge_winner")   # Creates a local copy of PropertyMap for SG.
        self.assertIsInstance(self.subgraph1["node_winner"], graph.NodePropertyMap)
        self.assertIsInstance(self.subgraph1["edge_winner"], graph.EdgePropertyMap)

        # SG["node_winner"][uid] returns G["node_winner"][uid]
        self.assertEqual(self.subgraph1["node_winner"][5], self.graph["node_winner"][5])
        self.assertEqual(self.subgraph1["edge_winner"][5, 3, 0], self.graph["edge_winner"][5, 3, 0])

        # On updating value of G["node_property"], no changes are reflected in SG property.
        self.graph["node_winner"][5] = 2
        self.assertEqual(self.graph["node_winner"][5], 2)
        self.assertEqual(self.subgraph1["node_winner"][5], 1)

        self.graph["edge_winner"][5, 3, 0] = 2
        self.assertEqual(self.graph["edge_winner"][5, 3, 0], 2)
        self.assertEqual(self.subgraph1["edge_winner"][5, 3, 0], 1)

        # On updating value of SG["node_property"], no changes are reflected in G property.
        self.subgraph1["node_winner"][6] = 2
        self.assertEqual(self.graph["node_winner"][6], 1)
        self.assertEqual(self.subgraph1["node_winner"][6], 2)

        self.subgraph1["edge_winner"][4, 3, 0] = 2
        self.assertEqual(self.graph["edge_winner"][4, 3, 0], 1)
        self.assertEqual(self.subgraph1["edge_winner"][4, 3, 0], 2)

    def test_sg_property_modification_type2(self):
        # 2. Create a new property
        self.subgraph1["SG.node_winner"] = graph.NodePropertyMap(self.subgraph1, self.graph["node_winner"].default)
        self.subgraph1["SG.edge_winner"] = graph.EdgePropertyMap(self.subgraph1, self.graph["edge_winner"].default)

        self.assertEqual(self.subgraph1["SG.node_winner"].default, self.graph["node_winner"].default)
        self.assertEqual(self.subgraph1["SG.edge_winner"].default, self.graph["edge_winner"].default)

        # Manually copy the data (can be modified by user, if necessary)
        for uid in self.subgraph1.nodes():
            self.subgraph1["SG.node_winner"][uid] = self.graph["node_winner"][uid]

        for edge in self.subgraph1.edges():
            self.subgraph1["SG.edge_winner"][edge] = self.graph["edge_winner"][edge]

        # The new properties are PropertyMaps, while the parent graph's properties are still accessible.
        self.assertIsInstance(self.subgraph1["SG.node_winner"], graph.NodePropertyMap)
        self.assertIsInstance(self.subgraph1["SG.edge_winner"], graph.EdgePropertyMap)

        self.assertIsInstance(self.subgraph1["node_winner"], graph.PMapView)
        self.assertIsInstance(self.subgraph1["edge_winner"], graph.PMapView)


if __name__ == '__main__':
    unittest.main()
