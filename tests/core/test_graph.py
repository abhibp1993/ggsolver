import inspect
import itertools
import random
import unittest

import pytest

import ggsolver
import types
from collections.abc import Iterator


class TestGraphTopology(unittest.TestCase):
    def setUp(self):
        self.graph = ggsolver.Graph()

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
        pass

    def test_serialization(self):
        pass

