import ggsolver
import ggsolver.ioutils as io
import unittest
import pathlib
import shutil


class TestGraphToDot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create folder if it does not exist.
        if not pathlib.Path("out/").exists():
            pathlib.Path("out/").mkdir()

    @classmethod
    def tearDownClass(cls):
        # shutil.rmtree("out/")
        pass

    def setUp(self):
        self.graph = ggsolver.Graph()

        self.graph.add_nodes(5)
        self.graph.add_edges([(0, 1), (0, 2), (0, 3), (1, 4), (2, 4)])

        np_state = self.graph.create_np("state")
        np_turn = self.graph.create_np("turn")
        np_act = self.graph.create_ep("act")

        for i in range(5):
            np_state[i] = f"s{i}"
            if i % 2 == 0:
                np_turn[i] = 1
            else:
                np_turn[i] = 2
        for u, v, k in self.graph.edges():
            np_act[u, v, k] = f"a{(u, v, k)}"

        self.fpath_dot = str(pathlib.Path("out/graph.dot").absolute())
        self.fpath_png = str(pathlib.Path("out/graph.png").absolute())
        self.fpath_pdf = str(pathlib.Path("out/graph.pdf").absolute())
        self.fpath_svg = str(pathlib.Path("out/graph.svg").absolute())

        self.fpath_dot2 = str(pathlib.Path("out/graph2.dot").absolute())
        self.fpath_png2 = str(pathlib.Path("out/graph2.png").absolute())
        self.fpath_pdf2 = str(pathlib.Path("out/graph2.pdf").absolute())
        self.fpath_svg2 = str(pathlib.Path("out/graph2.svg").absolute())

    def test_graph_to_dot(self):
        io.to_dot(self.fpath_dot, self.graph)
        io.dot2png(self.fpath_dot, self.fpath_png)
        io.dot2pdf(self.fpath_dot, self.fpath_pdf)
        io.dot2svg(self.fpath_dot, self.fpath_svg)

    def test_graph_to_dot2(self):
        io.to_dot(self.fpath_dot2, self.graph, node_props=["state"], edge_props=["act"])
        io.dot2png(self.fpath_dot2, self.fpath_png2)
        io.dot2pdf(self.fpath_dot2, self.fpath_pdf2)
        io.dot2svg(self.fpath_dot2, self.fpath_svg2)


class TestAutToDot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create folder if it does not exist.
        if not pathlib.Path("out/").exists():
            pathlib.Path("out/").mkdir()

    @classmethod
    def tearDownClass(cls):
        # shutil.rmtree("out/")
        pass

    def setUp(self):
        self.graph = ggsolver.Graph()

        self.graph.add_nodes(5)
        self.graph.add_edges([(0, 1), (0, 2), (0, 3), (1, 4), (2, 4)])

        np_final = self.graph.create_np("final")
        np_state = self.graph.create_np("state", default="s")
        np_formula = self.graph.create_ep("formula")
        np_tmp = self.graph.create_ep("tmp", default="a")
        self.graph["init_state"] = 0

        for i in range(5):
            if i % 2 == 0:
                np_final[i] = (0, 1)
            else:
                np_final[i] = tuple()
        for u, v, k in self.graph.edges():
            np_formula[u, v, k] = f"a | b"

        self.fpath_dot = str(pathlib.Path("out/aut.dot").absolute())
        self.fpath_png = str(pathlib.Path("out/aut.png").absolute())
        self.fpath_pdf = str(pathlib.Path("out/aut.pdf").absolute())
        self.fpath_svg = str(pathlib.Path("out/aut.svg").absolute())

        self.fpath_dot2 = str(pathlib.Path("out/aut2.dot").absolute())
        self.fpath_png2 = str(pathlib.Path("out/aut2.png").absolute())
        self.fpath_pdf2 = str(pathlib.Path("out/aut2.pdf").absolute())
        self.fpath_svg2 = str(pathlib.Path("out/aut2.svg").absolute())

    def test_aut_to_dot(self):
        io.to_dot(self.fpath_dot, self.graph, formatting="aut")
        io.dot2png(self.fpath_dot, self.fpath_png)
        io.dot2pdf(self.fpath_dot, self.fpath_pdf)
        io.dot2svg(self.fpath_dot, self.fpath_svg)

    def test_aut_to_dot2(self):
        io.to_dot(self.fpath_dot2, self.graph, node_props=["state"], edge_props=["tmp"], formatting="aut")
        io.dot2png(self.fpath_dot2, self.fpath_png2)
        io.dot2pdf(self.fpath_dot2, self.fpath_pdf2)
        io.dot2svg(self.fpath_dot2, self.fpath_svg2)




