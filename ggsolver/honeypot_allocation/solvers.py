import os
import typing
import pygraphviz
import pickle

import ggsolver.graph as ggraph
import ggsolver.models as models
from ggsolver.dtptb.dtptb_reach import SWinReach
from ggsolver.honeypot_allocation import util
from typing import Union
from loguru import logger
from tqdm import tqdm

MAX_COMBINATIONS = 2500


class DSWinReach(models.Solver):
    def __init__(self, game_graph: ggraph.Graph, traps: Union[list, set, tuple], fakes: Union[list, set, tuple], debug=False,
                 **kwargs):
        super(DSWinReach, self).__init__(game_graph)

        # Game parameters
        self._traps = set(traps)
        self._fakes = set(fakes)
        self._hypergame_graph = None
        self._vod = 0.0

        # Intermediate outputs
        self._debug = debug
        self._base_game_solution = None
        self._p2_game_solution = None
        self._hgame_solution = None
        self._base2hg_nodes = None
        self._base2hg_edges = None
        self._hg2base_nodes = None
        self._hg2base_edges = None

        # Directory setup
        self._path = kwargs.get("path", "out/")
        self._filename = kwargs.get("filename", "dswin")

    def solve(self, skip_solution=False):
        # 1. Solve game. (V, E, F)
        base_game_solution = self.solve_base_game()
        # logger.info("Base game solved.")

        # 2. Construct and solve G2.
        true_final = {uid for uid in self._solution.nodes() if self.graph()["final"][uid]}
        p2_game_solution = self.solve_p2_game(true_final)
        # logger.info("P2 game solved.")

        # 3. SR_E = {e \in E | e is subjectively rationalizable in G2}
        sr_edges = self.sr_edges()
        # sr_edges = {
        #     (u, v, k) for u, v, k in self._solution.edges()
        #     if base_game_solution.node_winner(u) == 2 and (
        #             (self._solution["turn"][u] == 1) or
        #             (self._solution["turn"][u] == 2 and p2_game_solution.edge_winner(u, v, k) == 2)
        #     )
        # }
        # logger.info("SREdges identified.")

        # 4. Construct H. (Win2, SR_E, X U Y)
        hgame_graph, base2hg_nodes, hg2base_nodes, base2hg_edges, hg2base_edges = \
            self._construct_hypergame(base_game_solution, sr_edges)
        hgame_solution = SWinReach(hgame_graph, player=1, path=self._path, filename=f"{self._filename}_hgame", tqdm_disable=True)
        hgame_solution.solve()
        self._hypergame_graph = hgame_graph
        # logger.info("Hypergame solved.")

        # 5. Compute VoD
        dswin1 = set(hgame_solution.winning_nodes(player=1))
        win2 = set(base_game_solution.winning_nodes(player=2))
        self._vod = len(set.intersection({hg2base_nodes[node] for node in dswin1}, win2)) / len(win2 - true_final)
        # logger.info(f"VoD: {self._vod}.")

        # 6. Fix P1 and P2 strategies. Mark node and edge winners
        sources = {base2hg_nodes[node] for node in self._fakes | self._traps}
        reachable_nodes = set()
        if len(sources) > 0:
            hg_reachable_nodes = hgame_graph.reverse_bfs(sources=sources)
            reachable_nodes = {hg2base_nodes[node] for node in hg_reachable_nodes}

        if not skip_solution:
            for node in tqdm(list(self._solution.nodes()), desc="Processing nodes for DSWin solution"):
                # P1 wins a node if it is winning for P1 in base-game or is deceptively winning in hypergame.
                if base_game_solution.node_winner(node) == 1:
                    self._solution["node_winner"][node] = 1
                    for u, v, k in self._solution.out_edges(node):
                        self._solution["edge_winner"][u, v, k] = base_game_solution.edge_winner(u, v, k)

                elif hgame_graph.has_node(node) and hgame_solution.node_winner(node) == 1:
                    self._solution["node_winner"][node] = 1
                    for u, v, k in self._solution.out_edges(node):
                        try:
                            self._solution["edge_winner"][u, v, k] = hgame_solution.edge_winner(u, v, k)
                        except KeyError:
                            # Case 1. Edge leaves Win2.
                            if v not in win2:
                                self._solution["edge_winner"][u, v, k] = 1
                            # Case 2. Edge is not rank-decreasing, therefore, not subjectively rationalizable.
                            else:
                                self._solution["edge_winner"][u, v, k] = 0

                else:  # When P1 is not winning...
                    # If node is connected with traps or fakes, winner is undetermined.
                    self._solution["node_winner"][node] = 3  # PATCH. Temporry for coloring.

                    # if node in reachable_nodes:
                    #     self._solution["node_winner"][node] = 0
                    # else:
                    #     self._solution["node_winner"][node] = 2

        # Save intermediate solutions for debugging
        self._base_game_solution = base_game_solution
        self._p2_game_solution = p2_game_solution
        self._hgame_solution = hgame_solution
        self._base2hg_nodes = base2hg_nodes
        self._base2hg_edges = base2hg_edges
        self._hg2base_nodes = hg2base_nodes
        self._hg2base_edges = hg2base_edges
        if self._debug:
            self._save_debug_output()

        self._is_solved = True

    def solve_base_game(self, save_svg=False, path=None, filename=None):
        base_game_solution = SWinReach(self._solution, player=2, path=self._path, filename=f"{self._filename}_base_game",
                                       tqdm_disable=True)
        base_game_solution.solve()
        self._base_game_solution = base_game_solution

        if save_svg:
            assert path is not None and filename is not None
            util.write_dot_file(self._base_game_solution.solution(), path=path, filename=filename)

        return self._base_game_solution

    def solve_p2_game(self, true_final, save_svg=False, path=None, filename=None):
        p2_final = true_final | self._fakes
        p2_game_solution = SWinReach(self._solution, player=2, final=p2_final, path=self._path, filename=f"{self._filename}_p2_game",
                                     tqdm_disable=True)
        p2_game_solution.solve()
        self._p2_game_solution = p2_game_solution

        if save_svg:
            assert path is not None and filename is not None
            util.write_dot_file(self._p2_game_solution.solution(), path=path, filename=filename)

        return p2_game_solution

    def sr_edges(self):
        # SR_E = {e \in E | e.source is in Win2(G, F) AND e is SR(G2)} | {e \in E | e.source is in Win1(G, F)}
        sr_edges = set()
        for u, v, k in self._solution.edges():
            if self._base_game_solution.node_winner(u) == 2:
                if ((self._solution["turn"][u] == 1) or (
                        self._solution["turn"][u] == 2 and self._p2_game_solution.edge_winner(u, v, k) == 2)):
                    sr_edges.add((u, v, k))

                #     sr_edges = {
                #     (u, v, k)
                #     for u, v, k in self._solution.edges()
                #         if self._base_game_solution.node_winner(u) == 2 and (
                #                 (self._solution["turn"][u] == 1) or
                #                 (self._solution["turn"][u] == 2 and self._p2_game_solution.edge_winner(u, v, k) == 2)
                #         )
                # }
                # | {
                #     (u, v, k) for u, v, k in self._solution.edges()
                #     if self._base_game_solution.node_winner(u) == 1
                # }

        return sr_edges

    def _construct_hypergame(self, base_game_solution, sr_edges):
        hypergame_graph = ggraph.Graph()
        hypergame_graph["is_deterministic"] = True
        hypergame_graph["is_turn_based"] = True

        # Add nodes and store map of hgame <-> graph nodes.
        num_nodes = len(base_game_solution.winning_nodes(2))
        hg_nodes = hypergame_graph.add_nodes(num_nodes)
        base_hg_node_map = dict(zip(base_game_solution.winning_nodes(2), hg_nodes))
        hg_base_node_map = dict(zip(hg_nodes, base_game_solution.winning_nodes(2)))

        # Add edges (key map is maintained)
        hg_base_edge_map = dict()
        base_hg_edge_map = dict()
        for u, v, k in sr_edges:
            if u in base_game_solution.final():
                hg_uid = base_hg_node_map[u]
                hg_key = hypergame_graph.add_edge(hg_uid, hg_uid)
                base_hg_edge_map[u, v, k] = (hg_uid, hg_uid, hg_key)
                hg_base_edge_map[hg_uid, hg_uid, hg_key] = (u, v, k)
            else:
                hg_uid = base_hg_node_map[u]
                hg_vid = base_hg_node_map[v]
                hg_key = hypergame_graph.add_edge(hg_uid, hg_vid)
                base_hg_edge_map[u, v, k] = (hg_uid, hg_vid, hg_key)
                hg_base_edge_map[hg_uid, hg_vid, hg_key] = (u, v, k)

        # Ensure no spurious state was added while adding edges
        if hypergame_graph.number_of_nodes() != num_nodes:
            raise RuntimeError("Number of nodes changed while adding edges.")

        # Add state property
        hypergame_graph.create_node_property("state", None)
        for uid in hypergame_graph.nodes():
            hypergame_graph["state"][uid] = self._solution["state"][hg_base_node_map[uid]]

        # Add turn property
        hypergame_graph.create_node_property("turn", None)
        for uid in hypergame_graph.nodes():
            hypergame_graph["turn"][uid] = self._solution["turn"][hg_base_node_map[uid]]

        # Add final property
        hypergame_graph.create_node_property("final", False)
        for uid in self._fakes | self._traps:
            hypergame_graph["final"][base_hg_node_map[uid]] = True

        return hypergame_graph, base_hg_node_map, hg_base_node_map, base_hg_edge_map, hg_base_edge_map

    def _save_debug_output(self):
        util.write_dot_file(self._base_game_solution.solution(), path=self._path, filename=f"{self._filename}_base_game")
        util.write_dot_file(self._p2_game_solution.solution(), path=self._path, filename=f"{self._filename}_p2_game")
        util.write_dot_file(self._hgame_solution.solution(), path=self._path, filename=f"{self._filename}_hgame")

    def save_svg(self, path, filename, **kwargs):
        # Generate DOT file.
        fpath = os.path.join(path, f"{filename}.dot")
        with open(fpath, 'w') as file:
            contents = list()
            contents.append("digraph G {\noverlap=scale;\n")

            for node in self._solution.nodes():
                node_properties = {
                    # "shape": 'circle' if self._solution['turn'][node] == 1 else 'box',
                    "width": 2,
                    "height": 2,
                    "style": kwargs.get("node_style", "solid"),
                    "label": self._solution['state'][node],
                    "peripheries": '2' if self._solution['final'][node] else '1',
                }

                if node in self._traps | self._fakes:
                    node_properties["shape"] = "diamond"
                else:
                    node_properties["shape"] = 'circle' if self._solution['turn'][node] == 1 else 'box'

                if "node_winner" in self._solution.node_properties:
                    if self._base_game_solution.node_winner(node) == 1:
                        node_properties |= {"color": 'blue'}
                    # elif self._solution['node_winner'][self._base2hg_nodes[node]] == 1:
                    elif self._hgame_solution.node_winner(self._base2hg_nodes[node]) == 1:
                        node_properties |= {"color": 'green'}
                    else:
                        node_properties |= {"color": 'red'}

                contents.append(
                    f"N{node} [" + ", ".join(f'{k}="{v}"' for k, v in node_properties.items()) + "];\n"
                )

                for uid, vid, key in self._solution.out_edges(node):
                    edge_properties = {
                        "label": self._solution["input"][uid, vid, key] if kwargs.get("show_actions", False) else ""
                    }
                    # if "edge_winner" in self._solution.edge_properties:
                    #     if self._solution['edge_winner'][uid, vid, key] == 1:
                    #         edge_properties |= {"color": 'blue'}
                    #     elif self._solution['edge_winner'][uid, vid, key] == 2:
                    #         edge_properties |= {"color": 'red'}
                    #     else:
                    #         edge_properties |= {"color": 'black'}

                    contents.append(
                        f"N{uid} -> N{vid} [" + ", ".join(f'{k}="{v}"' for k, v in edge_properties.items()) + "];\n"
                    )

            contents.append("}")
            file.writelines(contents)

        # Generate SVG
        g = pygraphviz.AGraph(fpath)
        if self._solution.number_of_nodes() + self._solution.number_of_edges() > 200:
            logger.warning(f"Graph size is larger than 200, using Force-Directed Layout. Generating PNG instead of SVG.")
            g.layout('sfdp')
            path = os.path.join(path, f"{filename}.png")
            g.draw(path=path, format='png')
        else:
            g.layout('dot')
            path = os.path.join(path, f"{filename}.svg")
            g.draw(path=path, format='svg')

    def vod(self):
        return self._vod


class DASWinReach(models.Solver):
    def __init__(self, game_graph: ggraph.Graph, traps: Union[list, set, tuple], fakes: Union[list, set, tuple], debug=False,
                 **kwargs):
        super(DASWinReach, self).__init__(game_graph)

        # Game parameters
        self._traps = set(traps)
        self._fakes = set(fakes)
        self._hypergame_graph = None
        self._vod = 0.0

        # Intermediate outputs
        self._debug = debug
        self._base_game_solution = None
        self._p2_game_solution = None
        self._hgame_solution = None
        self._base2hg_nodes = None
        self._base2hg_edges = None
        self._hg2base_nodes = None
        self._hg2base_edges = None

        # Directory setup
        self._path = kwargs.get("path", "out/")
        self._filename = kwargs.get("filename", "daswin")

    def solve(self, skip_solution=False):
        # 1. Solve game. (V, E, F)
        base_game_solution = self.solve_base_game()
        # logger.info("Base game solved.")

        # 2. Construct and solve G2.
        true_final = {uid for uid in self._solution.nodes() if self.graph()["final"][uid]}
        p2_game_solution = self.solve_p2_game(true_final)
        # logger.info("P2 game solved.")

        # 3. SR_E = {e \in E | e is subjectively rationalizable in G2}
        sr_edges = self.sr_edges()
        # sr_edges = {
        #     (u, v, k) for u, v, k in self._solution.edges()
        #     if base_game_solution.node_winner(u) == 2 and (
        #             (self._solution["turn"][u] == 1) or
        #             (self._solution["turn"][u] == 2 and p2_game_solution.edge_winner(u, v, k) == 2)
        #     )
        # }
        # logger.info("SREdges identified.")

        # 4. Construct H. (Win2, SR_E, X U Y)
        hgame_graph, base2hg_nodes, hg2base_nodes, base2hg_edges, hg2base_edges = \
            self._construct_hypergame(base_game_solution, sr_edges)
        hgame_solution = SWinReach(hgame_graph, player=1, path=self._path, filename=f"{self._filename}_hgame", tqdm_disable=True)
        hgame_solution.solve()
        self._hypergame_graph = hgame_graph
        # logger.info("Hypergame solved.")

        # 5. Compute VoD
        dswin1 = set(hgame_solution.winning_nodes(player=1))
        win2 = set(base_game_solution.winning_nodes(player=2))
        self._vod = len(set.intersection({hg2base_nodes[node] for node in dswin1}, win2)) / len(win2 - true_final)
        # logger.info(f"VoD: {self._vod}.")

        # 6. Fix P1 and P2 strategies. Mark node and edge winners
        sources = {base2hg_nodes[node] for node in self._fakes | self._traps}
        reachable_nodes = set()
        if len(sources) > 0:
            hg_reachable_nodes = hgame_graph.reverse_bfs(sources=sources)
            reachable_nodes = {hg2base_nodes[node] for node in hg_reachable_nodes}

        if not skip_solution:
            for node in tqdm(list(self._solution.nodes()), desc="Processing nodes for DSWin solution"):
                # P1 wins a node if it is winning for P1 in base-game or is deceptively winning in hypergame.
                if base_game_solution.node_winner(node) == 1:
                    self._solution["node_winner"][node] = 1
                    for u, v, k in self._solution.out_edges(node):
                        self._solution["edge_winner"][u, v, k] = base_game_solution.edge_winner(u, v, k)

                elif hgame_graph.has_node(node) and hgame_solution.node_winner(node) == 1:
                    self._solution["node_winner"][node] = 1
                    for u, v, k in self._solution.out_edges(node):
                        try:
                            self._solution["edge_winner"][u, v, k] = hgame_solution.edge_winner(u, v, k)
                        except KeyError:
                            # Case 1. Edge leaves Win2.
                            if v not in win2:
                                self._solution["edge_winner"][u, v, k] = 1
                            # Case 2. Edge is not rank-decreasing, therefore, not subjectively rationalizable.
                            else:
                                self._solution["edge_winner"][u, v, k] = 0

                else:  # When P1 is not winning...
                    # If node is connected with traps or fakes, winner is undetermined.
                    self._solution["node_winner"][node] = 3  # PATCH. Temporry for coloring.

                    # if node in reachable_nodes:
                    #     self._solution["node_winner"][node] = 0
                    # else:
                    #     self._solution["node_winner"][node] = 2

        # Save intermediate solutions for debugging
        self._base_game_solution = base_game_solution
        self._p2_game_solution = p2_game_solution
        self._hgame_solution = hgame_solution
        self._base2hg_nodes = base2hg_nodes
        self._base2hg_edges = base2hg_edges
        self._hg2base_nodes = hg2base_nodes
        self._hg2base_edges = hg2base_edges
        if self._debug:
            self._save_debug_output()

        self._is_solved = True

    def solve_base_game(self, save_svg=False, path=None, filename=None):
        base_game_solution = SWinReach(self._solution, player=2, path=self._path, filename=f"{self._filename}_base_game",
                                       tqdm_disable=True)
        base_game_solution.solve()
        self._base_game_solution = base_game_solution

        if save_svg:
            assert path is not None and filename is not None
            util.write_dot_file(self._base_game_solution.solution(), path=path, filename=filename)

        return self._base_game_solution

    def solve_p2_game(self, true_final, save_svg=False, path=None, filename=None):
        p2_final = true_final | self._fakes
        p2_game_solution = SWinReach(self._solution, player=2, final=p2_final, path=self._path, filename=f"{self._filename}_p2_game",
                                     tqdm_disable=True)
        p2_game_solution.solve()
        self._p2_game_solution = p2_game_solution

        if save_svg:
            assert path is not None and filename is not None
            util.write_dot_file(self._p2_game_solution.solution(), path=path, filename=filename)

        return p2_game_solution

    def sr_edges(self):
        # SR_E = {e \in E | e.source is in Win2(G, F) AND e is SR(G2)} | {e \in E | e.source is in Win1(G, F)}
        sr_edges = set()
        for u, v, k in self._solution.edges():
            if self._base_game_solution.node_winner(u) == 2:
                if ((self._solution["turn"][u] == 1) or (
                        self._solution["turn"][u] == 2 and self._p2_game_solution.edge_winner(u, v, k) == 2)):
                    sr_edges.add((u, v, k))

                #     sr_edges = {
                #     (u, v, k)
                #     for u, v, k in self._solution.edges()
                #         if self._base_game_solution.node_winner(u) == 2 and (
                #                 (self._solution["turn"][u] == 1) or
                #                 (self._solution["turn"][u] == 2 and self._p2_game_solution.edge_winner(u, v, k) == 2)
                #         )
                # }
                # | {
                #     (u, v, k) for u, v, k in self._solution.edges()
                #     if self._base_game_solution.node_winner(u) == 1
                # }

        return sr_edges

    def _construct_hypergame(self, base_game_solution, sr_edges):
        hypergame_graph = ggraph.Graph()
        hypergame_graph["is_deterministic"] = False
        hypergame_graph["is_turn_based"] = False
        hypergame_graph["is_probabilistic"] = True

        # Add nodes and store map of hgame <-> graph nodes.
        num_nodes = len(base_game_solution.winning_nodes(2))
        hg_nodes = hypergame_graph.add_nodes(num_nodes)
        base_hg_node_map = dict(zip(base_game_solution.winning_nodes(2), hg_nodes))
        hg_base_node_map = dict(zip(hg_nodes, base_game_solution.winning_nodes(2)))

        # Add edges (key map is maintained)
        hg_base_edge_map = dict()
        base_hg_edge_map = dict()
        for u, v, k in sr_edges:

            if u in base_game_solution.final():
                hg_uid = base_hg_node_map[u]
                hg_key = hypergame_graph.add_edge(hg_uid, hg_uid)
                base_hg_edge_map[u, v, k] = (hg_uid, hg_uid, hg_key)
                hg_base_edge_map[hg_uid, hg_uid, hg_key] = (u, v, k)
            else:
                hg_uid = base_hg_node_map[u]
                hg_vid = base_hg_node_map[v]
                hg_key = hypergame_graph.add_edge(hg_uid, hg_vid)
                base_hg_edge_map[u, v, k] = (hg_uid, hg_vid, hg_key)
                hg_base_edge_map[hg_uid, hg_vid, hg_key] = (u, v, k)

        # Ensure no spurious state was added while adding edges
        if hypergame_graph.number_of_nodes() != num_nodes:
            raise RuntimeError("Number of nodes changed while adding edges.")

        # Add state property
        hypergame_graph.create_node_property("state", None)
        for uid in hypergame_graph.nodes():
            hypergame_graph["state"][uid] = self._solution["state"][hg_base_node_map[uid]]

        # Add turn property
        hypergame_graph.create_node_property("turn", None)
        for uid in hypergame_graph.nodes():
            hypergame_graph["turn"][uid] = self._solution["turn"][hg_base_node_map[uid]]

        # Add final property
        hypergame_graph.create_node_property("final", False)
        for uid in self._fakes | self._traps:
            hypergame_graph["final"][base_hg_node_map[uid]] = True

        return hypergame_graph, base_hg_node_map, hg_base_node_map, base_hg_edge_map, hg_base_edge_map

    def _save_debug_output(self):
        util.write_dot_file(self._base_game_solution.solution(), path=self._path, filename=f"{self._filename}_base_game")
        util.write_dot_file(self._p2_game_solution.solution(), path=self._path, filename=f"{self._filename}_p2_game")
        util.write_dot_file(self._hgame_solution.solution(), path=self._path, filename=f"{self._filename}_hgame")

    def save_svg(self, path, filename, **kwargs):
        # Generate DOT file.
        fpath = os.path.join(path, f"{filename}.dot")
        with open(fpath, 'w') as file:
            contents = list()
            contents.append("digraph G {\noverlap=scale;\n")

            for node in self._solution.nodes():
                node_properties = {
                    # "shape": 'circle' if self._solution['turn'][node] == 1 else 'box',
                    "width": 2,
                    "height": 2,
                    "style": kwargs.get("node_style", "solid"),
                    "label": self._solution['state'][node],
                    "peripheries": '2' if self._solution['final'][node] else '1',
                }

                if node in self._traps | self._fakes:
                    node_properties["shape"] = "diamond"
                else:
                    node_properties["shape"] = 'circle' if self._solution['turn'][node] == 1 else 'box'

                if "node_winner" in self._solution.node_properties:
                    if self._base_game_solution.node_winner(node) == 1:
                        node_properties |= {"color": 'blue'}
                    # elif self._solution['node_winner'][self._base2hg_nodes[node]] == 1:
                    elif self._hgame_solution.node_winner(self._base2hg_nodes[node]) == 1:
                        node_properties |= {"color": 'green'}
                    else:
                        node_properties |= {"color": 'red'}

                contents.append(
                    f"N{node} [" + ", ".join(f'{k}="{v}"' for k, v in node_properties.items()) + "];\n"
                )

                for uid, vid, key in self._solution.out_edges(node):
                    edge_properties = {
                        "label": self._solution["input"][uid, vid, key] if kwargs.get("show_actions", False) else ""
                    }
                    # if "edge_winner" in self._solution.edge_properties:
                    #     if self._solution['edge_winner'][uid, vid, key] == 1:
                    #         edge_properties |= {"color": 'blue'}
                    #     elif self._solution['edge_winner'][uid, vid, key] == 2:
                    #         edge_properties |= {"color": 'red'}
                    #     else:
                    #         edge_properties |= {"color": 'black'}

                    contents.append(
                        f"N{uid} -> N{vid} [" + ", ".join(f'{k}="{v}"' for k, v in edge_properties.items()) + "];\n"
                    )

            contents.append("}")
            file.writelines(contents)

        # Generate SVG
        g = pygraphviz.AGraph(fpath)
        if self._solution.number_of_nodes() + self._solution.number_of_edges() > 200:
            logger.warning(f"Graph size is larger than 200, using Force-Directed Layout. Generating PNG instead of SVG.")
            g.layout('sfdp')
            path = os.path.join(path, f"{filename}.png")
            g.draw(path=path, format='png')
        else:
            g.layout('dot')
            path = os.path.join(path, f"{filename}.svg")
            g.draw(path=path, format='svg')

    def vod(self):
        return self._vod


class DecoyAllocator(models.Solver):
    def __init__(self,
                 game_graph: ggraph.Graph,
                 num_traps: int,
                 num_fakes: int,
                 candidates=None,
                 algo="greedy",
                 debug=False,
                 **kwargs):
        super(DecoyAllocator, self).__init__(game_graph)
        # Assertions
        assert algo.lower() in ["greedy", "enumerative"], "algo can be either 'greedy' or 'enumerative'."
        assert num_fakes >= 0 and num_traps >= 0, "num_traps and num_fakes should be non-negative integers."

        # Game parameters
        self._num_traps = num_traps
        self._num_fakes = num_fakes
        self._debug = debug
        self._algorithm = algo
        if candidates is None:
            candidates = dict()
            for node in self._solution.nodes():
                candidates[node] = {node}
        self._candidates = candidates

        # Output parameters
        self._vod = 0
        self._dswin = None
        self._traps = set()
        self._fakes = set()
        self._base_game_solution = kwargs.get("base_game_solution", None)

        # Directory setup
        self._path = kwargs.get("path", "out/")
        self._filename = kwargs.get("filename", "dswin")

    def solve_base_game(self, save_svg=False, path=None, filename=None):
        base_game_solution = SWinReach(self._solution, player=2, path=self._path, filename=f"{self._filename}_base_game",
                                       tqdm_disable=True)
        base_game_solution.solve()
        self._base_game_solution = base_game_solution

        if save_svg:
            assert path is not None and filename is not None
            util.write_dot_file(self._base_game_solution.solution(), path=path, filename=filename)

        return self._base_game_solution

    def solve(self):
        # Solve base game
        if self._base_game_solution is None:
            self.solve_base_game()
            logger.info("DecoyAllocator: Base game solved...")

        # Place decoys
        if self._algorithm == "greedy":
            self._solve_greedy()
        else:
            self._solve_enumerative()

    def _solve_enumerative(self):
        pass

    def _solve_greedy(self):
        # Terminate if no decoys are to be placed
        if self._num_fakes + self._num_traps == 0:
            self._hgame = self._solution
            self._hgame_sol = DSWinReach(
                game_graph=self._graph,
                traps=set(),
                fakes=set(),
                debug=self._debug,
                path=self._path,
                filename=self._filename
            )
            self._hgame_sol.solve()
            self._is_solved = True
            return

        # Initialize traps and fakes as empty set
        traps = set()
        fakes = set()

        # Identify candidate decoys
        win2 = set(self._base_game_solution.winning_nodes(player=2))
        final = set(self._base_game_solution.final())
        candidate2nodes = dict()
        for candidate in self._candidates:
            potential_finals = set.intersection(self._candidates[candidate], win2) - final
            if len(potential_finals) > 0:
                candidate2nodes[candidate] = potential_finals
        logger.info(f"Candidate Decoys: {set(candidate2nodes.keys())} ")

        # 1. ALLOCATE FAKES
        intermediate_vod_fakes = []
        while self._num_fakes - len(fakes) > 0:
            # Collect potential states that can be allocated as next fake
            potential_decoys = set(candidate2nodes.keys()) - fakes
            logger.info(f"Identified {len(potential_decoys)} potential candidates for {len(fakes) + 1}-th fake target.")
            if len(potential_decoys) == 0:
                break

            iteration_vod_map = dict().fromkeys(potential_decoys)
            best_fake = None
            best_vod = 0.0
            fake_nodes = set.union(set(), *[candidate2nodes[fake] for fake in fakes])
            for candidate in potential_decoys:
                dswin = DSWinReach(self._solution, traps=set(), fakes=fake_nodes | candidate2nodes[candidate], path=self._path)
                dswin.solve(skip_solution=True)
                iteration_vod_map[candidate] = dswin.vod()
                if dswin.vod() > best_vod:
                    best_vod = dswin.vod()
                    best_fake = candidate
                logger.info(
                    f"Explored Candidate {candidate} for fake no. {len(fakes) + 1}: fakes={fakes} and traps={traps}. VoD: {dswin.vod()}")

            intermediate_vod_fakes.append(iteration_vod_map)
            fakes.add(best_fake)
            logger.info(f"Selected {best_fake} for fake no. {len(fakes)}: fakes={fakes} and traps={traps}. Resulting VoD: {best_vod}")

        # 2. ALLOCATE TRAPS
        intermediate_vod_traps = []
        while self._num_traps - len(traps) > 0:
            # Collect potential states that can be allocated as next fake
            potential_decoys = set(candidate2nodes.keys()) - fakes - traps
            logger.info(f"Identified {len(potential_decoys)} potential candidates for {len(fakes) + 1}-th trap.")
            if len(potential_decoys) == 0:
                break

            iteration_vod_map = dict().fromkeys(potential_decoys)
            best_trap = None
            best_vod = 0.0
            fake_nodes = set.union(set(), *[candidate2nodes[fake] for fake in fakes])
            trap_nodes = set.union(set(), *[candidate2nodes[trap] for trap in traps])
            for candidate in potential_decoys:
                dswin = DSWinReach(self._solution, traps=trap_nodes | candidate2nodes[candidate], fakes=fake_nodes, path=self._path)
                dswin.solve(skip_solution=True)
                iteration_vod_map[candidate] = dswin.vod()
                if dswin.vod() > best_vod:
                    best_vod = dswin.vod()
                    best_trap = candidate
                logger.info(
                    f"Explored Candidate {candidate} for trap no. {len(traps) + 1}: fakes={fakes} and traps={traps}. VoD: {dswin.vod()}")

            intermediate_vod_traps.append(iteration_vod_map)
            traps.add(best_trap)
            logger.info(f"Selected {best_trap} for trap no. {len(traps)}: fakes={fakes} and traps={traps}. Resulting VoD: {best_vod}")

        with open(f"{self._path}/vod_fakes.pkl", "wb") as file:
            pickle.dump(intermediate_vod_fakes, file)
        with open(f"{self._path}/vod_traps.pkl", "wb") as file:
            pickle.dump(intermediate_vod_traps, file)

        # # Mark node and edge winners
        # fake_nodes = set.union(set(), *[candidate2nodes[fake] for fake in fakes])
        # trap_nodes = set.union(set(), *[candidate2nodes[trap] for trap in traps])
        # self._dswin = DSWinReach(game_graph=self._solution, traps=trap_nodes, fakes=fake_nodes,
        #                          debug=self._debug, path=self._path, filename=f"{self._filename}_optimal")
        # self._dswin.solve()
        # self._vod = self._dswin.vod()
        # self._solution["node_winner"].update(self._dswin._solution["node_winner"])
        # self._solution["edge_winner"].update(self._dswin._solution["edge_winner"])
        # logger.info(f"Best decoy allocation configuration: {traps=}, {fakes=} with VoD={self._dswin.vod()}")

        self._is_solved = True

    def _solve_greedy_asw(self):
        # Terminate if no decoys are to be placed
        if self._num_fakes + self._num_traps == 0:
            self._hgame = self._solution
            self._hgame_sol = DASWinReach(
                game_graph=self._graph,
                traps=set(),
                fakes=set(),
                debug=self._debug,
                path=self._path,
                filename=self._filename
            )
            self._hgame_sol.solve()
            self._is_solved = True
            return

        # Initialize traps and fakes as empty set
        traps = set()
        fakes = set()

        # Identify candidate decoys
        win2 = set(self._base_game_solution.winning_nodes(player=2))
        final = set(self._base_game_solution.final())
        candidate2nodes = dict()
        for candidate in self._candidates:
            potential_finals = set.intersection(self._candidates[candidate], win2) - final
            if len(potential_finals) > 0:
                candidate2nodes[candidate] = potential_finals
        logger.info(f"Candidate Decoys: {set(candidate2nodes.keys())} ")

        # 1. ALLOCATE FAKES
        intermediate_vod_fakes = []
        while self._num_fakes - len(fakes) > 0:
            # Collect potential states that can be allocated as next fake
            potential_decoys = set(candidate2nodes.keys()) - fakes
            logger.info(f"Identified {len(potential_decoys)} potential candidates for {len(fakes) + 1}-th fake target.")
            if len(potential_decoys) == 0:
                break

            iteration_vod_map = dict().fromkeys(potential_decoys)
            best_fake = None
            best_vod = 0.0
            fake_nodes = set.union(set(), *[candidate2nodes[fake] for fake in fakes])
            for candidate in potential_decoys:
                daswin = DASWinReach(self._solution, traps=set(), fakes=fake_nodes | candidate2nodes[candidate], path=self._path)
                daswin.solve(skip_solution=True)
                iteration_vod_map[candidate] = daswin.vod()
                if daswin.vod() > best_vod:
                    best_vod = daswin.vod()
                    best_fake = candidate
                logger.info(
                    f"Explored Candidate {candidate} for fake no. {len(fakes) + 1}: fakes={fakes} and traps={traps}. VoD: {daswin.vod()}")

            intermediate_vod_fakes.append(iteration_vod_map)
            fakes.add(best_fake)
            logger.info(f"Selected {best_fake} for fake no. {len(fakes)}: fakes={fakes} and traps={traps}. Resulting VoD: {best_vod}")

        # 2. ALLOCATE TRAPS
        intermediate_vod_traps = []
        while self._num_traps - len(traps) > 0:
            # Collect potential states that can be allocated as next fake
            potential_decoys = set(candidate2nodes.keys()) - fakes - traps
            logger.info(f"Identified {len(potential_decoys)} potential candidates for {len(fakes) + 1}-th trap.")
            if len(potential_decoys) == 0:
                break

            iteration_vod_map = dict().fromkeys(potential_decoys)
            best_trap = None
            best_vod = 0.0
            fake_nodes = set.union(set(), *[candidate2nodes[fake] for fake in fakes])
            trap_nodes = set.union(set(), *[candidate2nodes[trap] for trap in traps])
            for candidate in potential_decoys:
                daswin = DASWinReach(self._solution, traps=trap_nodes | candidate2nodes[candidate], fakes=fake_nodes, path=self._path)
                daswin.solve(skip_solution=True)
                iteration_vod_map[candidate] = daswin.vod()
                if daswin.vod() > best_vod:
                    best_vod = daswin.vod()
                    best_trap = candidate
                logger.info(
                    f"Explored Candidate {candidate} for trap no. {len(traps) + 1}: fakes={fakes} and traps={traps}. VoD: {daswin.vod()}")

            intermediate_vod_traps.append(iteration_vod_map)
            traps.add(best_trap)
            logger.info(f"Selected {best_trap} for trap no. {len(traps)}: fakes={fakes} and traps={traps}. Resulting VoD: {best_vod}")

        with open(f"{self._path}/vod_fakes.pkl", "wb") as file:
            pickle.dump(intermediate_vod_fakes, file)
        with open(f"{self._path}/vod_traps.pkl", "wb") as file:
            pickle.dump(intermediate_vod_traps, file)

        # # Mark node and edge winners
        # fake_nodes = set.union(set(), *[candidate2nodes[fake] for fake in fakes])
        # trap_nodes = set.union(set(), *[candidate2nodes[trap] for trap in traps])
        # self._dswin = DSWinReach(game_graph=self._solution, traps=trap_nodes, fakes=fake_nodes,
        #                          debug=self._debug, path=self._path, filename=f"{self._filename}_optimal")
        # self._dswin.solve()
        # self._vod = self._dswin.vod()
        # self._solution["node_winner"].update(self._dswin._solution["node_winner"])
        # self._solution["edge_winner"].update(self._dswin._solution["edge_winner"])
        # logger.info(f"Best decoy allocation configuration: {traps=}, {fakes=} with VoD={self._dswin.vod()}")

        self._is_solved = True

    def save_svg(self, path, filename, **kwargs):
        self._dswin.save_svg(path, filename, **kwargs)

    def save_dot(self, path, filename, **kwargs):
        # Generate DOT file.
        fpath = os.path.join(path, f"{filename}.dot")
        with open(fpath, 'w') as file:
            contents = list()
            contents.append("digraph G {\n")

            for node in self._solution.nodes():
                node_properties = {
                    # "shape": 'circle' if self._solution['turn'][node] == 1 else 'box',
                    "style": "filled",
                    "label": self._solution['state'][node],
                    "peripheries": '2' if self._solution['final'][node] else '1',
                }

                if node in self._traps | self._fakes:
                    node_properties["shape"] = "diamond"
                else:
                    node_properties["shape"] = 'circle' if self._solution['turn'][node] == 1 else 'box'

                if "node_winner" in self._solution.node_properties:
                    if self._base_game_solution.node_winner(node) == 1:
                        node_properties |= {"color": 'blue'}
                    # elif self._solution['node_winner'][self._base2hg_nodes[node]] == 1:
                    elif self._dswin.node_winner(node) == 1:
                        node_properties |= {"color": 'green'}
                    else:
                        node_properties |= {"color": 'red'}

                contents.append(
                    f"N{node} [" + ", ".join(f'{k}="{v}"' for k, v in node_properties.items()) + "];\n"
                )

                for uid, vid, key in self._solution.out_edges(node):
                    edge_properties = {
                        "label": self._solution["input"][uid, vid, key] if kwargs.get("no_actions", False) else ""
                    }
                    # if "edge_winner" in self._solution.edge_properties:
                    #     if self._solution['edge_winner'][uid, vid, key] == 1:
                    #         edge_properties |= {"color": 'blue'}
                    #     elif self._solution['edge_winner'][uid, vid, key] == 2:
                    #         edge_properties |= {"color": 'red'}
                    #     else:
                    #         edge_properties |= {"color": 'black'}

                    contents.append(
                        f"N{uid} -> N{vid} [" + ", ".join(f'{k}="{v}"' for k, v in edge_properties.items()) + "];\n"
                    )

            contents.append("}")
            file.writelines(contents)

    def save_pickle(self, path, filename):
        with open(os.path.join(path, f"{filename}.pkl"), "wb") as file:
            pickle.dump(self.solution(), file)

    def vod(self):
        return self._vod


if __name__ == '__main__':
    with logger.catch():
        # Construct game
        import ggsolver.honeypot_allocation.game_generator as gen
        import random

        random.seed(0)
        game = gen.Hybrid(num_nodes=12, max_out_degree=3)
        game_graph = game.graphify()

        # # Manually set traps, fakes and solve for DSWinReach
        # fdir = "out_t4_f0"
        # win = DSWinReach(game_graph, traps={197, 72, 29, 54}, fakes=set(), debug=True)
        # win.solve()
        # win.save_svg(fdir, filename="colored_graph")
        # logger.info(f"VOD: {win._vod}")

        # Allocate greedy traps and fakes
        fdir = "out_t0_f2"
        alloc = DecoyAllocator(game_graph, num_traps=0, num_fakes=2, debug=True, path=fdir)
        alloc.solve()
        # alloc.save_pickle(fdir, filename="dswin_sol_graph")
        # alloc.save_dot(fdir, filename="colored_graph")
