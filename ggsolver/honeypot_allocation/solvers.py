import os
import typing

import pygraphviz

import ggsolver.graph as ggraph
import ggsolver.models as models
from ggsolver.dtptb.dtptb_reach import SWinReach
from ggsolver.honeypot_allocation import util
from typing import Union
from loguru import logger

MAX_COMBINATIONS = 100000


class DSWinReach(models.Solver):
    def __init__(self, game_graph: ggraph.Graph, traps: Union[list, set, tuple], fakes: Union[list, set, tuple], debug=False):
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

    def solve(self):
        # 1. Solve game. (V, E, F)
        base_game_solution = self.solve_base_game()

        # 2. Construct and solve G2. (V, E, F U Y)
        p2_game_solution, true_final = self.solve_p2_game()

        # 3. SR_E = {e \in E | e is subjectively rationalizable in G2}
        sr_edges = {
            (u, v, k) for u, v, k in self._solution.edges()
            if base_game_solution.node_winner(u) == 2 and (
                    (self._solution["turn"][u] == 1) or
                    (self._solution["turn"][u] == 2 and p2_game_solution.edge_winner(u, v, k) == 2)
            )
        }

        # 4. Construct H. (Win2, SR_E, X U Y)
        hgame_graph, base2hg_nodes, hg2base_nodes, base2hg_edges, hg2base_edges = \
            self._construct_hypergame(base_game_solution, sr_edges)
        hgame_solution = SWinReach(hgame_graph, player=1, filename="hgame")
        hgame_solution.solve()
        self._hypergame_graph = hgame_graph

        # 5. Compute VoD
        dswin1 = set(hgame_solution.winning_nodes(player=1))
        win2 = set(base_game_solution.winning_nodes(player=2))
        self._vod = len(set.intersection({hg2base_nodes[node] for node in dswin1}, win2)) / len(win2 - true_final)

        # 6. Fix P1 and P2 strategies. Mark node and edge winners
        sources = {base2hg_nodes[node] for node in self._fakes | self._traps}
        reachable_nodes = set()
        if len(sources) > 0:
            hg_reachable_nodes = hgame_graph.reverse_bfs(sources=sources)
            reachable_nodes = {hg2base_nodes[node] for node in hg_reachable_nodes}

        for node in self._solution.nodes():
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

    def solve_base_game(self, save_svg=False, path=None, filename=None):
        base_game_solution = SWinReach(self._solution, player=2, filename="base_game")
        base_game_solution.solve()
        self._base_game_solution = base_game_solution

        if save_svg:
            assert path is not None and filename is not None
            util.write_dot_file(self._base_game_solution.solution(), path=path, filename=filename)

        return self._base_game_solution

    def solve_p2_game(self, save_svg=False, path=None, filename=None):
        true_final = {uid for uid in self._solution.nodes() if self.graph()["final"][uid]}
        p2_final = true_final | self._fakes
        p2_game_solution = SWinReach(self._solution, player=2, final=p2_final, filename="p2_game")
        p2_game_solution.solve()
        self._p2_game_solution = p2_game_solution

        if save_svg:
            assert path is not None and filename is not None
            util.write_dot_file(self._p2_game_solution.solution(), path=path, filename=filename)

        return p2_game_solution, true_final

    def sr_edges(self):
        # SR_E = {e \in E | e.source is in Win2(G, F) AND e is SR(G2)} | {e \in E | e.source is in Win1(G, F)}
        sr_edges = {
                       (u, v, k) for u, v, k in self._solution.edges()
                       if self._base_game_solution.node_winner(u) == 2 and (
                    (self._solution["turn"][u] == 1) or
                    (self._solution["turn"][u] == 2 and self._p2_game_solution.edge_winner(u, v, k) == 2)
            )
                   } | {
                       (u, v, k) for u, v, k in self._solution.edges()
                       if self._base_game_solution.node_winner(u) == 1
                   }

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
            hg_uid = base_hg_node_map[u]
            hg_vid = base_hg_node_map[v]
            hg_key = hypergame_graph.add_edge(hg_uid, hg_vid)
            base_hg_edge_map[u, v, k] = (hg_uid, hg_vid, hg_key)
            hg_base_edge_map[hg_uid, hg_vid, hg_key] = (u, v, k)

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
        util.write_dot_file(self._base_game_solution.solution(), path="out/", filename="base_game")
        util.write_dot_file(self._p2_game_solution.solution(), path="out/", filename="p2_game")
        util.write_dot_file(self._hgame_solution.solution(), path="out/", filename="hgame")

    def save_svg(self, path, filename, **kwargs):
        # Generate DOT file.
        fpath = os.path.join(path, f"{filename}.dot")
        with open(fpath, 'w') as file:
            contents = list()
            contents.append("digraph G {\n")

            for node in self._solution.nodes():
                node_properties = {
                    # "shape": 'circle' if self._solution['turn'][node] == 1 else 'box',
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
                        node_properties |= {"color": 'orange'}

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

        # Generate SVG
        g = pygraphviz.AGraph(fpath)
        g.layout('dot')
        path = os.path.join(path, f"{filename}.svg")
        g.draw(path=path, format='svg')


if __name__ == '__main__':
    with logger.catch():
        # Construct game
        import ggsolver.honeypot_allocation.game_generator as gen
        import random
        random.seed(50)
        game = gen.Hybrid(num_nodes=10, max_out_degree=3)
        game_graph = game.graphify()

        # Allocate traps, fakes and solve for DSWinReach
        win = DSWinReach(game_graph, traps={5}, fakes=set(), debug=True)
        win.solve()
        win.save_svg("out/", filename="colored_graph")
        logger.info(f"VOD: {win._vod}")
