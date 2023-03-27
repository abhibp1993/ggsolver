"""
Parity game solver using PGSolver (https://github.com/tcsprojects/pgsolver)
"""

import logging
import os.path
import pathlib
import subprocess
import datetime
from functools import reduce
from tqdm import tqdm
from pprint import pprint
import networkx as nx
from networkx.drawing.nx_agraph import read_dot

import ggsolver.graph as mod_graph
import ggsolver.util as util
import ggsolver.models as models

logger = logging.getLogger(__name__)


class SWinReach(models.Solver):
    def __init__(self, graph, final=None, path="out/", filename=None, save_output=False, **kwargs):
        if not graph["is_deterministic"]:
            logger.warning(util.ColoredMsg.warn(f"dtptb.SWinReach expects deterministic game graph. Input parameters: "
                                                f"is_deterministic={graph['is_deterministic']}, "
                                                f"is_probabilistic={graph['is_probabilistic']}."))

        if not graph["is_turn_based"]:
            logger.warning(util.ColoredMsg.warn(f"dtptb.SWinReach expects turn-based game graph. Input parameters: "
                                                f"is_turn_based={graph['is_turn_based']}."))

        super(SWinReach, self).__init__(graph, **kwargs)
        self._player = 1  # For PGSolver, we can only solve for P1's reachability.
        self._final = {self.state2node(st) for st in final} if final is not None else self.get_final_states()
        if len(self._final) == 0:
            logging.warning(f"dtptb.SWinReach.__init__(): Final state set is empty.")

        self._turn = self._solution["turn"]
        self._rank = mod_graph.NodePropertyMap(self._solution, default=float("inf"))
        self._solution["rank"] = self._rank
        self._path = path
        if not os.path.exists(self._path):
            os.mkdir(self._path)

        self._filename = filename if filename is not None else \
            f'pgzlk_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
        self._save_output = save_output

    def _gen_pgsolver_input(self):
        """
        Converts `self._graph` to PGSolver input format.

        .. note:: PGSolver does not accept multi-digraphs. It expects a digraph.
                    In DTPTBGame, this is reasonable because how many parallel edges exist between two nodes
                    does not affect the decision whether to mark a node as winning.

        Programmer's notes:

            - PGSolver solves for max parity (even/odd doesn't matter).
                Because SWinReach is concerned with reachability games, two parity values are sufficient.
                We assign parity `2` to final states, remaining states are assigned `1`.

        .. warning:: PGSolver expects node set to be contiguous. That is, nodes must be {0, 1, 2, ..., n - 1}.
                In the case when nodes are not contiguous, special care must be taken. Current this is not
                handled by this function.

        """
        # Header
        game_graph = f"parity {self._graph.number_of_nodes() - 1};"

        # Add node specifications
        for uid in self._solution.nodes():
            game_graph += "\n" + f"{uid} {2 if uid in self._final else 1} {0 if self._turn[uid] == 1 else 1} " + \
                          ",".join(list(map(str, self._graph.successors(uid)))) + f' "{self._graph["state"][uid]}";'

        return game_graph

    def _run_pgsolver(self):
        # Construct PGSolver input
        pg_input = self._gen_pgsolver_input()

        # Save PGSolver input file
        with open(os.path.join(self._path, f"{self._filename}.gm"), "w") as file:
            file.write(pg_input)

        # Run PGSolver
        pgsolver_output = \
            subprocess.run(['/pgsolver/bin/pgsolver',  # command
                            '-global', 'recursive',  # solver (=zielonka)
                            '-d', os.path.join(self._path, f"{self._filename}.dot"),  # generate dot form of solution
                            '--printsolonly',  # print solution option of PGSolver
                            os.path.join(self._path, f"{self._filename}.gm")  # input file
                            ],
                           stdout=subprocess.PIPE)

        # Save output of PGSolver
        with open(os.path.join(self._path, f"{self._filename}.out"), "w") as file:
            file.write(pgsolver_output.stdout.decode())

        # Parse output
        # return self._parse_pgsolver_output()
        # return self._parse_pgsolver_dot()

    def _parse_pgsolver_output(self):

        with open(os.path.join(self._path, f"{self._filename}.out"), "r") as file:
            pgsolver_output = file.readlines()

        # Parse solution
        # # 1. Decode from byte string
        # solution_str = pgsolver_output.stdout.decode()
        # # 2. Create list of lines
        # solution_str = solution_str.split("\n")
        # 3. Ignore first and last lines as they do not contain information about solution
        # solution_str = solution_str[1:-1]
        solution = pgsolver_output[1:-1]
        # 4. Remove any line termination symbols
        solution = map(lambda x: x.replace(";", "").split(" "), solution)
        # 5. Extract winning nodes.
        win1 = set()
        win2 = set()
        for obj in solution:
            if int(obj[1]) == 0:  # P1 wins if node winner=0.
                win1.add(int(obj[0]))
            else:  # P2 wins if node winner=1.
                win2.add(int(obj[0]))
        return win1, win2

    def _process_pgsolver_dot(self):
        # Read the DOT file as a networkx graph
        dot_graph = read_dot(os.path.join(self._path, f"{self._filename}.dot"))

        # Iterate over nodes to extract information.
        for node, data in dot_graph.nodes(data=True):
            # Get node id
            uid = int(node[1:])

            # Mark node winner
            if data["color"] == 'green':
                self._node_winner[uid] = 1
            else:  # if data["color"] == 'red':
                self._node_winner[uid] = 2

            # Mark edge winners
            for _, vid, key in self._solution.out_edges(uid):
                # Programmer's note: dot_graph is a nx.MultiDigraph instance.
                #   Hence, the edges of dot_graph have format: (u, v, k).
                #   The key here may not correspond to that in self._solution.
                #   Such confusion should be avoided.
                # We use 0-th key because the color of all parallel edge between fixed pair of nodes
                # is same for Zielonka's algorithm.
                data = dot_graph.get_edge_data(f"N{uid}", f"N{vid}", 0)

                # If uid is P1 state, any black edge is losing.
                if self._solution["turn"][uid] == 1:
                    if data["color"] == "green":
                        self._edge_winner[uid, vid, key] = 1
                    else:
                        self._edge_winner[uid, vid, key] = 2

                # If uid is P2 state, any black edge is losing.
                # Programmer's Note: PGSolver (as far as I understand) only determines "a" strategy
                #   for P2 to win from its winning state. If more winning edges exist at a P2 win node,
                #   then they may be black. In this case, we use permissive strategy to determine their winner.
                else:  # self._graph["turn"][uid] == 2:
                    if data["color"] == "red":
                        self._edge_winner[uid, vid, key] = 2
                    else:
                        if self._node_winner[vid] == 2:
                            self._edge_winner[uid, vid, key] = 2
                        else:
                            self._edge_winner[uid, vid, key] = 1

    def reset(self):
        """ Resets the solver to initial state. """
        super(SWinReach, self).reset()
        self._rank = mod_graph.NodePropertyMap(self._solution)
        self._is_solved = False

    def get_final_states(self):
        """ Determines the final states using "final" property of the input graph. """
        return {uid for uid in self.graph().nodes() if self.graph()["final"][uid]}

    def solve(self):
        # If game is solved, do not resolve it.
        if self._is_solved:
            logging.warning(
                f"dtptb.pgsolver.SWinReach.solve:: Game is solved. To resolve, call `reset` before `solve`.")
            return

        # If output is to be saved, save the game graph
        if self._save_output:
            self._graph.save(os.path.join(self._path, f"{self._filename}.ggraph"))

        try:
            # Invoke PGSolver using command-line tool to solve the game.
            self._run_pgsolver()

            # Process PGSolver output to mark node, edge winners
            #   (PGSolver generates dot file and console output. The following code uses dot)
            self._process_pgsolver_dot()

        except Exception as err:
            logger.error(f"dtptb.pgsolver.SWinReach.solve:: {err}")

        # If user has not requested to save data, remove it.
        if not self._save_output:
            game_file = os.path.join(self._path, f"{self._filename}.gm")
            out_file = os.path.join(self._path, f"{self._filename}.out")
            dot_file = os.path.join(self._path, f"{self._filename}.dot")

            if os.path.exists(game_file):
                os.remove(game_file)

            if os.path.exists(out_file):
                os.remove(out_file)

            if os.path.exists(dot_file):
                os.remove(dot_file)


class SWinReach2(models.Solver):
    """
    Computes sure winning region for player 1 or player 2 to reach a set of final states in a deterministic
    two-player turn-based game.

    Implements Zielonka's recursive algorithm.

    :param graph: (Graph or SubGraph instance) A graph or subgraph of a deterministic two-player turn-based game.
    :param final: (Iterable) The set of final states. By default, the final states are determined using
        node property "final" of the graph.
    :param player: (int) The player who has the reachability objective.
        Value should be 1 for player 1, and 2 for player 2.
    """

    def __init__(self, graph, final=None, player=1, **kwargs):
        if not graph["is_deterministic"]:
            logger.warning(util.ColoredMsg.warn(f"dtptb.SWinReach expects deterministic game graph. Input parameters: "
                                                f"is_deterministic={graph['is_deterministic']}, "
                                                f"is_probabilistic={graph['is_probabilistic']}."))

        if not graph["is_turn_based"]:
            logger.warning(util.ColoredMsg.warn(f"dtptb.SWinReach expects turn-based game graph. Input parameters: "
                                                f"is_turn_based={graph['is_turn_based']}."))

        super(SWinReach2, self).__init__(graph, **kwargs)
        self._player = player
        self._final = {self.state2node(st) for st in final} if final is not None else self.get_final_states()
        self._turn = self._solution["turn"]
        self._rank = mod_graph.NodePropertyMap(self._solution, default=float("inf"))
        self._solution["rank"] = self._rank

        if len(self._final) == 0:
            logging.warning(f"dtptb.SWinReach.__init__(): Final state set is empty.")

    def reset(self):
        """ Resets the solver to initial state. """
        super(SWinReach, self).reset()
        self._rank = mod_graph.NodePropertyMap(self._solution)
        self._is_solved = False

    def get_final_states(self):
        """ Determines the final states using "final" property of the input graph. """
        return {uid for uid in self.graph().nodes() if self.graph()["final"][uid]}

    def solve(self):
        """ Solves two player reachability game by invoking PGSolver. """
        # Construct PGSolver input
        pg_input = self.to_pgsolver()

        # Invoke pgsolver using command-line tool.
        win1, win2 = self.run_pgsolver(pg_input)
        print(win1, win2)

        # Update properties
        for uid in self._graph.nodes():
            winner = 1 if uid in win1 else 2
            self._node_winner[uid] = winner

    def to_pgsolver(self):
        """
        1. PGSolver does not accept multi-digraphs. It expects a digraph.
            In DTPTBGame, this is reasonable because how many parallel edges exist between two nodes
            does not affect the decision whether to mark a node as winning.
        2.
        """
        # Header
        game_graph = f"parity {self._graph.number_of_nodes() - 1};"

        # Add node specifications
        # TODO. What is node ids are not contiguous?
        # PGSolver solves for max-even parity.
        for uid in self._solution.nodes():
            game_graph += "\n" + f"{uid} {2 if uid in self._final else 1} {0 if self._turn[uid] == 1 else 1} " + \
                          ",".join(list(map(str, self._graph.successors(uid)))) + f' "{self._graph["state"][uid]}";'

        print(game_graph)
        return game_graph

    def run_pgsolver(self, inp):
        """
        inp: pgsolver input file as multiline string.
        """
        with open("tmp.gm", "w") as file:
            file.write(inp)

        pgsolver_output = subprocess.run(['/pgsolver/bin/pgsolver', '-global', 'recursive', '-d', 'graph.dot',
                                          '--printsolonly', "tmp.gm"],
                                         stdout=subprocess.PIPE)

        print("==========================")
        solution_str = pgsolver_output.stdout.decode()
        print(solution_str)

        solution_str = solution_str.split("\n")
        solution_str = solution_str[1:-1]
        # count = 0
        # for i in range(len(solution_str)):
        #     count += 1
        #     if "parity" in solution_str[i]:
        #         break
        # solution_str = solution_str[count:]

        solution_str = map(lambda x: x.replace(";", "").split(" "), solution_str)
        win1 = set()
        win2 = set()
        for obj in solution_str:
            if int(obj[1]) == 0:
                win1.add(int(obj[0]))
            else:
                win2.add(int(obj[0]))
        return win1, win2
