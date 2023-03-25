"""
Parity game solver using PGSolver (https://github.com/tcsprojects/pgsolver)
"""

import logging
import subprocess
from functools import reduce
from tqdm import tqdm
from pprint import pprint

import ggsolver.graph as mod_graph
import ggsolver.util as util
import ggsolver.models as models

logger = logging.getLogger(__name__)


class SWinReach(models.Solver):
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

        super(SWinReach, self).__init__(graph, **kwargs)
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
