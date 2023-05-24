"""
Zielonka solver based on dtptb-reach utility.
Currently, uses abhibp1993/dtptb-reach docker image.
# TODO. Merge the utility with ggsolver:devel image.
"""
import ast
import json
import pathlib
import os
import subprocess

import ggsolver.models as models
from loguru import logger
from datetime import datetime
from tqdm import tqdm


class SWinReach(models.Solver):
    def __init__(self, graph, final=None, path="out/", filename=None, save_output=False, **kwargs):
        if not graph["is_deterministic"]:
            logger.warning(f"dtptb.SWinReach expects deterministic game graph. Input parameters: "
                           f"is_deterministic={graph['is_deterministic']}, "
                           f"is_probabilistic={graph['is_probabilistic']}.")

        if not graph["is_turn_based"]:
            logger.warning(f"dtptb.SWinReach expects turn-based game graph. Input parameters: "
                           f"is_turn_based={graph['is_turn_based']}.")

        super(SWinReach, self).__init__(graph, **kwargs)
        self._player = kwargs.get("player", 1)
        if final is None:
            self._final = self._get_final_states()
        else:
            self._solution.create_node_property("final", default=False, overwrite=True)
            for node in final:
                self._solution["final"][node] = True
            self._final = final

        if len(self._final) == 0:
            logger.critical(f"dtptb.SWinReach.__init__(): Final state set is empty.")

        self._turn = self._solution["turn"]
        self._rank = self._solution.create_node_property("rank", default=float("inf"))
        path = pathlib.Path(path)
        self._path = path.absolute()
        if not path.exists():
            path.mkdir()
        self._filename = filename if filename is not None else \
            f'ggzlk_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
        self._save_output = save_output

    def _gen_dtptb_reach_input(self):
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
        json_graph = self._solution.serialize()
        out = dict()
        out["nodes"] = json_graph["nodes"]
        out["turn"] = json_graph["np.turn"]["map"]
        out["final"] = list(self._final)
        out["edges"] = dict()

        edges = {tuple(ast.literal_eval(edge)[0:2]) for edge in json_graph["edges"]}
        count = 0
        for edge in edges:
            out["edges"][f"e{count}"] = edge
            count += 1

        return out

    def _run_dtptb_reach(self):
        # Construct dtptb_reach input (json)
        game_input = self._gen_dtptb_reach_input()

        # Save dtptb_reach input file
        with open(os.path.join(self._path, f"{self._filename}.gm"), "w") as file:
            json.dump(game_input, file, indent=2)

        # Run dtptb_reach
        dtptb_reach_output = \
            subprocess.run(['dtptb-reach',  # command
                            '-p', f'{self._player}',  # player
                            '-o', os.path.join(self._path, f"{self._filename}.solution"),  # generate json form of solution
                            os.path.join(self._path, f"{self._filename}.gm")  # input file
                            ],
                           stdout=subprocess.PIPE)

        # Save output of dtptb_reach
        with open(os.path.join(self._path, f"{self._filename}.out"), "w") as file:
            file.write(dtptb_reach_output.stdout.decode())

    def _process_dtptb_reach_solution(self):
        with open(os.path.join(self._path, f"{self._filename}.solution"), "r") as file:
            json_sol = json.load(file)

        # Mark node winner
        tqdm_disable = True
        if len(json_sol["node_winner"]) > 5000 or len(json_sol["edge_winner"]) > 5000:
            tqdm_disable = False

        for node, winner in tqdm(json_sol["node_winner"].items(),
                                 desc="dtptb-reach:: Processing nodes from solution.",
                                 disable=tqdm_disable):
            uid = int(node)
            self._solution["node_winner"][uid] = winner

        # Mark edge winner
        json_edges = {tuple((u, v)): eid for eid, (u, v) in json_sol["edges"].items()}
        for u, v, k in tqdm(self._solution.edges(),
                            desc="dtptb-reach:: Processing edges from solution.",
                            disable=tqdm_disable):
            if (u, v) in json_edges:
                eid = json_edges[u, v]
                self._solution["edge_winner"][u, v, k] = json_sol["edge_winner"][eid]
        # for eid, edge in tqdm(json_sol["edges"].items()):
        #     for u, v, k in self._solution.out_edges(edge[0]):
        #         if v == edge[1]:
        #             self._solution["edge_winner"][u, v, k] = json_sol["edge_winner"][eid]

    def reset(self):
        """ Resets the solver to initial state. """
        super(SWinReach, self).reset()
        self._rank = self._solution.create_node_property("rank", float("inf"), overwrite=True)
        self._is_solved = False

    def _get_final_states(self):
        """ Determines the final states using "final" property of the input graph. """
        return {uid for uid in self.graph().nodes() if self.graph()["final"][uid]}

    def solve(self):
        # If game is solved, do not resolve it.
        if self._is_solved:
            logger.warning(f"Game is solved. To re-solve, call `reset` before `solve`.")
            return

        # If output is to be saved, save the game graph
        if self._save_output:
            self._graph.save(os.path.join(self._path, f"{self._filename}.ggraph"))

        # If no final states, do not run CLI utility. Declare all states are winning for opponent.
        if len(self._final) == 0:
            self._no_final_solution()
            self._is_solved = True
            return

        # Invoke dtptb-reach CLI utility to generate solution.
        try:
            # Invoke dtptb-reach utility using command-line tool to solve the game.
            self._run_dtptb_reach()

            # Process dtptb-reach utility output to mark node, edge winners
            #   (dtptb-reach utility generates json file and console output. The following code uses json)
            self._process_dtptb_reach_solution()

        except Exception as err:
            logger.exception(f"{err}")

        # If user has not requested to save data, remove it.
        if not self._save_output:
            game_file = os.path.join(self._path, f"{self._filename}.gm")
            out_file = os.path.join(self._path, f"{self._filename}.out")
            dot_file = os.path.join(self._path, f"{self._filename}.dot")
            dot_file = os.path.join(self._path, f"{self._filename}.solution")

            if os.path.exists(game_file):
                os.remove(game_file)

            if os.path.exists(out_file):
                os.remove(out_file)

            if os.path.exists(dot_file):
                os.remove(dot_file)

        # Mark the game to be solved.
        self._is_solved = True

    def _no_final_solution(self):
        player = self._player
        opponent = 2 if player == 1 else 1

        # Mark node winner
        for node in self._solution.nodes():
            self._solution["node_winner"][node] = opponent

        # Mark edge winner
        for edge in self._solution.edges():
            self._solution["edge_winner"][edge] = opponent

    def final(self):
        return self._final