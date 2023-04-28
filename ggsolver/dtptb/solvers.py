import datetime
import os
import pathlib
import subprocess
from loguru import logger
from tqdm import tqdm
from networkx.drawing.nx_agraph import read_dot

import ggsolver


class SWinReach(ggsolver.Solver):
    def __init__(self, graph, **kwargs):
        """
        :param graph:
        :param kwargs: Supported keyword arguments.
            - solver: (str) ["ggsolver", "pgsolver"]
            - player: (int) 1 or 2.
            - final: override graph["final"]
            - verbosity: (int) 0: no logs, 1: critical warn/err, 2: tqdm + debug logs
            - directory: directory for intermediate output files
            - filename: file name for intermediate output files
        """
        super(SWinReach, self).__init__(graph, **kwargs)
        assert graph["is_deterministic"], f"SWinReach expects input graph to be deterministic."
        assert graph["is_turn_based"], f"SWinReach expects input graph to be turn-based."

        # Parameters
        self._solver = kwargs.get("solver", "ggsolver")
        self._player = kwargs.get("player", 1)
        self._turn = self._solution["turn"]
        if "final" in kwargs:
            self._final = graph["final"]  # Expects nodes (not states).
        else:
            self._final = {self.state2node(st) for st in graph["final"]}

        # Solver specific properties
        #  1. Rank of state
        if "rank" in self._solution.node_properties:
            self._rank = self._solution.make_property_local("rank")
            self._rank.clear()
        else:
            self._rank = self._solution.create_np("rank", float("inf"))

        #  2. File paths for intermediate files
        self._dir = kwargs.get("directory", "out/")
        self._filename = kwargs.get("filename", f'pgzlk_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}')
        if not pathlib.Path(self._dir).exists():
            pathlib.Path(self._dir).mkdir()

        # Logging properties (0: no logs, 1: critical warn/err, 2: tqdm + debug logs)
        self._verbosity = kwargs.get("verbosity", 1)

    def reset(self):
        """ Resets the solver to initial state. """
        super(SWinReach, self).reset()
        self._rank.clear()
        self._is_solved = False

    def solve(self):
        if self._solver == "ggsolver":
            self._solve_ggsolver()
        elif self._solver == "pgsolver":
            self._solve_pgsolver()
        else:
            raise ValueError(f"Solver: {self._solver} is not supported.")

    def _solve_ggsolver(self, force=False):
        if self._is_solved and not force:
            logger.warning("SWinReach was already solved. Did not re-solve it.")
            return

        # Reset solver
        self.reset()

        # Set up data structures
        rank = 0
        win = set(self._final)
        for node in win:
            self._rank[node] = rank

        # Recursive algorithm for reachability
        with tqdm(total=self._solution.number_of_nodes(),
                  desc="Running recursive algorithm",
                  disable=False if self._verbosity < 2 else True) as progress_bar:
            while True:
                # Explore one-step back
                predecessors = (st for uid in win for st in self._solution.predecessors(uid))
                c_pre = {uid for uid in predecessors if self._turn[uid] == self._player}
                uc_pre = {uid for uid in predecessors
                          if self._turn != self._player and set(self._solution.successors(uid)).issubset(win)}

                # Termination condition
                if len(c_pre) == 0 and len(uc_pre) == 0:
                    break

                # Update rank
                rank += 1
                for uid in c_pre | uc_pre:
                    self._rank[uid] = rank

                # Update winning region
                win = win | c_pre | uc_pre

                # Update progress_bar
                progress_bar.update(1)

        # Post-processing
        for node in tqdm(self._solution.nodes(),
                         desc="Postprocessing SWinReach() nodes",
                         disable=False if self._verbosity < 2 else True):
            # Mark node winner
            self._node_winner[node] = self._player if node in win else (1 if self._player == 2 else 2)
            # Mark edge winner
            for u, v, k in self._solution.out_edges(node):
                self._edge_winner[u, v, k] = self._player \
                    if self._rank[u] > self._rank[v] or u in self._final else (1 if self._player == 2 else 2)

        # Mark the game is solved
        self._is_solved = True

    def _solve_pgsolver(self, force=False):
        if self._is_solved and not force:
            logger.warning("SWinReach was already solved. Did not re-solve it.")
            return

        try:
            # Invoke PGSolver using command-line tool to solve the game.
            if self._verbosity > 0:
                logger.info(f"Invoking PGSolver... This may take a few minutes.")
            self._run_pgsolver()
            if self._verbosity > 0:
                logger.info(f"PGSolver completed execution.")

            # Process PGSolver output to mark node, edge winners
            #   (PGSolver generates dot file and console output. The following code uses dot)
            # Read the DOT file as a networkx graph
            dot_graph = read_dot(os.path.join(self._dir, f"{self._filename}.dot"))
            self._process_pgsolver_dot(dot_graph)

            # Mark the game is solved
            self._is_solved = True
        except Exception as err:
            logger.error(f"dtptb.pgsolver.SWinReach.solve:: {err}")

    def _run_pgsolver(self):
        # Construct PGSolver input
        fpath_gm = os.path.join(self._dir, f"{self._filename}.gm")
        self._gen_pgsolver_input(fpath_gm)

        # Run PGSolver
        fpath_dot = os.path.join(self._dir, f"{self._filename}.dot")
        pgsolver_output = \
            subprocess.run(['/pgsolver/bin/pgsolver',  # command
                            '-global', 'recursive',  # solver (=zielonka)
                            '-d', fpath_dot,  # generate dot form of solution
                            '--printsolonly',  # print solution option of PGSolver
                            fpath_gm  # input file
                            ],
                           stdout=subprocess.PIPE)

        # Save output of PGSolver
        fpath_out = os.path.join(self._dir, f"{self._filename}.out")
        with open(fpath_out, "w") as file:
            file.write(pgsolver_output.stdout.decode())

    def _gen_pgsolver_input(self, fpath):
        # Header
        game_graph = f"parity {self._graph.number_of_nodes() - 1};"

        # Add node specifications
        for uid in self._solution.nodes():
            # Get priority of node
            priority = 2 if uid in self._final else 1
            # Get turn of player. When solving for P2's reachability, swap player because PGSolver only solves for first player
            turn = 0 if self._turn[uid] == 1 else 1
            if self._player == 2:
                turn = 0 if turn == 1 else 1
            # Get successors (Empty successors is not allowed, pgsolver gives parsing error).
            successors = set(self._solution.successors(uid))
            if len(successors) == 0:
                raise ValueError(f"Game is not `complete`: successors({self._solution['state'][uid]}) is empty.")
            successors_str = ",".join(list(map(str, self._graph.successors(uid))))
            # Get state name
            state = self._graph["state"][uid]

            # Compose a line for PGSolver input
            game_graph += f'\n{uid} {priority} {turn} {successors_str} "{state}";'
            game_graph += "\n" + f"{uid} {2 if uid in self._final else 1} {0 if self._turn[uid] == 1 else 1} " + \
                          ",".join(list(map(str, self._graph.successors(uid)))) + f' "{self._graph["state"][uid]}";'

        # Save file to given path
        with open(fpath, "w") as file:
            file.write(game_graph)

    def _process_pgsolver_dot(self, dot_graph):
        # Iterate over nodes to extract information.
        for node, data in tqdm(dot_graph.nodes(data=True), total=dot_graph.number_of_nodes(),
                               desc="Extracting node winners from PGSolver solution."):
            # Get node id
            uid = int(node[1:])

            # Mark node winner
            if data["color"] == 'green':
                self._node_winner[uid] = 1
            else:  # if data["color"] == 'red':
                self._node_winner[uid] = 2

        # Iterate over edges to extract information
        for uid, vid, key in tqdm(self._solution.edges(), total=self.graph().number_of_edges(),
                                  desc="Extracting edge winners from PGSolver solution."):
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
                    self._edge_winner[uid, vid, key] = 1 if self._player == 1 else 2
                else:
                    self._edge_winner[uid, vid, key] = 2 if self._player == 1 else 1

            # If uid is P2 state, any black edge is losing.
            # Programmer's Note: PGSolver (as far as I understand) only determines "a" strategy
            #   for P2 to win from its winning state. If more winning edges exist at a P2 win node,
            #   then they may be black. In this case, we use permissive strategy to determine their winner.
            else:  # self._graph["turn"][uid] == 2:
                if data["color"] == "red":
                    self._edge_winner[uid, vid, key] = 2 if self._player == 1 else 1
                else:
                    if self._node_winner[vid] == 2:
                        self._edge_winner[uid, vid, key] = 2 if self._player == 1 else 1
                    else:
                        self._edge_winner[uid, vid, key] = 1 if self._player == 1 else 2
