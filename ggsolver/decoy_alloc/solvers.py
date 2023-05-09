import multiprocessing

import ggsolver.dtptb as dtptb
import ggsolver.graph as ggraph
import ggsolver.models as models
import concurrent.futures
import os
import math
from itertools import combinations
from ggsolver.dtptb.pgsolver import SWinReach
import loguru

logger = loguru.logger

MAX_COMBINATIONS = 100000


class EnumerativeTrapsAllocator(models.Solver):
    """
    :param graph: graph of hypergame
    """

    def __init__(self, graph: ggraph.Graph,
                 num_decoys: int,
                 max_combinations=MAX_COMBINATIONS,
                 cpu_count=0,
                 directory=None,
                 fname=None,
                 save_output=False
                 ):
        super(EnumerativeTrapsAllocator, self).__init__(graph)
        self.num_decoys = num_decoys
        self.max_combinations = max_combinations
        self.cpu_count = multiprocessing.cpu_count() if cpu_count == "all" else cpu_count
        self.directory = directory
        self.fname = fname
        self._save_output = save_output

        # dict with the decoys, VOD, and solver for the best decoy allocation
        self.deception_dict = None

        self._value_of_deception = self._solution["value_of_deception"] = dict()

    def _multicore_solve(self, decoy_combinations):
        # with concurrent.futures.ProcessPoolExecutor(max_workers=self.cpu_count) as executor:
        #     args = (
        #         (self.graph(), [self._graph["state"][uid] for uid in decoys],
        #          i, "winning_states", self.directory, self.fname)
        #         for i, decoys in enumerate(decoy_combinations)
        #     )
        #     results = executor.map(get_value_of_deception_pair, args)
        #
        #     for result in results:
        #         print(result)
        #
        #     return max(result, key=lambda decoy_set: len(decoy_set["value_of_deception"]))
        raise NotImplementedError("Multicore is not supported due to pickling issues with SubGraph class.")

    def _singlecore_solve(self, decoy_combinations):
        results = []
        for i, decoys in enumerate(decoy_combinations):
            decoys = [self._graph["state"][uid] for uid in decoys]
            # Remove out going edges from decoy states
            # (the hypergame only has out going edges removed from the original final states)
            sub_graph = remove_out_going_final_edges(self.graph(), decoys, self._state2node)
            # Solve the sub_graph
            args = (sub_graph, decoys, i, "winning_states", self.directory, self.fname, self._save_output)
            result = get_value_of_deception_pair(args)
            results.append(result)
            self._value_of_deception[str(result["decoys"])] = result["value_of_deception"]
        logger.debug(f"Solved deceptive planning for {decoys=}.")

        return max(results, key=lambda decoy_set: decoy_set["value_of_deception"])

    def solve(self):
        """
        # FIXME: Not checking node siblings for now. (decoy_subsets/arena_maping or so.)
        :return:
        """
        # Check for computability
        num_combinations = math.comb(self._graph.number_of_nodes(), self.num_decoys)
        if num_combinations > self.max_combinations:
            raise RuntimeError(f"Cannot process more than {self.max_combinations} games.")
        logger.debug(f"Setting up solvers for {num_combinations} games.")

        # Define combinations and extract the final states.
        possible_decoys = set(self._graph.nodes()) - set(
            uid for uid in self.graph().nodes() if self.graph()["final"][uid])
        decoy_combinations = combinations(possible_decoys, self.num_decoys)

        # Solve trivial case where number of potential decoy states is less than number decoys to allocate
        if len(possible_decoys) < self.num_decoys:
            possible_decoy_states = [self._graph["state"][uid] for uid in possible_decoys]
            sub_graph = remove_out_going_final_edges(self.graph(), possible_decoy_states, self._state2node)
            args = (
            sub_graph, possible_decoy_states, 0, "winning_states", self.directory, self.fname, self._save_output)
            self.deception_dict = get_value_of_deception_pair(args)
        # Based on multiprocessing, solve for each decoy placement.
        elif self.cpu_count > 1:
            self.deception_dict = self._multicore_solve(decoy_combinations)
        else:
            self.deception_dict = self._singlecore_solve(decoy_combinations)

        self._edge_winner.update(self.deception_dict["solver"]._edge_winner)
        self._node_winner.update(self.deception_dict["solver"]._node_winner)
        self._solution["vod"] = self.deception_dict["value_of_deception"]
        self._is_solved = True


class GreedyTrapsAllocator(models.Solver):
    def __init__(self, graph: ggraph.Graph,
                 num_decoys: int,
                 arena2states: dict = None,
                 max_combinations=MAX_COMBINATIONS,
                 cpu_count=0,
                 directory=None,
                 fname=None,
                 save_output=False
                 ):
        super(GreedyTrapsAllocator, self).__init__(graph)
        self.num_decoys = num_decoys
        self.arena2states = arena2states
        self.max_combinations = max_combinations
        self.cpu_count = multiprocessing.cpu_count() if cpu_count == "all" else cpu_count
        self.directory = directory
        self.fname = fname
        self._save_output = save_output

        # dict with the decoys, VOD, and solver for the best decoy allocation
        self.deception_dict = None

        self._value_of_deception = self._solution["value_of_deception"] = dict()

    def _multicore_solve(self):
        raise NotImplementedError("Multicore is not supported due to pickling issues with SubGraph class.")

    def _singlecore_solve(self):
        states = set(self._graph["state"][uid] for uid in self._graph.nodes())
        final_states = set(self.graph()["state"][uid] for uid in self.graph().nodes() if self.graph()["final"][uid])
        trap_states = set()
        covered_states = set()
        iter_count = 0
        while len(states - covered_states) > 0 and len(trap_states) < self.num_decoys:
            iter_count += 1
            potential_traps = states - trap_states - final_states
            updated_winning_regions = list()
            # Consider the trivial case with more available decoys than states to potentially make traps
            if len(potential_traps) <= self.num_decoys:
                sub_graph = remove_out_going_final_edges(self.graph(), potential_traps, self._state2node)
                args = (sub_graph, potential_traps, iter_count, "winning_states", self.directory, self.fname,
                        self._save_output)
                return get_value_of_deception_pair(args)

            for potential_trap in potential_traps:
                new_final_states = trap_states.union(set([potential_trap]))
                # Remove out going edges from decoy states
                # (the hypergame only has out going edges removed from the original final states)
                sub_graph = remove_out_going_final_edges(self.graph(), new_final_states, self._state2node)
                # Solve the sub_graph
                args = (sub_graph, new_final_states, iter_count, "winning_states", self.directory, self.fname,
                        self._save_output)
                result = get_value_of_deception_pair(args)
                new_region = {"result": result, "new_trap": potential_trap}
                updated_winning_regions.append(new_region)
            next_trap_set = max(updated_winning_regions,
                                key=lambda decoy_set: decoy_set["result"]["value_of_deception"])
            trap_states.add(next_trap_set["new_trap"])
            covered_states.update(next_trap_set["result"]["solver"].winning_states(1))
        return next_trap_set["result"]

    def _arena_game_solve(self):
        states = set()
        arena_points = set()
        for arena_point, state_list in self.arena2states.items():
            arena_points.add(arena_point)
            for state in state_list:
                states.add(state)

        arena_traps = set()  # set of arena points

        covered_states = set()  # set of states
        trap_states = set()  # set of states

        iter_count = 0
        # Allocate traps
        while len(states - covered_states) > 0 and len(arena_traps) < self.num_decoys:
            iter_count += 1
            print(f"Iteration {iter_count}")

            potential_arena_traps = arena_points - arena_traps
            updated_winning_regions = list()

            for arena_point in potential_arena_traps:
                # the list of final states if this arena point is made into a trap
                new_final_states = list(trap_states) + self.arena2states[arena_point]

                sub_graph = remove_out_going_final_edges(self.graph(), new_final_states, self._state2node)
                args = (sub_graph, new_final_states, iter_count, "winning_states", self.directory, self.fname,
                        self._save_output)
                result = get_value_of_deception_pair(args)
                new_region = {"result": result, "new_arena_trap": arena_point}
                updated_winning_regions.append(new_region)

            next_trap_set = max(updated_winning_regions,
                                key=lambda decoy_set: decoy_set["result"]["value_of_deception"])
            arena_traps.add(next_trap_set["new_arena_trap"])
            trap_states.update(self.arena2states[next_trap_set["new_arena_trap"]])
            covered_states.update(next_trap_set["winning_states"])
        return next_trap_set["result"]

    def solve(self):
        # Check if there is a mapping from arena points to states
        if self.arena2states is not None:
            self.deception_dict = self._arena_game_solve()
        # Based on multiprocessing, solve for each decoy placement.
        elif self.cpu_count > 1:
            self.deception_dict = self._multicore_solve()
        else:
            self.deception_dict = self._singlecore_solve()

        self._edge_winner.update(self.deception_dict["solver"]._edge_winner)
        self._node_winner.update(self.deception_dict["solver"]._node_winner)
        self._solution["vod"] = self.deception_dict["value_of_deception"]
        self._is_solved = True


class EnumerativeFakesAllocator(models.Solver):
    def __init__(self, graph: ggraph.Graph, num_decoys, use_multiprocessing=False):
        pass

    def solve(self):
        pass


class GreedyFakesAllocator(models.Solver):
    def __init__(self, graph: ggraph.Graph,
                 num_fakes: int,
                 arena2states: dict = None,
                 max_combinations=MAX_COMBINATIONS,
                 cpu_count=0,
                 directory=None,
                 fname=None,
                 save_output=False
                 ):
        super(GreedyFakesAllocator, self).__init__(graph)
        self.solver = None
        self.hypergame = None
        self.vod = None
        self.set_of_fakes = None

        self.num_fakes = num_fakes
        # self.arena2states = arena2states
        self.max_combinations = max_combinations
        self.cpu_count = multiprocessing.cpu_count() if cpu_count == "all" else cpu_count
        self.directory = directory
        self.fname = fname
        self._save_output = save_output

        # dict with the decoys, VOD, and solver for the best decoy allocation
        self.deception_dict = None

        self._value_of_deception = self._solution["value_of_deception"] = dict()

    def _singlecore_solve(self):
        states = set(self._graph["state"][uid] for uid in self._graph.nodes())
        final_states = set(self.graph()["state"][uid] for uid in self.graph().nodes() if self.graph()["final"][uid])
        fake_states = set()
        covered_states = set()
        iter_count = 0
        while len(states - covered_states) > 0 and len(fake_states) < self.num_decoys:
            iter_count += 1
            potential_fakes = states - fake_states - final_states
            updated_winning_regions = list()
            # Consider the trivial case with more available decoys than states to potentially make fakes
            if len(potential_fakes) <= self.num_fakes:
                raise NotImplementedError("Solution is trivial (number of fakes to be allocated is <= the number"
                                          " of potential states available to allocate fakes")
                # sub_graph = remove_out_going_final_edges(self.graph(), potential_fakes, self._state2node)
                # args = (sub_graph, potential_fakes, iter_count, "winning_states", self.directory, self.fname,
                #         self._save_output)
                # result = get_value_of_deception_pair(args)
                # set_of_fakes = result["decoys"]
                # vod = result["value_of_deception"]
                # solver = result["result"]["solver"]
                # hypergame = result["hypergame"]

            for potential_fake in potential_fakes:
                iter_count += 1
                p1_final_states = fake_states.union(set([potential_fake]))
                # Construct G(y) (Remove out going edges from decoy states)
                sub_graph = remove_out_going_final_edges(self.graph(), p1_final_states, self._state2node)
                # Solve G(Y) for P2
                fakes_solver = SWinReach(sub_graph, final=p1_final_states, path=self.directory,
                                         save_output=self.save_output,
                                         filename=f"{self.filename}_fake_game_{iter_count}", player=2)
                fakes_solver.solve()
                # Generate hypergame based on subjectively rationalizable actions of P2
                fakes_hypergame = gen_hypergame(sub_graph, fakes_solver)
                # Solve hypergame for P1 with final states Y
                args = (fakes_hypergame, p1_final_states, iter_count, "winning_states", self.directory, self.fname,
                        self._save_output)
                result = get_value_of_deception_pair(args)
                new_region = {"result": result, "new_fake": potential_fake, "hypergame": fakes_hypergame}
                updated_winning_regions.append(new_region)
            next_fake_set = max(updated_winning_regions,
                                key=lambda decoy_set: decoy_set["result"]["value_of_deception"])
            fake_states.add(next_fake_set["new_fake"])
            covered_states.update(next_fake_set["result"]["solver"].winning_states(1))

        set_of_fakes = next_fake_set["result"]["decoys"]
        vod = next_fake_set["result"]["value_of_deception"]
        solver = next_fake_set["result"]["solver"]
        hypergame = next_fake_set["hypergame"]
        return set_of_fakes, vod, hypergame, solver

    def solve(self):
        # Check if there is a mapping from arena points to states
        if self.arena2states is not None:
            raise NotImplementedError()
        else:
            self.set_of_fakes, self.vod, self.hypergame, self.solver = self._singlecore_solve()

            self._edge_winner.update(self.solver._edge_winner)
            self._node_winner.update(self.solver._node_winner)
            self._is_solved = True


class EnumerativeMixedAllocator(models.Solver):
    def __init__(self, graph: ggraph.Graph, num_decoys, use_multiprocessing=False):
        pass

    def solve(self):
        pass


class GreedyMixedAllocator(models.Solver):
    def __init__(self, graph: ggraph.Graph, num_decoys, use_multiprocessing=False):
        pass

    def solve(self):
        pass


def remove_out_going_final_edges(graph: ggraph.Graph, final_states: set, state2node):
    hidden_edges = set()
    out_going_final_edges = list()
    for state in final_states:
        for out_edge in graph.out_edges(state2node[state]):
            out_going_final_edges.append(out_edge)
    hidden_edges.update(out_going_final_edges)
    sub_graph = ggraph.SubGraph(graph, hidden_edges=hidden_edges)
    return sub_graph


# TODO (Note.). The following function is only for traps.
def get_value_of_deception_pair(args):
    """ Returns the (decoy,vod) pair for a given decoy combination"""
    logger.debug(f"{args}")
    graph, decoys, solution_count, metric, directory, f_name, save_output = args

    # Solve new game
    solver = SWinReach(graph, final=decoys, path=directory, save_output=save_output,
                       filename=f"pgzlk_{'_'.join(decoys)}")
    solver.solve()
    logger.info(f"Solved game {f_name}_{solution_count} with {decoys}.")

    if directory is not None and f_name is not None:
        solver.solution().save(os.path.join(directory, f"{f_name}_{solution_count}.solution"), overwrite=True)

    if metric == "winning_states":
        vod = len(solver.winning_states(1)) / \
              (graph.number_of_nodes() - len([uid for uid in graph.nodes() if graph["final"][uid]]))
        pair = {"decoys": decoys, "value_of_deception": vod, "solver": solver}
        return pair
    else:
        raise NotImplementedError


def gen_hypergame(game_graph, swin_game: dtptb.SWinReach):
    """
    :param game_graph: base game graph
    :param swin_game: solution to base game
    :param config:
    :return:
    """
    hidden_nodes = {uid for uid in swin_game.winning_nodes(1)}
    hidden_edges = set()
    for uid in swin_game.winning_nodes(2):
        for _, vid, key in swin_game.solution().out_edges(uid):
            if swin_game.solution()["rank"][vid] >= swin_game.solution()["rank"][uid]:
                hidden_edges.add((uid, vid, key))

    # Remove outgoing edges from final states
    out_going_final_edges = [game_graph.out_edges(state) for state in swin_game.get_final_states()]
    hidden_edges.update(out_going_final_edges[0])
    hgame_graph = ggraph.SubGraph(game_graph, hidden_nodes=hidden_nodes, hidden_edges=hidden_edges)
    return hgame_graph
