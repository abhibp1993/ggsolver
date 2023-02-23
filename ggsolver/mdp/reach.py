import random
# from ggsolver.models import Solver
import ggsolver.models as models
from tqdm import tqdm


class ASWinReach(models.Solver):
    def __init__(self, graph, final=None, player=1, **kwargs):
        """
        Instantiates a sure winning reachability game solver.

        :param graph: (Graph instance)
        :param final: (iterable) A list/tuple/set of final nodes in graph.
        :param player: (int) Either 1 or 2.
        :param kwargs: SureWinReach accepts no keyword arguments.
        """
        super(ASWinReach, self).__init__(graph, **kwargs)
        self._player = player
        self._final = set(final) if final is not None else {n for n in graph.nodes() if self._graph["final"][n] == 0}
        self._strategy_graph = None

    def solve(self):
        """
        Alg. 45 from Principles of Model Checking.
        Using the same variable names as Alg. 45.
        """
        # Initialize algorithm variables
        graph = self._solution
        b = self._final

        # Make B absorbing
        for uid in b:
            for _, vid, key in graph.out_edges(uid):
                graph.hide_edge(uid, vid, key)

        # Compute the set of nodes disconnected from B
        disconnected = self.disconnected(graph, b)
        set_u = {s for s in graph.nodes() if s in disconnected}

        while True:
            set_r = set_u.copy()
            while len(set_r) > 0:
                u = set_r.pop()

                for t, a in self.pre(graph, u):
                    if t in set_u:
                        continue
                    self.remove_act(graph, t, a)
                    if len(graph.successors(t)) == 0:
                        set_r.add(t)
                        set_u.add(t)
                graph.hide_node(u)
            disconnected = self.disconnected(graph, b)
            set_u = {s for s in set(graph.nodes()) - set_u if s in disconnected}
            if len(set_u) == 0:
                break

        # Process node, edge winners
        for uid in tqdm(self._solution.nodes(), desc="Processing node, edge winners..."):
            self._node_winner[uid] = 1 if self._solution.is_node_visible(uid) else 3
            out_edges = self._solution.out_edges(uid)
            winning_acts = {self._solution["input"][uid, vid, key]
                            for _, vid, key in out_edges if self._solution.is_edge_visible(uid, vid, key)}
            for _, vid, key in out_edges:
                self._edge_winner[uid, vid, key] = 1 if self._solution["input"][uid, vid, key] in winning_acts else 3

    @staticmethod
    def disconnected(graph, sources):
        reachable_nodes = graph.reverse_bfs(sources)
        return set(graph.visible_nodes()) - reachable_nodes

    def pre(self, graph, vid):
        if graph.has_node(vid):
            return {(uid, graph["input"][uid, vid, key]) for uid, _, key in graph.in_edges(vid)}
        return set()

    def remove_act(self, graph, uid, act):
        for _, vid, key in graph.out_edges(uid):
            if graph["input"][uid, vid, key] == act:
                # print(f"\tHiding {uid}, {act}, edge:{uid, vid, key}")
                graph.hide_edge(uid, vid, key)


class PWinReach(models.Solver):
    def __init__(self, graph, final=None, player=1, **kwargs):
        """
        Instantiates a sure winning reachability game solver.

        :param graph: (Graph instance)
        :param final: (iterable) A list/tuple/set of final nodes in graph.
        :param player: (int) Either 1 or 2.
        :param kwargs: SureWinReach accepts no keyword arguments.
        """
        super(PWinReach, self).__init__(graph, **kwargs)
        self._player = player
        self._final = set(final) if final is not None else {n for n in graph.nodes() if self._graph["final"][n] == 0}
        self._strategy_graph = None

    def solve(self):
        # Reset the solver
        self.reset()

        # Get final states
        final = self._final

        with tqdm(total=self._solution.number_of_nodes()) as progress_bar:
            # Identify the set of nodes from which a final state can be reached (i.e., there exists a path in graph)
            progress_bar.set_description("Running reverse BFS...")
            reachable_nodes = self._solution.reverse_bfs(final)

            # Hide the nodes in MDP.
            progress_bar.set_description("Marking node, edge winners...")
            for uid in self._solution.nodes():
                progress_bar.update(1)
                self._node_winner[uid] = 1 if uid in reachable_nodes else 3
                out_edges = self._solution.out_edges(uid)
                winning_acts = {self._solution["input"][uid, vid, key]
                                for _, vid, key in out_edges if vid not in reachable_nodes}
                for _, vid, key in out_edges:
                    self._edge_winner[uid, vid, key] = 1 if self._solution["input"][uid, vid, key] in winning_acts else 3

        # Mark the game as solved.
        self._is_solved = True
