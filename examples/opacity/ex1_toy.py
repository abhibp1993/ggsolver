"""
Implement toy problem.

Refer to dtptb/examples folder.
"""

import examples.opacity.models as mod_opacity
import ggsolver.graph as ggraph


class MyGame(mod_opacity.ReachabilityGame):
    """
    # TODO. Implement this for your example.
    """
    def states(self):
        pass

    def turn(self, state):
        pass

    def actions(self):
        pass

    def delta(self, state, act):
        pass

    def atoms(self):
        pass

    def label(self, state):
        pass

    def formula(self):
        pass

    def attacker_observation(self, state, act):
        pass


def solve(game: mod_opacity.BeliefGame):
    """
    Solver for Reach-Avoid Game.
    :return:
    """
    # Graphify the game
    graph = None

    # Define a safety solver. (see dtptb.solvers.SWinSafe)
    swin_safety = None

    # Solve the safety game.
    swin_safety.solve()

    # Construct new graph using solution of safety game
    solution = swin_safety.solution()
    hidden_nodes = []       # Use swin_safety.node_winner function
    hidden_edges = []       # Use swin_safety.winning_edges function
    safe_graph = ggraph.SubGraph(graph=solution, hidden_nodes=hidden_nodes, hidden_edges=hidden_edges)

    # Define a reachability solver (see dtptb.solvers.SWinReach)
    swin_reach = None

    # Solve the safety game.
    swin_reach.solve()

    # Return solution to reachability game.
    return swin_reach


def main():
    # Instantiate MyGame here
    # Call solve()
    # Analyze the output (see API for models.Solver)
    pass
