"""
Implement toy problem.
Refer to dtptb/examples folder.
"""

import models as mod_opacity
import ggsolver.logic as logic
# import ggsolver.graph as ggraph
import ggsolver.dtptb as dtptb
import logging
logging.basicConfig(level=logging.DEBUG)


class MyGame(mod_opacity.Arena):
    """
    # TODO. Implement this for your example.
    """
    def states(self):
        return list(range(7))

    def turn(self, state):
        if state in [0, 2, 6]:
            return 1
        else:
            return 2

    def actions(self):
        return [(0,1),(0,4),(0,5),(1,2),(1,6),(2,3),(4,6),(5,2),(5,6),(6,3),(3,3)]

    def delta(self, state, act):
        if state == act[0]:
            return act[1]
        return None

    def atoms(self):
        return ["h1", "h2", "p3"]

    def label(self, state):
        if state == 5:
            return ["h1"]
        elif state == 5:
            return ["h2"]
        elif state == 3:
            return ["p3"]
        else:
            return []

    def formula(self):
        return logic.ltl.ScLTL("Fh1 || Fh2 && Fp3")

    def attacker_observation(self, state, act, next_state):
        if (state in [1,4]) and (next_state in [1,4]):
            return "1a4" + "1a4"
        elif (state in [2,5,6]) and (next_state in [1,4]):
            return "2ah" + "1a4"
        elif (state in [1,4]) and (next_state in [2,5,6]):
            return "1a4" + "2ah"
        elif (state in [2,5,6]) and (next_state in [2,5,6]):
            return "2ah" + "2ah"
        else:
            return str(state) + str(next_state)


def solve(game: mod_opacity.BeliefGame):
    """
    Solver for Reach-Avoid Game.
    :return:
    """
    # Graphify the game
    graph = game.graphify(pointed=True)
    print("grphify done.")

    # Define a reachability solver (see dtptb.solvers.SWinReach)
    swin_reach = dtptb.SWinReach(graph)
    print("Swin_reach created")

    # Solve the safety game.
    swin_reach.solve()
    print("SWIN rECH solved..")

    print(f"{swin_reach.win_region(1)=}")

    # Return solution to reachability game.
    return swin_reach


if __name__ == "__main__":
    # Instantiate MyGame here
    game = MyGame()
    game.initialize(0)
    aut = game.formula().translate()

    belief_game = mod_opacity.BeliefGame(game, aut)
    belief_game.initialize(belief_game.init_state())
    solve(belief_game)

    # Analyze the output (see API for models.Solver)#