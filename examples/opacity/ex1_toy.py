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
    Toy problem in paper, Fig. 1.
    """

    def states(self):
        return list(range(7))

    def turn(self, state):
        if state in [0, 2, 6]:
            return 1
        else:
            return 2

    def actions(self):
        return ["a1", "a2", "a3", "b1", "b2"]

    def delta(self, state, act):
        # [(0, 1), (0, 4), (0, 5), (1, 2), (1, 6), (2, 3), (4, 6), (5, 2), (5, 6), (6, 3), (3, 3)]
        trans_dict = {
            0: {"a1": 5, "a2": 1, "a3": 4},
            1: {"b1": 6, "b2": 2},
            2: {"a1": 3, "a2": 3, "a3": 3},
            3: {"a1": 3, "a2": 3, "a3": 3, "b1": 3, "b2": 3},
            4: {"b1": 6, "b2": 6},
            5: {"b1": 2, "b2": 6},
            6: {"a1": 3, "a2": 3, "a3": 3}
        }
        try:
            return trans_dict[state][act]
        except KeyError:
            return None

    def atoms(self):
        return ["p1", "p2", "p3"]

    def label(self, state):
        if state == 5:
            return ["p1"]
        elif state == 6:
            return ["p2"]
        elif state == 3:
            return ["p3"]
        else:
            return []

    def formula(self):
        return logic.ltl.ScLTL("Fp1 || Fp2 && Fp3", atoms=self.atoms())

    def attacker_observation(self, state, act, next_state):
        obs_dict = {
            0: {"a1": {5: "o1"}, "a2": {1: "o2"}, "a3": {4: "o2"}},
            1: {"b1": {6: "o3"}, "b2": {2: "o3"}},
            2: {"a1": {3: "o4"}, "a2": {3: "o4"}, "a3": {3: "o4"}},
            3: {"a1": {3: "o5"}, "a2": {3: "o5"}, "a3": {3: "o5"}, "b1": {3: "o5"}, "b2": {3: "o5"}},
            4: {"b1": {6: "o3"}, "b2": {6: "o3"}},
            5: {"b1": {2: "o6"}, "b2": {6: "o6"}},
            6: {"a1": {3: "o4"}, "a2": {3: "o4"}, "a3": {3: "o4"}}
        }
        try:
            return obs_dict[state][act][next_state]
        except KeyError:
            return None

        # if (state in [1, 4]) and (next_state in [1, 4]):
        #     return "1a4" + "1a4"
        # elif (state in [2, 5, 6]) and (next_state in [1, 4]):
        #     return "2ah" + "1a4"
        # elif (state in [1, 4]) and (next_state in [2, 5, 6]):
        #     return "1a4" + "2ah"
        # elif (state in [2, 5, 6]) and (next_state in [2, 5, 6]):
        #     return "2ah" + "2ah"
        # else:
        #     return str(state) + str(next_state)


def solve(game: mod_opacity.BeliefGame):
    """
    Solver for Reach-Avoid Game.
    :return:
    """
    # Graphify the game
    game_graph = game.graphify(pointed=True)
    print("grphify done.")

    # Define a reachability solver (see dtptb.solvers.SWinReach)
    swin_reach = dtptb.SWinReach(game_graph)
    print("Swin_reach created")

    # Define reachability solver for P2 (see Thm. 2)
    final = set()
    for uid in game_graph.nodes():
        if game.final_p2(game_graph["state"][uid]):
            final.add(game_graph["state"][uid])

    swin_reach_p2 = dtptb.SWinReach(game_graph, final=final)
    print("P2's SWinReach object created")

    # Solve the safety game.
    swin_reach.solve()
    print("SWIN reach solved..")

    # Solve the safety game.
    swin_reach_p2.solve()
    print("P2's SWinReach object solved..")

    print(f"{swin_reach.winning_states(1)=}")
    print(f"{swin_reach_p2.winning_states(1)=}")

    print(f"{len(swin_reach.winning_states(1))=}")
    print(f"{len(swin_reach_p2.winning_states(1))=}")
    print(f"{len(game_graph.nodes()) = }")

    # Return solution to reachability game.
    return swin_reach


if __name__ == "__main__":
    # Instantiate MyGame here
    game = MyGame()
    game.initialize(0)
    aut = game.formula().translate()
    graph = game.graphify(pointed=True)
    graph.to_png("graph.png", nlabel=["state"], elabel=["input", "attacker_observation"])

    belief_game = mod_opacity.BeliefGame(game, aut)
    belief_game.initialize(belief_game.init_state())
    graph = belief_game.graphify(pointed=True)
    graph.to_png("belief_graph.png", nlabel=["state"], elabel=["input"])
    solve(belief_game)

    # Analyze the output (see API for models.Solver)#
