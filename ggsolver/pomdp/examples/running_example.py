"""
Running example from paper.
"""
from ggsolver.pomdp.models import ActivePOMDP, ProductWithDFA, OpacityEnforcingGame
from ggsolver.pomdp.reach import ASWinReach
import sys
import ggsolver.logic as logic
sys.path.append('/home/ggsolver/')

if __name__ == '__main__':
    M = ActivePOMDP(
        states=[f"s{i}" for i in range(5)],
        actions=['a', 'b'],
        trans_dict={
            # "s0": {'a': ["s1", "s2"], 'b': ["s1", "s2"]},
            "s0": {'a': ["s1", "s2"], 'b': []},
            "s1": {'a': ["s3"], 'b': ["s4"]},
            "s2": {'a': ["s4"], 'b': ["s3"]},
            "s3": {'a': ["s3"], 'b': ["s3"]},
            # "s4": {'a': ["s4"], 'b': ["s4"]},
            "s4": {'a': [], 'b': []},

        },
        init_state="s0",
        atoms=["s0", "s1", "s2", "s3", "s4"],
        label={
            "s0": ["s0"],
            "s1": ["s1"],
            "s2": ["s2"],
            "s3": ["s3"],
            "s4": ["s4"],

        },
        final=["s4"],
        sensors={
            "A": ["s1", "s2"],
            "B": ["s2"],
            "C": ["s3", "s4"],
            "D": ["s1", "s2", "s3"],
            },
        secured_sensors=["B"],
        sensors_unsecured=["A", "C", "D"],
        sensor_query={
            "1": ["A", "B"],
            "2": ["B", "C"],
            "3": ["C", "D"],
            "4": ["A", "D"],
            "5": ["A", "C"],
            "6": ["B", "D"],
                      },

        init_observation=["s0"]

    )

    # pomdp_M_graph = M.graphify()
    # pomdp_M_graph.to_png("pomdp_M.png", nlabel=["state"], elabel=["input"])



    # testing the spot automata generation.
    # aut = logic.ltl.ScLTL("F(s4)").translate()
    aut = logic.ltl.ScLTL("F(s4)", atoms=['s0', 's1', 's2', 's3', 's4']).translate()
    # aut_graph = aut.graphify()
    # aut_graph.to_png("automata.png", nlabel=["state"], elabel=["input"])

    # # testing productPOMDP
    prod_POMDP_M = ProductWithDFA(M, aut)
    # # s0, q0 = prod_POMDP_M.init_state()
    prod_POMDP_M.initialize(M.init_state)
    # prod_POMDP_M_graph = prod_POMDP_M.graphify(pointed=True)
    # prod_POMDP_M_graph.to_png("product_pomdp_M.png", nlabel=["state"], elabel=["input"])


    # # testing OpacityEnforcingGame.
    opacity_game_G = OpacityEnforcingGame(prod_POMDP_M)
    #
    opacity_game_G.initialize(prod_POMDP_M.init_state())
    opacity_game_G_graph = opacity_game_G.graphify(pointed=True)
    # opacity_game_G_graph.to_png("opacity_enforcing_game.png", nlabel=["state"], elabel=["input"])

    # testing the solver.
    win = ASWinReach(opacity_game_G_graph)
    win.solve()
    print("********************** ASW Region ***********************************")
    print(win.win_region(1))

    print("*************** Winning Actions **************************************")
    for st in win.win_region(1):
        print("State: ", st)
        print("Winning actions: ", win.win_acts(st))

