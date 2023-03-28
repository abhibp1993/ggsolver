"""
Experiment 1. (border delivery example)
"""
from ggsolver.pomdp.models import ActivePOMDP, ProductWithDFA, OpacityEnforcingGame, OpacityNotEnforcingGame
from ggsolver.pomdp.reach import ASWinReach
# import sys
import ggsolver.logic as logic

# sys.path.append('/home/ggsolver/')

if __name__ == '__main__':
    M = ActivePOMDP(
        states=[f"s{i}" for i in range(12)],
        actions=['d1', 'd2', 'd3', 'd4', 'd5', 'd6'],
        trans_dict={
            "s0": {'d1': ["s1", "s7"], 'd2': [], 'd3': [], 'd4': [], 'd5': [], 'd6': []},
            "s1": {'d1': ["s0", "s6", "s2"], 'd2': [], 'd3': [], 'd4': ["s6", "s2"], 'd5': [], 'd6': []},
            "s2": {'d1': ["s1"], 'd2': [], 'd3': ["s3"], 'd4': ["s5"], 'd5': [], 'd6': []},
            "s3": {'d1': ["s3"], 'd2': ["s3"], 'd3': ["s3"], 'd4': ["s3"], 'd5': ["s3"], 'd6': ["s3"]},
            "s4": {'d1': [], 'd2': [], 'd3': [], 'd4': [], 'd5': ["s11"], 'd6': ["s3"]},
            "s5": {'d1': [], 'd2': ["s10"], 'd3': ["s6"], 'd4': ["s2"], 'd5': [], 'd6': []},
            "s6": {'d1': ["s1", "s7", "s9"], 'd2': [], 'd3': ["s5"], 'd4': ["s9"], 'd5': ["s7"], 'd6': []},
            "s7": {'d1': ["s1", "s6", "s8"], 'd2': [], 'd3': [], 'd4': [], 'd5': ["s8", "s6"], 'd6': []},
            "s8": {'d1': ["s7"], 'd2': [], 'd3': ["s9"], 'd4': [], 'd5': ["s7"], 'd6': []},
            "s9": {'d1': ["s9"], 'd2': ["s9"], 'd3': ["s9"], 'd4': ["s9"], 'd5': ["s9"], 'd6': ["s9"]},
            "s10": {'d1': [], 'd2': [], 'd3': [], 'd4': [], 'd5': ["s9"], 'd6': ["s11"]},
            "s11": {'d1': ["s11"], 'd2': ["s11"], 'd3': ["s11"], 'd4': ["s11"], 'd5': ["s11"], 'd6': ["s11"]},

        },
        init_state="s0",
        atoms=["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11"],
        label={
            "s0": ["s0"],
            "s1": ["s1"],
            "s2": ["s2"],
            "s3": ["s3"],
            "s4": ["s4"],
            "s5": ["s5"],
            "s6": ["s6"],
            "s7": ["s7"],
            "s8": ["s8"],
            "s9": ["s9"],
            "s10": ["s10"],
            "s11": ["s11"],

        },
        final=["s11"],
        sensors={
            "A": ["s10", "s4"],
            "B": ["s4", "s5", "s6"],
            "C": ["s3", "s11", "s9"],
            "D": ["s10", "s4", "s3", "s9", "s7"],
            "E": ["s1", "s2", "s5", "s6", "s8"],
        },
        secured_sensors=["B"],
        sensors_unsecured=["A", "C", "D", "E"],
        sensor_query={
            "1": ["A", "B"],
            "2": ["B", "C"],
            "3": ["C", "D"],
            "4": ["A", "D"],
            "5": ["A", "C"],
            "6": ["B", "D"],
            "7": ["A", "E"],
            "8": ["B", "E"],
            "9": ["C", "E"],
            "10": ["D", "E"],
        },

        init_observation=["s0"]

    )

    aut = logic.ltl.ScLTL("F(s11)",
                          atoms=['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11']).translate()

    prod_POMDP_M = ProductWithDFA(M, aut)
    prod_POMDP_M.initialize(M.init_state)

    opacity_game_G = OpacityEnforcingGame(prod_POMDP_M)

    opacity_game_G.initialize(prod_POMDP_M.init_state())
    opacity_game_G_graph = opacity_game_G.graphify(pointed=True)

    # # Running the solver.
    win = ASWinReach(opacity_game_G_graph)
    win.solve()
    # print("********************** ASW Region ***********************************")
    # print(win.win_region(1))
    #
    # print("*************** Winning Actions **************************************")
    # for st in win.win_region(1):
    #     print("State: ", st)
    #     print("Winning actions: ", win.win_acts(st))

    # Saving a set of winning initial states.
    p1_win_init_states_opacity = set()
    for i in win.win_region(1):
        st, b1, b2 = i
        if type(b1[1]) == int:
            p1_win_init_states_opacity.add(st[0])


    print("******************** Number of ASW with opacity enforcing *******************")
    print(len(p1_win_init_states_opacity))

    # Solving for Opacity not enforcing.

    opacity_not_game_G = OpacityNotEnforcingGame(prod_POMDP_M)

    opacity_not_game_G.initialize(prod_POMDP_M.init_state())
    opacity_not_game_G_graph = opacity_not_game_G.graphify(pointed=True)

    win_not_opacity = ASWinReach(opacity_not_game_G_graph)
    win_not_opacity.solve()

    # Saving set of winning initial states.
    p1_win_init_states_no_opacity = set()


    for j in win_not_opacity.win_region(1):
        stn, bn1, bn2 = j
        if type(bn1[1]) == int:
            p1_win_init_states_no_opacity.add(stn[0])


    print("***************** Number of ASW without enforcing opacity *********************")
    print(len(p1_win_init_states_no_opacity))



