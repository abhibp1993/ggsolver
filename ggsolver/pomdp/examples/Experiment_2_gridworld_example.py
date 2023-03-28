"""
Experiment 2. (gridworld example 1)
"""
import itertools

from ggsolver.pomdp.models import ActivePOMDP, ProductWithDFA, OpacityEnforcingGame
from ggsolver.pomdp.reach import ASWinReach
import ggsolver.logic as logic

if __name__ == '__main__':

    states_in_grid = [f"s{i}" for i in range(25)]
    actions_in_grid = ['N', 'S', 'E', 'W']

    transition_dict = dict()
    for st in states_in_grid:
        temp_set = set()
        temp_dict = dict()
        for act in actions_in_grid:
            if act == 'N':
                temp = int(st[1]) + 5
                if st[0] + str(temp) in states_in_grid:
                    temp_set.add(st[0] + str(temp))
                else:
                    temp_set.add(st)

                temp = int(st[1]) + 1
                if st[0] + str(temp) in states_in_grid:
                    temp_set.add(st[0] + str(temp))
                else:
                    temp_set.add(st)

                temp = int(st[1]) - 1
                if st[0] + str(temp) in states_in_grid:
                    temp_set.add(st[0] + str(temp))
                else:
                    temp_set.add(st)


            elif act == 'S':
                temp = int(st[1]) - 5
                if st[0] + str(temp) in states_in_grid:
                    temp_set.add(st[0] + str(temp))
                else:
                    temp_set.add(st)

                temp = int(st[1]) + 1
                if st[0] + str(temp) in states_in_grid:
                    temp_set.add(st[0] + str(temp))
                else:
                    temp_set.add(st)

                temp = int(st[1]) - 1
                if st[0] + str(temp) in states_in_grid:
                    temp_set.add(st[0] + str(temp))
                else:
                    temp_set.add(st)

            elif act == 'E':
                temp = int(st[1]) + 5
                if st[0] + str(temp) in states_in_grid:
                    temp_set.add(st[0] + str(temp))
                else:
                    temp_set.add(st)

                temp = int(st[1]) + 1
                if st[0] + str(temp) in states_in_grid:
                    temp_set.add(st[0] + str(temp))
                else:
                    temp_set.add(st)

                temp = int(st[1]) - 5
                if st[0] + str(temp) in states_in_grid:
                    temp_set.add(st[0] + str(temp))
                else:
                    temp_set.add(st)

            else:
                temp = int(st[1]) + 4
                if st[0] + str(temp) in states_in_grid:
                    temp_set.add(st[0] + str(temp))
                else:
                    temp_set.add(st)

                temp = int(st[1]) - 4
                if st[0] + str(temp) in states_in_grid:
                    temp_set.add(st[0] + str(temp))
                else:
                    temp_set.add(st)

                temp = int(st[1]) - 1
                if st[0] + str(temp) in states_in_grid:
                    temp_set.add(st[0] + str(temp))
                else:
                    temp_set.add(st)

            temp_dict[act] = list(temp_set)

        transition_dict[st] = temp_dict


    def sensor_queries(sensors_secured, sensors_unsecured):
        unified_sensors = set(sensors_secured).union(set(sensors_unsecured))
        sensor_queries_dict = dict()
        completed_set = list()
        item = "1"

        for sen1, sen2 in itertools.product(unified_sensors, unified_sensors):
            if sen1 != sen2 and {sen1, sen2} not in completed_set:
                sensor_queries_dict[item] = [sen1, sen2]
                item = str(int(item) + 1)
                completed_set.append({sen1, sen2})

        return sensor_queries_dict


    M = ActivePOMDP(
        states=states_in_grid,
        actions=actions_in_grid,
        trans_dict=transition_dict,
        init_state="s0",
        atoms=["A", "B", "C", "E"],
        label={
            "s0": [],
            "s1": [],
            "s2": [],
            "s3": [],
            "s4": [],
            "s5": [],
            "s6": [],
            "s7": ["E"],
            "s8": ["A"],
            "s9": [],
            "s10": [],
            "s11": ["C"],
            "s12": [],
            "s13": [],
            "s14": [],
            "s15": [],
            "s16": [],
            "s17": [],
            "s18": [],
            "s19": [],
            "s20": [],
            "s21": [],
            "s22": ["B"],
            "s23": [],
            "s24": [],
        },
        # final=["s11"],
        sensors={
            "Y": ["s4", "s17"],
            "NG": ["s9"],
            "B": ["s6", "s8"],
            "R": ["s7"],
            "P": ["s11", "s13"],
            "GR": ["s12", "s16"],
            "PE": ["s10", "s18", "s19"],
            "G": ["s21", "s22"]
        },
        secured_sensors=["Y"],
        sensors_unsecured=["NG", "B", "R", "P", "GR", "PE", "G"],
        # sensor_query={
        #     "1": ["Y", "NG"],
        #     "2": ["NG", "B"],
        #     "3": ["B", "R"],
        #     "4": ["A", "D"],
        #     "5": ["A", "C"],
        #     "6": ["B", "D"],
        #     "7": ["A", "E"],
        #     "8": ["B", "E"],
        #     "9": ["C", "E"],
        #     "10": ["D", "E"],
        # },
        sensor_query=sensor_queries(["Y"], ["NG", "B", "R", "P", "GR", "PE", "G"]),

        init_observation=["s0"]

    )

    # E = s7, A = s8 , B = s22 , C = s11

    aut = logic.ltl.LTL("G(!E) & (!(B|C) U(A & F(B|C)))",
                        atoms=["A", "B", "C", "D"]).translate()

    # atoms = ["A", "B", "C", "E"]
    # aut = logic.automata.DFA(
    #     states=[0, 1, 2, 3],
    #     atoms=atoms,
    #     trans_dict={
    #         0: {"!s7": 0, "s7": 3},
    #         1: {"!s7 & !s8 & !s11 & !s22": 1,
    #             "(!s7 & s8 & s11) | (!s7 & s8 & s22)": 0,
    #             "!s7 & s8 & !s11 & !s22": 2,
    #             "s7 | (!s8 & s11) | (!s8 & s22)": 3},
    #         2: {"(!s7 & s11) | (!s7 & s22)": 0,
    #             "!s7 & !s11 & !s22": 2, "s7": 3},
    #         3: {"1": 3},
    #     },
    #     init_state=[1],
    #     final=[0]
    # )

    # aut_graph = aut.graphify()
    #
    # aut_graph.to_png("automata_experiment_2.png", nlabel=["state"], elabel=["input"])

    prod_POMDP_M = ProductWithDFA(M, aut)
    prod_POMDP_M.initialize(M.init_state)

    prod_POMDP_M_graph = prod_POMDP_M.graphify(pointed=True)
    # prod_POMDP_M_graph.to_png("product_pomdp_M.png", nlabel=["state"], elabel=["input"])

    opacity_game_G = OpacityEnforcingGame(prod_POMDP_M_graph)
    #
    opacity_game_G.initialize(prod_POMDP_M.init_state())
    opacity_game_G_graph = opacity_game_G.graphify(pointed=True)
    opacity_game_G_graph.to_png("opacity_enforcing_game.png", nlabel=["state"], elabel=["input"])


