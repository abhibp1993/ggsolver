import itertools
import json

from ggsolver.pbp.safeimp.models import PrefModel, ImprovementMDP
from ggsolver.pbp.safeimp.reachability import SASIReach, SPIReach
from ggsolver.models import register_property
from ggsolver.mdp.models import QualitativeMDP
from ggsolver.gridworld import util

import logging
logging.basicConfig(level=logging.ERROR)


class MDPGridworld(QualitativeMDP):
    GRAPH_PROPERTY = QualitativeMDP.GRAPH_PROPERTY.copy()

    def __init__(self, dim, batt, obstacles, goals, batt_cell, accessibility_trans, restrict_acts=None):
        """
        dim: (ROWS, COLS) of gridworld
        batt: maximum charge of robot battery
            E.g., batt=4 means battery can take values between 0 and 4, inclusive.)
        obstacles: [cell]
        goals: [cell]
        accessibility_trans: {goal: [goal]}

        """
        super(MDPGridworld, self).__init__()

        # Class variables
        self._dim = dim
        self._batt = batt
        self._obs = obstacles
        self._goals = goals
        self._batt_cell = batt_cell
        self._accessibility_trans = accessibility_trans
        self._restrict_acts = restrict_acts if restrict_acts is not None else dict()

        # Helper variable (stochasticity of actions)
        # self._stochasticity = {
        #     "N": ["N", "NE", "NW", "STAY"],
        #     "E": ["E", "NE", "SE", "STAY"],
        #     "S": ["S", "SE", "SW", "STAY"],
        #     "W": ["W", "NW", "SW", "STAY"],
        # }
        self._stochasticity = {
            "N": ["N", "NE", "NW"],  # "STAY"],
            "E": ["E"],  # "STAY"],
            "S": ["S"],  # "STAY"],
            "W": ["W", "N"],  # "STAY"],
        }

    def states(self):
        """
        state = (p1.row: int, p1.col: int, p1.batt: int, accessible:List[Bool])
        """
        accessibility = itertools.product(
            [1],
            [1],
            [1],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1]      # BATTERY ACTIVE OR NOT.
        )
        states = list(itertools.product(
            range(self.dim()[0]),
            range(self.dim()[1]),
            range(self.batt()),
            # PATCH to reduce states. [HARD-CODED]
            # list(itertools.product([0, 1, 2], repeat=len(self.roi())))
            list(accessibility)
        ))
        # states = [st for st in states if st[0:2] not in self._obs]
        return states

    def actions(self):
        return [
            util.GW_ACT_N,
            util.GW_ACT_E,
            util.GW_ACT_S,
            util.GW_ACT_W
        ]

    def delta(self, state, act):
        row, col, batt, accessibility = state

        # Manage battery constraint
        if batt == 0:
            return [state]
        else:
            n_batt = batt - 1

        # Cell transition
        # 1. Handle restricted actions
        if (row, col) in self._restrict_acts:
            if act not in self._restrict_acts[row, col]:
                act = util.GW_ACT_STAY

        # 2. Apply cell transition (stochasticity specialized!)
        # PATCH: HARD CODED.
        next_cell = util.move((row, col), act)
        if next_cell in [(1, 1), (3, 1), (1, 3)]:
            next_cells = [next_cell, util.move(next_cell, util.GW_ACT_N), util.move(next_cell, util.GW_ACT_S)]
        elif next_cell in [(3, 3)]:
            next_cells = [next_cell, util.move(next_cell, util.GW_ACT_N), util.move(next_cell, util.GW_ACT_W), 
                          util.move(next_cell, util.GW_ACT_S)]

        else:
            next_cells = [next_cell]
        next_cells = util.bouncy_wall((row, col), next_cells, self.dim())
        next_cells = util.bouncy_obstacle((row, col), next_cells, self.obs())

        # Apply accessibility modification rules and construct next state
        next_states = set()
        for cell in next_cells:
            n_accessibility = self._update_accessibility(accessibility, cell)
            n_batt = self._batt - 1 if cell == self._batt_cell and n_accessibility[-1] == 1 else n_batt
            next_states.add((cell[0], cell[1], n_batt, tuple(n_accessibility)))

        # Return next states
        return list(next_states)

    @register_property(GRAPH_PROPERTY)
    def dim(self):
        return self._dim

    @register_property(GRAPH_PROPERTY)
    def batt(self):
        return self._batt

    @register_property(GRAPH_PROPERTY)
    def obs(self):
        return self._obs

    @register_property(GRAPH_PROPERTY)
    def roi(self):
        return self._goals

    def _apply_non_det_actions(self, cell, act):
        actions = self._stochasticity[act]
        next_cells = set()
        for a in actions:
            next_cells.add(util.move(cell, a))
        return list(next_cells)

    def _update_accessibility(self, accessibility, cell):
        """
        0, 2: inaccessible
        1: accessible
        """

        # If cell is not an outcome, then no changes to accessibility
        if cell not in outcomes.values():
            return accessibility

        # Initialize new_accessibility tuple
        n_accessibility = list(accessibility)

        # Get index of cell in outcomes
        outcome_i = [key for key, value in outcomes.items() if value == cell][0]

        # Get activate, block lists
        activate = self._accessibility_trans[outcome_i][0]
        # block = self._accessibility_trans[outcome_i][1]

        for j in activate:
            n_accessibility[j] = 1

        # PATCH for battery
        if outcome_i in [0, 1, 2]:
            n_accessibility[-1] = 1

        return tuple(n_accessibility)


if __name__ == '__main__':
    outcomes = {
        0: (0, 0),
        1: (2, 0),
        2: (4, 0),
        3: (2, 4),
        4: (4, 4),
        5: (1, 2),
        # 6: (3, 0),
    }

    obs = [
        (0, 2),
        (1, 0),
        (3, 0),
        (1, 4),
        (3, 4)
    ]

    restrict_acts = {
        (4, 2): ["N", "S"],      # enabled actions in the cell.
        (2, 2): ["N", "S"]       # enabled actions in the cell.
    }

    gw = MDPGridworld(
        dim=(5, 5),
        batt=9,
        obstacles=obs,
        goals=list(outcomes.values()),
        batt_cell=(0, 4),
        restrict_acts=restrict_acts,
        accessibility_trans={
            0: ([3, 4], []),        # A
            1: ([3, 4], []),        # B
            2: ([4, 5], []),        # C
            3: ([], []),            # D
            4: ([], []),            # E
            5: ([], []),            # F
            # 6: ([], []),
        },
    )

    outcome_0 = [st for st in gw.states() if tuple(st[0:2]) == outcomes[0] and st[3][0] == 1]
    outcome_1 = [st for st in gw.states() if tuple(st[0:2]) == outcomes[1] and st[3][1] == 1]
    outcome_2 = [st for st in gw.states() if tuple(st[0:2]) == outcomes[2] and st[3][2] == 1]
    outcome_3 = [st for st in gw.states() if tuple(st[0:2]) == outcomes[3] and st[3][3] == 1]
    outcome_4 = [st for st in gw.states() if tuple(st[0:2]) == outcomes[4] and st[3][4] == 1]
    outcome_5 = [st for st in gw.states() if tuple(st[0:2]) == outcomes[5] and st[3][5] == 1]
    # outcome_6 = [st for st in gw.states() if tuple(st[0:2]) == outcomes[6] and st[3][6] == 1]

    # pprint(outcome_0)
    pref = PrefModel(
        outcomes={
            0: outcome_0,
            1: outcome_1,
            2: outcome_2,
            3: outcome_3,
            4: outcome_4,
            5: outcome_5,
            # 6: outcome_6,
        },
        pref=[
            (3, 0), (4, 0), (3, 1), (4, 1), (4, 2), (5, 2)
        ]
    )

    imdp = ImprovementMDP(gw, pref)
    imdp_graph = imdp.graphify(base_only=True)

    with open("out/mp_outcomes.json", "w") as file:
        json.dump({str(st): str(value) for st, value in imdp._mp_outcomes.items()}, file, indent=2)

    with open("out/outcomes.json", "w") as file:
        json.dump({str(st): str(value) for st, value in imdp._outcomes.items()}, file, indent=2)

    print("Graphify done.")
    # graph.save(fpath="imdp_5_5.model", overwrite=True)
    final_nodes = {node for node in imdp_graph.nodes() if imdp_graph["state"][node][1] == 1}

    sasi = SASIReach(imdp_graph, final=final_nodes)
    sasi.solve()

    with open("out/rank.json", "w") as file:
        json.dump({str(st): str(sasi.rank(st)) for st in imdp.states()}, file, indent=2)

    with open("out/win0.json", "w") as file:
        win = imdp._winning_regions[0]
        json.dump([win.graph()["state"][uid] for uid in win.win1()], file)

    with open("out/win1.json", "w") as file:
        win = imdp._winning_regions[1]
        json.dump([win.graph()["state"][uid] for uid in win.win1()], file)

    with open("out/win2.json", "w") as file:
        win = imdp._winning_regions[2]
        json.dump([win.graph()["state"][uid] for uid in win.win1()], file)

    print(f"level 3 has {len(sasi.win1()[3])} states, i_state=0: ",
          len([imdp_graph["state"][uid] for uid in imdp_graph.nodes()
               if uid in sasi.win1()[3] and imdp_graph["state"][uid][1] == 0]))

    print(f"level 2 has {len(sasi.win1()[2])} states, i_state=0: ",
          len([imdp_graph["state"][uid] for uid in imdp_graph.nodes()
               if uid in sasi.win1()[2] and imdp_graph["state"][uid][1] == 0]))

    print(f"level 1 has {len(sasi.win1()[1])} states, i_state=0: ",
          len([imdp_graph["state"][uid] for uid in imdp_graph.nodes()
               if uid in sasi.win1()[1] and imdp_graph["state"][uid][1] == 0]))

    spi = SPIReach(imdp_graph, final=final_nodes)
    spi.solve()

    print("--------- ")
    print(f"level 3 has {len(spi.win1()[3])} states, i_state=0: ",
          len([imdp_graph["state"][uid] for uid in imdp_graph.nodes()
               if uid in spi.win1()[3] and imdp_graph["state"][uid][1] == 0]))

    print(f"level 2 has {len(spi.win1()[2])} states, i_state=0: ",
          len([imdp_graph["state"][uid] for uid in imdp_graph.nodes()
               if uid in spi.win1()[2] and imdp_graph["state"][uid][1] == 0]))

    print(f"level 1 has {len(spi.win1()[1])} states, i_state=0: ",
          len([imdp_graph["state"][uid] for uid in imdp_graph.nodes()
               if uid in spi.win1()[1] and imdp_graph["state"][uid][1] == 0]))

    # pprint([imdp_graph["state"][uid] for uid in imdp_graph.nodes()
    #            if uid in sasi.win1()[1] and imdp_graph["state"][uid][1] == 0])
    print()
    # import pickle
    # with open("sasi.model", "w") as file:
    #     pickle.dump(sasi, file=file)
    # pprint(sasi.win1())
