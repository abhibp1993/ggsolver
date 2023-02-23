from ggsolver.dtptb.examples import jobstmann
from ggsolver.gridworld.models import *
from ggsolver.models import *
import logging
logging.basicConfig(level=logging.INFO)


class JobstmannGame(Game):
    def __init__(self, final):
        super(JobstmannGame, self).__init__()
        self.param_final = final

    def states(self):
        return list(range(8))

    def actions(self):
        return [(0, 1), (0, 3), (1, 0), (1, 2), (1, 4), (2, 4), (2, 2), (3, 0), (3, 4), (3, 5), (4, 1), (4, 3), (5, 3),
                (5, 6), (6, 6), (6, 7), (7, 0), (7, 3)]

    def delta(self, state, act):
        """
        Return `None` to skip adding an edge.
        """
        if state == act[0]:
            return act[1]
        return None

    def final(self, state):
        return True if state in self.param_final else False

    def turn(self, state):
        if state in [0, 4, 6]:
            return 1
        else:
            return 2


if __name__ == '__main__':
    game = JobstmannGame(final={3, 5, 6})
    graph = game.graphify()
    sm = StateMachine(graph)
    print(f"{list(sm.states())=}")
    print(f"{list(sm.actions())=}")
    sm.initialize(0)

    print()
    print("sm.step_forward((0, 1))")
    sm.step_forward((0, 1))
    print(f"{sm._curr_time_step=}")
    print(f"{sm.curr_state=}")
    print(f"{sm._state_history=}")
    print(f"{sm._action_history=}")

    print()
    print("sm.step_backward(n=1)")
    sm.step_backward(n=1)
    print(f"{sm._curr_time_step=}")
    print(f"{sm.curr_state=}")
    print(f"{sm._state_history=}")
    print(f"{sm._action_history=}")

    print()
    print("sm.step_backward(n=1)")
    sm.step_backward(n=1)
    print(f"{sm._curr_time_step=}")
    print(f"{sm.curr_state=}")
    print(f"{sm._state_history=}")
    print(f"{sm._action_history=}")

    print()
    print("sm.step_forward((0, 2))")
    sm.step_forward((0, 2))
    print(f"{sm._curr_time_step=}")
    print(f"{sm.curr_state=}")
    print(f"{sm._state_history=}")
    print(f"{sm._action_history=}")

    print()
    print("sm.step_backward(n=1)")
    sm.step_backward(n=1)
    print(f"{sm._curr_time_step=}")
    print(f"{sm.curr_state=}")
    print(f"{sm._state_history=}")
    print(f"{sm._action_history=}")

    print()
    print("sm.step_forward((0, 3), override_act=True)")
    sm.step_forward((0, 3), override_act=True)
    print(f"{sm._curr_time_step=}")
    print(f"{sm.curr_state=}")
    print(f"{sm._state_history=}")
    print(f"{sm._action_history=}")

    print()
    print("sm.step_forward((3, 4))")
    sm.step_forward((3, 4))
    print(f"{sm._curr_time_step=}")
    print(f"{sm.curr_state=}")
    print(f"{sm._state_history=}")
    print(f"{sm._action_history=}")

    print()
    print("sm.step_backward(n=3, clear_history=True) --- has no effect.")
    sm.step_backward(n=3, clear_history=True)
    print(f"{sm._curr_time_step=}")
    print(f"{sm.curr_state=}")
    print(f"{sm._state_history=}")
    print(f"{sm._action_history=}")

    print()
    print("sm.step_backward(n=2, clear_history=True)")
    sm.step_backward(n=2, clear_history=True)
    print(f"{sm._curr_time_step=}")
    print(f"{sm.curr_state=}")
    print(f"{sm._state_history=}")
    print(f"{sm._action_history=}")

