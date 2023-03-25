import ggsolver.dtptb.pgsolver as pgsolver

import ggsolver.models as models
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


class JobstmannGame(models.Game):
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
    game = JobstmannGame(final={1, 2, 4})
    graph = game.graphify()
    win = pgsolver.SWinReach(graph)
    win.solve()
