import ggsolver.gridworld.color_util as colors
from ggsolver.gridworld.models import *
from ggsolver.models import Game


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


def window_on_key_down(args):
    print(f"Called: {args.sender}.{inspect.stack()[0][3]}")


def control1_on_key_down(args):
    print(f"Called: {args.sender}.{inspect.stack()[0][3]}")


def control2_on_key_down(args):
    print(f"Called: {args.sender}.{inspect.stack()[0][3]}")


if __name__ == '__main__':
    game = JobstmannGame(final={3, 5, 6})
    graph = game.graphify()

    window = Window(
        name="main",
        title="Main Window",
        size=(600, 600),
        backcolor=colors.AQUA,
        frame_rate=5,
        sm_update_rate=2,
        resizable=True,
        on_key_down=window_on_key_down,
    )

    control1 = Control(
        name="control1",
        parent=window,
        position=(100, 100),
        size=(100, 100),
        backcolor=colors.RED1,
        # on_key_down=control1_on_key_down
    )

    control2 = Control(
        name="control2",
        parent=control1,
        position=(50, 50),
        size=(20, 20),
        backcolor=colors.GREEN,
        anchor=DockStyle.CENTER,
        on_key_down=control2_on_key_down

    )

    grid = Grid(
        name="grid1",
        parent=window,
        position=(0, 0),
        size=(200, 200),
        grid_size=(2, 2),
        backcolor=colors.BEIGE,
        anchor=DockStyle.CENTER,
    )

    sim = GWSim(graph, window)
    sim.run()
