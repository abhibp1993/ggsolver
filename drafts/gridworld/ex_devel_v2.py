import ggsolver.gridworld.pygame_colors as colors
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

def window_on_key_down(sender, args):
    print("test. ok")


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
        resizable=True
    )
    window.register_handler(pygame.MOUSEBUTTONDOWN, window_on_key_down)

    # control1 = Control(name="control1", parent=window, position=(100, 100), size=(100, 100), backcolor=colors.RED1)
    # control2 = Control(name="control2", parent=control1, position=(50, 50), size=(20, 20), backcolor=colors.GREEN,
    #                    anchor=AnchorStyle.CENTER)

    sim = GWSim(graph, window)
    sim.run()
